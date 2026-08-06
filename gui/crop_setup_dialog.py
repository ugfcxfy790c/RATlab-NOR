"""
"Crop Videos…" dialog (Video menu) -- walks through videos one at a
time, the same way ObjectSetupDialog walks through videos for object
hitboxes, letting you drag a fixed-size (width x height) crop window
into place per video. Unlike the arena's clicked object positions, the
camera/arena framing isn't guaranteed to repeat exactly across every
recording, so this defaults to per-video positioning rather than one
uniform crop for a whole folder.

Confirming a video (Forward) crops it immediately (a single video, on
the main thread -- brief and bounded, like ObjectSetupDialog's own
per-video frame grabs). A "Use This Position for All Remaining" button
skips the rest of the walkthrough when the framing hasn't moved: it
crops every remaining video in one shot via CropRunner, a separate
process (see crop_worker_process.py for why cropping a whole batch
can't just happen inline here).

Positions are cached per video (video_crop.load_positions/
save_positions, keyed with object_picker.video_key() -- the same keying
convention object coordinates use) at <output_folder>/crop_positions.json,
so reopening this dialog on the same input/output folder resumes rather
than re-asking for videos already cropped.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QMessageBox, QPlainTextEdit, QProgressBar, QApplication,
)

from object_picker import video_key
from object_setup_dialog import _grab_first_frame, _bgr_frame_to_qimage
import video_crop as vc
from crop_runner import CropRunner

_RECT_COLOR = QColor(255, 140, 0)
_RECT_COLOR_BAD = QColor(200, 40, 40)


class _CropCanvas(QWidget):
    """Shows the loaded preview frame with a fixed-size draggable
    rectangle marking the crop window -- same interaction model as
    object_setup_dialog's hitbox canvas (drag, click-to-place, arrow-key
    nudge), just one rectangle instead of two."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.pixmap = None
        self.frame_w = 0
        self.frame_h = 0
        self.crop_w = 0
        self.crop_h = 0
        self.x = 0
        self.y = 0
        self._dragging = False
        self._drag_offset = (0, 0)
        self.on_change = None  # callable(x, y), invoked after any position change

    def set_frame(self, qimage, crop_w, crop_h):
        self.pixmap = QPixmap.fromImage(qimage)
        self.frame_w, self.frame_h = self.pixmap.width(), self.pixmap.height()
        self.setFixedSize(self.pixmap.size())
        self.set_crop_size(crop_w, crop_h)

    def set_crop_size(self, crop_w, crop_h):
        self.crop_w, self.crop_h = crop_w, crop_h
        self.set_position(self.x, self.y)  # re-clamp

    def set_position(self, x, y):
        max_x = max(0, self.frame_w - self.crop_w)
        max_y = max(0, self.frame_h - self.crop_h)
        self.x = min(max(0, x), max_x)
        self.y = min(max(0, y), max_y)
        self.update()
        if self.on_change:
            self.on_change(self.x, self.y)

    def fits(self) -> bool:
        return self.crop_w <= self.frame_w and self.crop_h <= self.frame_h

    def paintEvent(self, event):
        if self.pixmap is None:
            return
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)
        painter.setPen(QPen(_RECT_COLOR if self.fits() else _RECT_COLOR_BAD, 2))
        painter.drawRect(QRectF(self.x, self.y, self.crop_w, self.crop_h))
        painter.end()

    def _in_rect(self, pt: QPointF) -> bool:
        return self.x <= pt.x() <= self.x + self.crop_w and self.y <= pt.y() <= self.y + self.crop_h

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self.pixmap is None:
            return
        self.setFocus()  # so arrow-key nudging works right after a click, not just after Tab
        pt = event.position()
        if self._in_rect(pt):
            self._dragging = True
            self._drag_offset = (pt.x() - self.x, pt.y() - self.y)
        else:
            self.set_position(int(pt.x() - self.crop_w / 2), int(pt.y() - self.crop_h / 2))

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        pt = event.position()
        self.set_position(int(pt.x() - self._drag_offset[0]), int(pt.y() - self._drag_offset[1]))

    def mouseReleaseEvent(self, event):
        self._dragging = False

    # No keyPressEvent here on purpose -- CropSetupDialog handles every
    # keyboard shortcut (including arrow-key nudging) itself, the same
    # way ObjectSetupDialog does, rather than splitting handling between
    # this canvas and its parent dialog and relying on Qt's focus/event-
    # bubbling behavior to route things correctly.


class CropSetupDialog(QDialog):
    def __init__(self, video_paths, input_folder, output_folder, nor_classifier_dir,
                 width, height, parent=None, force_review=False):
        super().__init__(parent)
        self.setWindowTitle("Crop Setup")
        self.setFocusPolicy(Qt.StrongFocus)

        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.nor_classifier_dir = nor_classifier_dir
        self.runner: CropRunner | None = None
        self._bulk_xy = None

        self.positions_path = self.output_folder / "crop_positions.json"
        self.positions = vc.load_positions(self.positions_path)

        if force_review:
            self.session_videos = list(video_paths)
        else:
            self.session_videos = [
                v for v in video_paths if video_key(v, self.input_folder) not in self.positions
            ]
            if not self.session_videos:
                # Everything's already cropped -- fall back to reviewing
                # all of them rather than opening an empty dialog.
                self.session_videos = list(video_paths)

        self.last_known_xy = None
        if self.positions:
            self.last_known_xy = list(next(reversed(self.positions.values())))

        self.idx = 0

        # -- widgets --
        self.canvas = _CropCanvas()

        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 20000)
        self.width_spin.setValue(width)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 20000)
        self.height_spin.setValue(height)
        self.width_spin.valueChanged.connect(self._on_size_changed)
        self.height_spin.valueChanged.connect(self._on_size_changed)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.progress_label = QLabel()

        self.back_btn = QPushButton("< Back")
        self.forward_btn = QPushButton("Forward >")
        self.skip_btn = QPushButton("Skip (Esc)")
        self.apply_all_btn = QPushButton("Use This Position for All Remaining")
        for b in (self.back_btn, self.forward_btn):
            b.setMinimumHeight(34)
            b.setMinimumWidth(95)

        self.back_btn.clicked.connect(self._go_back)
        self.forward_btn.clicked.connect(self._go_forward)
        self.skip_btn.clicked.connect(self._skip)
        self.apply_all_btn.clicked.connect(self._on_apply_to_remaining)

        self.bulk_progress = QProgressBar()
        self.bulk_progress.setVisible(False)
        self.bulk_log = QPlainTextEdit()
        self.bulk_log.setReadOnly(True)
        self.bulk_log.setMaximumBlockCount(1000)
        self.bulk_log.setFixedHeight(90)
        self.bulk_log.setVisible(False)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Crop size:"))
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("x"))
        size_row.addWidget(self.height_spin)
        size_row.addStretch()
        size_row.addWidget(self.progress_label)

        nav_row = QHBoxLayout()
        nav_row.addWidget(self.back_btn)
        nav_row.addWidget(self.forward_btn)
        nav_row.addWidget(self.skip_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(size_row)
        layout.addWidget(self.canvas, alignment=Qt.AlignHCenter)
        layout.addWidget(self.info_label)
        layout.addLayout(nav_row)
        layout.addWidget(self.apply_all_btn)
        layout.addWidget(self.bulk_progress)
        layout.addWidget(self.bulk_log)

        if self.session_videos:
            self._load_video(0)
        else:
            self.info_label.setText("No videos to crop.")
            self.forward_btn.setEnabled(False)
            self.apply_all_btn.setEnabled(False)

    # -- per-video navigation --

    def _starting_xy_for(self, key):
        if key in self.positions:
            return list(self.positions[key])
        if self.last_known_xy is not None:
            return list(self.last_known_xy)
        return None

    def _load_video(self, idx):
        self.idx = idx
        video_path = self.session_videos[idx]
        try:
            frame = _grab_first_frame(video_path)
        except Exception as exc:
            QMessageBox.warning(self, "Couldn't load video", f"Could not read {video_path.name}:\n{exc}")
            self._advance()
            return

        qimage = _bgr_frame_to_qimage(frame)
        self.canvas.set_frame(qimage, self.width_spin.value(), self.height_spin.value())

        key = video_key(video_path, self.input_folder)
        starting = self._starting_xy_for(key)
        if starting is not None:
            self.canvas.set_position(starting[0], starting[1])
        else:
            self.canvas.set_position(
                (self.canvas.frame_w - self.canvas.crop_w) // 2,
                (self.canvas.frame_h - self.canvas.crop_h) // 2,
            )

        self.setWindowTitle(f"Crop Setup -- {video_path.name}  ({idx + 1}/{len(self.session_videos)})")
        self.progress_label.setText(f"{idx + 1} / {len(self.session_videos)}")
        self.back_btn.setEnabled(idx > 0)
        self._update_info()
        self.canvas.setFocus()

    def _update_info(self):
        if not self.canvas.fits():
            self.info_label.setText(
                f"Crop size {self.width_spin.value()}x{self.height_spin.value()} is larger than this "
                f"video's {self.canvas.frame_w}x{self.canvas.frame_h} frame -- can't crop this one."
            )
            self.info_label.setStyleSheet("color: #a94442;")
        else:
            self.info_label.setText("Drag the box into place (or arrow keys to nudge, Shift for 10px).")
            self.info_label.setStyleSheet("color: #666;")
        self.forward_btn.setEnabled(self.canvas.fits())
        self.apply_all_btn.setEnabled(self.canvas.fits())

    def _on_size_changed(self):
        self.canvas.set_crop_size(self.width_spin.value(), self.height_spin.value())
        self._update_info()

    def _out_path_for(self, video_path):
        rel = Path(video_path).relative_to(self.input_folder)
        return self.output_folder / rel

    def _go_forward(self):
        if not self.canvas.fits():
            return
        video_path = self.session_videos[self.idx]
        x, y = self.canvas.x, self.canvas.y

        self.info_label.setText(f"Cropping {video_path.name}...")
        self.info_label.repaint()
        QApplication.processEvents()

        try:
            vc.crop_video(video_path, self._out_path_for(video_path), x, y, self.width_spin.value(), self.height_spin.value())
        except Exception as exc:
            QMessageBox.warning(self, "Crop failed", f"Couldn't crop {video_path.name}:\n{exc}")
            self._update_info()
            return

        key = video_key(video_path, self.input_folder)
        self.positions[key] = [x, y]
        self.last_known_xy = [x, y]
        vc.save_positions(self.positions_path, self.positions)

        self._advance()

    def _skip(self):
        self._advance()

    def _advance(self):
        if self.idx >= len(self.session_videos) - 1:
            self.accept()
            return
        self._load_video(self.idx + 1)

    def _go_back(self):
        if self.idx == 0:
            return
        self._load_video(self.idx - 1)

    # -- bulk "use this position for all remaining" --

    def _on_apply_to_remaining(self):
        if not self.canvas.fits():
            return
        x, y = self.canvas.x, self.canvas.y
        remaining = self.session_videos[self.idx:]
        reply = QMessageBox.question(
            self, "Crop all remaining?",
            f"Crop the remaining {len(remaining)} video(s) using this same position "
            f"({x},{y}), {self.width_spin.value()}x{self.height_spin.value()}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._bulk_xy = (x, y)
        positions = [(str(v), x, y) for v in remaining]

        self._set_nav_enabled(False)
        self.bulk_progress.setVisible(True)
        self.bulk_progress.setValue(0)
        self.bulk_log.setVisible(True)
        self.bulk_log.appendPlainText(f"Cropping {len(remaining)} remaining video(s) at ({x},{y})...")

        self.runner = CropRunner(
            self.nor_classifier_dir, str(self.input_folder), str(self.output_folder),
            self.width_spin.value(), self.height_spin.value(), positions=positions,
        )
        self.runner.log.connect(self.bulk_log.appendPlainText)
        self.runner.progress.connect(self._on_bulk_progress)
        self.runner.video_done.connect(self._on_bulk_video_done)
        self.runner.finished_run.connect(self._on_bulk_finished_run)
        self.runner.finished.connect(self._on_bulk_process_finished)
        self.runner.start()

    def _on_bulk_progress(self, done, total):
        self.bulk_progress.setMaximum(total)
        self.bulk_progress.setValue(done)

    def _on_bulk_video_done(self, video_path_str):
        # Recorded incrementally (rather than only at the very end) so a
        # later failure in the same batch doesn't leave already-cropped
        # videos looking un-positioned in the cache.
        key = video_key(Path(video_path_str), self.input_folder)
        self.positions[key] = list(self._bulk_xy)
        vc.save_positions(self.positions_path, self.positions)

    def _on_bulk_finished_run(self, status, message):
        suffix = f" -- {message}" if message else ""
        self.bulk_log.appendPlainText(f"Bulk crop finished: {status}{suffix}")

    def _on_bulk_process_finished(self):
        self.runner = None
        self._set_nav_enabled(True)
        self.accept()  # nothing left to position either way, successful or not

    def _set_nav_enabled(self, enabled):
        self.back_btn.setEnabled(enabled and self.idx > 0)
        self.forward_btn.setEnabled(enabled and self.canvas.fits())
        self.skip_btn.setEnabled(enabled)
        self.apply_all_btn.setEnabled(enabled and self.canvas.fits())
        self.width_spin.setEnabled(enabled)
        self.height_spin.setEnabled(enabled)

    # -- keyboard shortcuts (dialog-level, so they work regardless of --
    # -- exactly which child widget currently has focus) --

    def keyPressEvent(self, event):
        key = event.key()
        step = 10 if event.modifiers() & Qt.ShiftModifier else 1
        if key == Qt.Key_Left:
            self.canvas.set_position(self.canvas.x - step, self.canvas.y)
        elif key == Qt.Key_Right:
            self.canvas.set_position(self.canvas.x + step, self.canvas.y)
        elif key == Qt.Key_Up:
            self.canvas.set_position(self.canvas.x, self.canvas.y - step)
        elif key == Qt.Key_Down:
            self.canvas.set_position(self.canvas.x, self.canvas.y + step)
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            self._go_forward()
        elif key == Qt.Key_Escape:
            self._skip()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.runner is not None:
            reply = QMessageBox.question(
                self, "Crop in progress",
                "A bulk crop is still running. Close anyway? The current video will be interrupted.",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self.runner.request_stop()
            self.runner.wait(5000)
        event.accept()
