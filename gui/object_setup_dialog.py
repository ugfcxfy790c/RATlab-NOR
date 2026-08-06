"""
Qt-native replacement for object_picker.setup_missing_videos()'s cv2
window, used by the GUI's "Set Up Objects…" button.

Why this exists: the original picker opens an OpenCV (`cv2.imshow`)
window and runs its own blocking event loop to handle clicks/keys. That
loop doesn't know about Qt's event loop, so while it's running the rest
of the app can't repaint or respond -- not because macOS requires this
(the only real OS constraint is that windows must be created on the main
thread), but because two separate, uncooperative GUI toolkits are being
mixed. Rebuilding the picker as an actual QDialog puts it in the same
event loop as everything else, so it behaves like a normal (modal)
window instead of freezing the app.

Reuses -- deliberately, not incidentally -- object_picker.py's pure state
machine (_PickerState) and persistence helpers (load/save_object_coords,
video_key), so drag/undo/swap/nudge behavior and the on-disk coordinate
format are identical to the original cv2 picker (and to `python main.py
setup`, which still uses that cv2 version directly). Only the rendering
and input plumbing are new.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QDialog, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel,
)

import object_picker as op
from geometry import hitbox_half_width_px, object_half_width_px

LABEL_COLORS = {"novel": QColor(255, 140, 0), "original": QColor(0, 140, 255)}
FOOTPRINT_OUTLINE_COLOR = QColor(190, 190, 190)

HINT_TEXT = (
    "Click empty space to place the next object.\n"
    "Drag a box to move it. Arrow keys nudge by 1px.\n\n"
    "Tab: select for nudging\n"
    "S: swap novel/original\n"
    "U: undo   R: reset video\n"
    "Enter/→: confirm, next video\n"
    "Esc: skip this video"
)


def _grab_first_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame of {video_path}")
    return frame


def _bgr_frame_to_qimage(frame) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return qimg.copy()  # own the buffer -- `rgb` goes out of scope after this call


class _Canvas(QWidget):
    """Paints the current frame + hitboxes and turns mouse input into
    calls on an object_picker._PickerState -- the same state object the
    cv2 version drives from its mouse callback."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.pixmap: QPixmap | None = None
        self.state: "op._PickerState | None" = None
        self.on_change = None  # callable, invoked after any state-changing input

    def set_frame(self, qimage: QImage, state):
        self.pixmap = QPixmap.fromImage(qimage)
        self.state = state
        self.setFixedSize(self.pixmap.size())
        self.update()

    def paintEvent(self, event):
        if self.pixmap is None:
            return
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)
        if self.state is not None:
            self._draw_hitboxes(painter)
        painter.end()

    def _draw_hitboxes(self, painter: QPainter):
        state = self.state
        for label in op.OBJECT_LABELS:
            pos = state.positions.get(label)
            if pos is None:
                continue
            x, y = pos
            hw = state.half_width
            color = LABEL_COLORS[label]
            is_selected = label == state.selected

            painter.setPen(QPen(color, 3 if is_selected else 2))
            painter.drawRect(QRectF(x - hw, y - hw, hw * 2, hw * 2))

            if state.object_half_width is not None:
                ohw = state.object_half_width
                painter.setPen(QPen(FOOTPRINT_OUTLINE_COLOR, 1))
                painter.drawRect(QRectF(x - ohw, y - ohw, ohw * 2, ohw * 2))

            if is_selected:
                pad = 4
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.drawRect(QRectF(x - hw - pad, y - hw - pad, hw * 2 + pad * 2, hw * 2 + pad * 2))

            painter.setPen(QPen(color))
            painter.drawText(QPointF(x - hw, y - hw - 6), label)

    def mousePressEvent(self, event):
        if self.state is None or event.button() != Qt.LeftButton:
            return
        pt = event.position()
        label = self.state.hit_test(pt.x(), pt.y())
        if label is not None:
            self.state.start_drag(label, pt.x(), pt.y())
        else:
            self.state.place_next_unset(pt.x(), pt.y())
        self._changed()

    def mouseMoveEvent(self, event):
        if self.state is None or self.state.dragging is None:
            return
        pt = event.position()
        self.state.drag_to(pt.x(), pt.y())
        self._changed()

    def mouseReleaseEvent(self, event):
        if self.state is None:
            return
        self.state.end_drag()
        self._changed()

    def _changed(self):
        self.update()
        if self.on_change:
            self.on_change()


class ObjectSetupDialog(QDialog):
    """Walks through every video in `session_videos` (already filtered to
    ones needing setup, or all of them if force-reviewing), letting the
    user place/confirm novel + original hitboxes for each, saving to
    `coords_path` as each video is confirmed -- same behavior as
    object_picker.setup_missing_videos(), different toolkit."""

    def __init__(self, video_paths, coords_path, cfg, parent=None, force_review=False):
        super().__init__(parent)
        self.setWindowTitle("Object Setup")
        self.setFocusPolicy(Qt.StrongFocus)

        self.cfg = cfg
        self.coords_path = Path(coords_path)
        self.half_width = hitbox_half_width_px(cfg)
        self.object_half_width = object_half_width_px(cfg)

        self.coords = op.load_object_coords(self.coords_path)
        if force_review:
            self.session_videos = list(video_paths)
        else:
            self.session_videos = [
                v for v in video_paths if op.video_key(v, cfg.VIDEO_FOLDER) not in self.coords
            ]

        self.working = {}
        self.last_known = {label: None for label in op.OBJECT_LABELS}
        if self.coords:
            last_entry = next(reversed(self.coords.values()))
            self.last_known = {label: last_entry.get(label) for label in op.OBJECT_LABELS}
        self.idx = 0
        self.state = None

        self.canvas = _Canvas()
        self.canvas.on_change = self._refresh

        self.select_btn = QPushButton("Select (Tab)")
        self.swap_btn = QPushButton("Swap (S)")
        self.undo_btn = QPushButton("Undo (U)")
        self.back_btn = QPushButton("< Back")
        self.forward_btn = QPushButton("Forward >")
        for btn in (self.back_btn, self.forward_btn):
            btn.setMinimumHeight(34)
            btn.setMinimumWidth(95)
        self.progress_label = QLabel()
        hint_label = QLabel(HINT_TEXT)
        hint_label.setStyleSheet("color: #666; font-size: 11px;")
        hint_label.setWordWrap(True)

        self.select_btn.clicked.connect(self._select_next)
        self.swap_btn.clicked.connect(self._swap)
        self.undo_btn.clicked.connect(self._undo)
        self.back_btn.clicked.connect(self._go_back)
        self.forward_btn.clicked.connect(self._go_forward)

        nav_row = QHBoxLayout()
        nav_row.addWidget(self.back_btn)
        nav_row.addWidget(self.forward_btn)

        sidebar = QVBoxLayout()
        sidebar.addWidget(self.select_btn)
        sidebar.addWidget(self.swap_btn)
        sidebar.addWidget(self.undo_btn)
        sidebar.addSpacing(16)
        sidebar.addLayout(nav_row)
        sidebar.addWidget(self.progress_label)
        sidebar.addStretch()
        sidebar.addWidget(hint_label)
        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)
        sidebar_widget.setFixedWidth(220)

        layout = QHBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addWidget(sidebar_widget)

        if self.session_videos:
            self._load_video(self.idx)

    # -- per-video loading --

    def _current_key(self):
        return op.video_key(self.session_videos[self.idx], self.cfg.VIDEO_FOLDER)

    def _starting_positions_for(self, key):
        if key in self.working:
            return self.working[key]
        if key in self.coords:
            return {
                label: (list(self.coords[key][label]) if self.coords[key].get(label) else None)
                for label in op.OBJECT_LABELS
            }
        return {
            label: (list(self.last_known[label]) if self.last_known.get(label) else None)
            for label in op.OBJECT_LABELS
        }

    def _load_video(self, idx):
        video_path = self.session_videos[idx]
        frame = _grab_first_frame(video_path)
        qimage = _bgr_frame_to_qimage(frame)
        key = op.video_key(video_path, self.cfg.VIDEO_FOLDER)
        self.state = op._PickerState(self._starting_positions_for(key), self.half_width, self.object_half_width)
        # Positions carry forward from the previous video (or the last
        # confirmed video), but selection doesn't -- without this, arrow-key
        # nudging silently does nothing on a fresh video until Tab is
        # pressed. select_next() picks the first already-placed hitbox
        # (novel if both are placed), same as pressing Tab once.
        self.state.select_next()
        self.canvas.set_frame(qimage, self.state)
        self.setWindowTitle(f"Object Setup -- {video_path.name}  ({idx + 1}/{len(self.session_videos)})")
        self.progress_label.setText(f"{idx + 1} / {len(self.session_videos)}")
        self._refresh()

    def _refresh(self):
        self.back_btn.setEnabled(self.idx > 0)
        self.forward_btn.setEnabled(self.state is not None and self.state.all_set())
        self.canvas.update()

    # -- state-mutating actions (shared by buttons and keyboard) --

    def _select_next(self):
        self.state.select_next()
        self._refresh()

    def _swap(self):
        self.state.swap()
        self._refresh()

    def _undo(self):
        self.state.undo()
        self._refresh()

    def _snapshot_working(self):
        key = self._current_key()
        self.working[key] = {
            label: (list(self.state.positions[label]) if self.state.positions[label] else None)
            for label in op.OBJECT_LABELS
        }

    def _go_forward(self):
        if self.state is None or not self.state.all_set():
            return
        self._snapshot_working()
        key = self._current_key()
        self.coords[key] = self.working[key]
        self.last_known = self.coords[key]
        op.save_object_coords(self.coords_path, self.coords)
        if self.idx == len(self.session_videos) - 1:
            self.accept()
            return
        self.idx += 1
        self._load_video(self.idx)

    def _go_back(self):
        if self.idx == 0:
            return
        self._snapshot_working()
        self.idx -= 1
        self._load_video(self.idx)

    def _skip(self):
        self._snapshot_working()
        if self.idx == len(self.session_videos) - 1:
            self.accept()
            return
        self.idx += 1
        self._load_video(self.idx)

    # -- keyboard shortcuts --

    def keyPressEvent(self, event):
        if self.state is None:
            super().keyPressEvent(event)
            return

        key = event.key()
        step = 1
        if key == Qt.Key_Left:
            self.state.move_selected(-step, 0)
            self._refresh()
        elif key == Qt.Key_Right:
            self.state.move_selected(step, 0)
            self._refresh()
        elif key == Qt.Key_Up:
            self.state.move_selected(0, -step)
            self._refresh()
        elif key == Qt.Key_Down:
            self.state.move_selected(0, step)
            self._refresh()
        elif key == Qt.Key_Tab:
            self._select_next()
        elif key == Qt.Key_S:
            self._swap()
        elif key == Qt.Key_U:
            self._undo()
        elif key == Qt.Key_R:
            self.state.reset()
            self._refresh()
        elif key == Qt.Key_Escape:
            self._skip()
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            self._go_forward()
        else:
            super().keyPressEvent(event)


def run_object_setup(video_paths, coords_path, cfg, parent=None, force_review=False):
    """Drop-in replacement for object_picker.setup_missing_videos() that
    shows a Qt dialog instead of a cv2 window. Returns the updated coords
    dict (same contract as the original)."""
    coords = op.load_object_coords(coords_path)
    if force_review:
        pending = list(video_paths)
    else:
        pending = [v for v in video_paths if op.video_key(v, cfg.VIDEO_FOLDER) not in coords]
    if not pending:
        return coords

    dialog = ObjectSetupDialog(video_paths, coords_path, cfg, parent=parent, force_review=force_review)
    dialog.exec()
    return op.load_object_coords(coords_path)
