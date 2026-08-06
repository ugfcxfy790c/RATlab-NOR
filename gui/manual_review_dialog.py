"""
Manual analysis utility for a single video -- lets a person re-score a
video by hand (nose-in-hitbox + oriented sniffing bouts) when the model's
own output looks suspicious, and write the result to the same Excel sheet
the batch pipeline writes to (see excel_writer.write_manual_review).

Deliberately shows only what a human reviewer needs to make that call:
the object hitboxes, the tracked skeleton (colored by confidence), and
the sniff-cone rays -- the same geometry validation_video.py draws. It
does NOT show anything about what the *system* decided (no "counted" /
"exploring" / "climbing" status, no green/magenta bout-state hitbox
coloring) -- that would just be showing the reviewer the very judgment
they're here to double-check.

Playback controls (rewind/step/play-pause/step/fast-forward, the normal-
speed toggle, and the Space shortcut) come from video_player_widget.py,
shared with batch_review_dialog.py's flagged-video viewer -- this dialog
only adds the geometry overlay (via VideoPlayerWidget's overlay_painter
hook) and everything bout-marking related. Tab (handled here, not in the
shared widget, since it's specific to bout marking) cycles which object
bouts are currently being marked for. Bout logging is a simple start/stop
state machine; nothing is written to disk until "Finalize".
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QPen, QColor, QBrush, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QTableWidget, QTableWidgetItem, QMessageBox, QHeaderView,
    QAbstractItemView, QGroupBox,
)

from geometry import hitbox_half_width_px, object_half_width_px
from video_player_widget import VideoPlayerWidget
import excel_writer

# -- overlay colors (RGB, for QPainter -- see validation_video.py for the
# BGR/cv2 originals this mirrors) --
_CONFIDENT_COLOR = QColor(0, 220, 0)
_LOW_CONF_COLOR = QColor(255, 0, 0)
_EDGE_COLOR = QColor(180, 180, 180)
_HITBOX_DEFAULT_COLOR = QColor(200, 200, 200)
_FOOTPRINT_OUTLINE_COLOR = QColor(210, 210, 210)
_CONE_CENTER_COLOR = QColor(0, 255, 255)   # cyan -- head vector
_CONE_EDGE_COLOR = QColor(255, 255, 0)     # yellow -- cone boundary rays
_OBJECT_COLORS = {"novel": QColor(255, 140, 0), "original": QColor(0, 140, 255)}
_SELECTED_OBJECT_RING = QColor(255, 255, 255)


def _skeleton_edges(cfg):
    """Same default connective chain as validation_video.py's
    _skeleton_edges -- purely visual, has no effect on anything written
    out."""
    return [
        (cfg.NODE_NOSE, cfg.NODE_NECK),
        (cfg.NODE_NECK, cfg.NODE_TORSO),
        (cfg.NODE_TORSO, cfg.NODE_TAIL_BASE),
        (cfg.NODE_NECK, cfg.NODE_LEFT_EAR),
        (cfg.NODE_NECK, cfg.NODE_RIGHT_EAR),
    ]


class ManualReviewDialog(QDialog):
    """One video's manual-review session: play/step through it, mark
    exploration bouts by hand, and write the result to the group's
    Excel workbook on Finalize.

    track: pose_utils.Track already loaded for this video (see
        pose_utils.load_track_from_slp).
    object_coords: {name: (x, y)} for this video, in the same order as
        `object_names` (object 1 = first item, object 2 = second).
    """

    def __init__(
        self, video_path, track, object_coords, cfg, object_names,
        output_path, rat_id, session_label, parent=None,
    ):
        super().__init__(parent)
        self.video_path = Path(video_path)
        self.track = track
        self.object_coords = object_coords
        self.cfg = cfg
        self.object_names = list(object_names)
        self.output_path = Path(output_path)
        self.rat_id = rat_id
        self.session_label = session_label

        self.setWindowTitle(f"Manual Review -- {self.video_path.name}")
        self.setFocusPolicy(Qt.StrongFocus)

        # Overlay-drawing prerequisites, needed before the first
        # VideoPlayerWidget frame paints.
        self.half_width = hitbox_half_width_px(cfg)
        self.footprint_half_width = object_half_width_px(cfg)
        self.edges = _skeleton_edges(cfg)
        self._active_object = self.object_names[0] if self.object_names else None

        # bouts[object_name] = [(start_s, stop_s, start_frame, stop_frame), ...]
        self.bouts: dict[str, list[tuple[float, float, int, int]]] = {name: [] for name in self.object_names}
        self.pending_start = None  # (frame_idx, time_s) or None

        try:
            self.player = VideoPlayerWidget(
                self.video_path, overlay_painter=self._draw_review_overlay,
                fps_hint=track.fps, n_frames_cap=track.n_frames,
            )
        except RuntimeError:
            raise

        self._build_ui()

    # -- UI construction --

    def _build_ui(self):
        # Tab (cycle marked object) works no matter which child widget
        # currently has keyboard focus -- Space (play/pause) is already
        # wired up by VideoPlayerWidget itself.
        self._tab_shortcut = QShortcut(QKeySequence(Qt.Key_Tab), self)
        self._tab_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._tab_shortcut.activated.connect(self._cycle_active_object)

        # Bout marking.
        self.object_combo = QComboBox()
        self.object_combo.addItems(self.object_names)
        self.object_combo.currentTextChanged.connect(self._update_object_highlight)

        self.start_btn = QPushButton("Bout Start")
        self.stop_btn = QPushButton("Bout Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_bout_start)
        self.stop_btn.clicked.connect(self._on_bout_stop)

        self.bout_status_label = QLabel("No bout in progress.")

        bout_controls = QHBoxLayout()
        bout_controls.addWidget(QLabel("Marking bouts for:"))
        bout_controls.addWidget(self.object_combo)
        bout_controls.addWidget(self.start_btn)
        bout_controls.addWidget(self.stop_btn)
        bout_controls.addStretch()

        self.bout_table = QTableWidget(0, 4)
        self.bout_table.setHorizontalHeaderLabels(["Object", "Start (s)", "Stop (s)", "Duration (s)"])
        self.bout_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.bout_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.bout_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.delete_bout_btn = QPushButton("Delete Selected Bout")
        self.delete_bout_btn.clicked.connect(self._on_delete_selected_bout)

        bout_box = QGroupBox("Logged bouts (nothing is saved until Finalize)")
        bout_box_layout = QVBoxLayout(bout_box)
        bout_box_layout.addLayout(bout_controls)
        bout_box_layout.addWidget(self.bout_status_label)
        bout_box_layout.addWidget(self.bout_table)
        bout_box_layout.addWidget(self.delete_bout_btn)

        self.finalize_btn = QPushButton("Finalize -- Write to Excel")
        self.close_btn = QPushButton("Close Without Saving")
        self.finalize_btn.clicked.connect(self._on_finalize)
        self.close_btn.clicked.connect(self.reject)

        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        bottom_row.addWidget(self.close_btn)
        bottom_row.addWidget(self.finalize_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.player)
        layout.addWidget(bout_box)
        layout.addLayout(bottom_row)

    # -- overlay drawing (VideoPlayerWidget's overlay_painter hook) --

    def _draw_review_overlay(self, painter, frame_idx):
        track = self.track
        if track is None or frame_idx >= track.n_frames:
            return
        cfg = self.cfg
        node_names = track.node_names

        # Skeleton connective lines (back to front, matching
        # validation_video.py's z-order).
        painter.setPen(QPen(_EDGE_COLOR, 2))
        for a, b in self.edges:
            if a not in node_names or b not in node_names:
                continue
            pa = track.xy(a)[frame_idx]
            pb = track.xy(b)[frame_idx]
            if np.isnan(pa).any() or np.isnan(pb).any():
                continue
            painter.drawLine(QPointF(pa[0], pa[1]), QPointF(pb[0], pb[1]))

        # Hitboxes + real-footprint outlines.
        for name, xy in self.object_coords.items():
            cx, cy = float(xy[0]), float(xy[1])
            color = _OBJECT_COLORS.get(name, _HITBOX_DEFAULT_COLOR)
            hw = self.half_width
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(cx - hw, cy - hw, hw * 2, hw * 2))

            ohw = self.footprint_half_width
            painter.setPen(QPen(_FOOTPRINT_OUTLINE_COLOR, 1))
            painter.drawRect(QRectF(cx - ohw, cy - ohw, ohw * 2, ohw * 2))

            if name == self._active_object:
                pad = 4
                painter.setPen(QPen(_SELECTED_OBJECT_RING, 1))
                painter.drawRect(QRectF(cx - hw - pad, cy - hw - pad, hw * 2 + pad * 2, hw * 2 + pad * 2))

            painter.setPen(QPen(color))
            painter.drawText(QPointF(cx - hw, cy - hw - 6), name)

        # Sniff-cone rays -- same geometry scoring._head_cone_intersects_object
        # tests against, drawn from the same pulled-back apex.
        if cfg.NODE_NOSE in node_names and cfg.NODE_NECK in node_names:
            nose_xy = track.xy(cfg.NODE_NOSE)[frame_idx]
            neck_xy = track.xy(cfg.NODE_NECK)[frame_idx]
            if not np.isnan(nose_xy).any() and not np.isnan(neck_xy).any():
                head_vec = nose_xy - neck_xy
                head_norm = np.linalg.norm(head_vec)
                if head_norm > 1e-6:
                    unit = head_vec / head_norm
                    ray_len = self.half_width * 1.8
                    half_angle_rad = math.radians(cfg.SNIFF_CONE_HALF_ANGLE_DEG)
                    origin_xy = nose_xy - cfg.SNIFF_RAY_ORIGIN_BACKSET_RATIO * head_vec
                    start = QPointF(origin_xy[0], origin_xy[1])
                    for angle, color in (
                        (0.0, _CONE_CENTER_COLOR),
                        (half_angle_rad, _CONE_EDGE_COLOR),
                        (-half_angle_rad, _CONE_EDGE_COLOR),
                    ):
                        c, s = math.cos(angle), math.sin(angle)
                        rx = unit[0] * c - unit[1] * s
                        ry = unit[0] * s + unit[1] * c
                        end = QPointF(origin_xy[0] + rx * ray_len, origin_xy[1] + ry * ray_len)
                        painter.setPen(QPen(color, 1))
                        painter.drawLine(start, end)

        # Node markers, colored by confidence -- drawn last so a tracked
        # point is never obscured by anything else.
        radius = 3
        for node_name in node_names:
            xy = track.xy(node_name)[frame_idx]
            if np.isnan(xy).any():
                continue
            conf = track.conf(node_name)[frame_idx]
            confident = not np.isnan(conf) and conf >= cfg.MIN_NODE_CONFIDENCE
            color = _CONFIDENT_COLOR if confident else _LOW_CONF_COLOR
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPointF(xy[0], xy[1]), radius, radius)

    # -- keyboard shortcuts --

    def _cycle_active_object(self):
        if self.pending_start is not None:
            return  # don't switch which object a bout is being marked for mid-bout
        n = self.object_combo.count()
        if n == 0:
            return
        idx = (self.object_combo.currentIndex() + 1) % n
        self.object_combo.setCurrentIndex(idx)

    # -- object selection --

    def _selected_object(self):
        return self.object_combo.currentText()

    def _update_object_highlight(self, *_args):
        self._active_object = self._selected_object()
        self.player.refresh_overlay()

    # -- bout marking --

    def _on_bout_start(self):
        if self.pending_start is not None:
            return
        idx = self.player.current_frame_idx
        self.pending_start = (idx, idx / self.player.fps)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.object_combo.setEnabled(False)
        self.bout_status_label.setText(
            f"Bout in progress for '{self._selected_object()}' -- started at frame "
            f"{self.pending_start[0]} (t={self.pending_start[1]:.3f}s). Navigate to the "
            f"end of the bout, then press Bout Stop."
        )

    def _on_bout_stop(self):
        if self.pending_start is None:
            return
        start_frame, start_s = self.pending_start
        stop_frame = self.player.current_frame_idx
        stop_s = stop_frame / self.player.fps

        if stop_frame < start_frame:
            start_frame, stop_frame = stop_frame, start_frame
            start_s, stop_s = stop_s, start_s
        elif stop_frame == start_frame:
            QMessageBox.warning(
                self, "Zero-length bout",
                "Bout Stop was pressed on the same frame as Bout Start -- "
                "navigate to a later frame first.",
            )
            return

        obj = self._selected_object()
        self.bouts.setdefault(obj, []).append((start_s, stop_s, start_frame, stop_frame))
        self.pending_start = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.object_combo.setEnabled(True)
        self.bout_status_label.setText("No bout in progress.")
        self._refresh_bout_table()

    def _refresh_bout_table(self):
        rows = []
        for obj in self.object_names:
            for i, (start_s, stop_s, _sf, _ef) in enumerate(self.bouts.get(obj, [])):
                rows.append((obj, i, start_s, stop_s))
        rows.sort(key=lambda r: r[2])

        self.bout_table.setRowCount(len(rows))
        self._bout_row_refs = []  # row index -> (object_name, list_index)
        for row_i, (obj, list_idx, start_s, stop_s) in enumerate(rows):
            self.bout_table.setItem(row_i, 0, QTableWidgetItem(obj))
            self.bout_table.setItem(row_i, 1, QTableWidgetItem(f"{start_s:.3f}"))
            self.bout_table.setItem(row_i, 2, QTableWidgetItem(f"{stop_s:.3f}"))
            self.bout_table.setItem(row_i, 3, QTableWidgetItem(f"{stop_s - start_s:.3f}"))
            self._bout_row_refs.append((obj, list_idx))

    def _on_delete_selected_bout(self):
        selected_rows = sorted({idx.row() for idx in self.bout_table.selectionModel().selectedRows()}, reverse=True)
        if not selected_rows:
            return
        # Delete by (object, start_s/stop_s) identity rather than raw list
        # index, since deleting one row shifts the list indices of every
        # later row for the same object -- collect the actual tuples first.
        to_remove = []
        for row_i in selected_rows:
            obj, list_idx = self._bout_row_refs[row_i]
            to_remove.append((obj, self.bouts[obj][list_idx]))
        for obj, bout in to_remove:
            if bout in self.bouts.get(obj, []):
                self.bouts[obj].remove(bout)
        self._refresh_bout_table()

    # -- finalize --

    def _on_finalize(self):
        if self.pending_start is not None:
            reply = QMessageBox.question(
                self, "Bout still in progress",
                "A bout was started but never stopped. Discard it and finalize the "
                "rest of your review anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            self.pending_start = None

        if not any(self.bouts.values()):
            reply = QMessageBox.question(
                self, "No bouts marked",
                "No bouts have been marked for this video. Write a zero-bout manual "
                "review anyway? (This will still displace any existing computed data "
                "to the reference area.)",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        if excel_writer.has_human_review(self.output_path, self.rat_id, self.session_label):
            reply = QMessageBox.question(
                self, "Overwrite previous manual review?",
                f"'{self.session_label}' already has a human-reviewed block in "
                f"{self.output_path.name}. Finalizing now will replace it with what's "
                f"logged here. The original computed reference data is unaffected. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        bouts_by_index = {}
        for i, name in enumerate(self.object_names, start=1):
            entries = sorted(self.bouts.get(name, []), key=lambda b: b[0])
            bouts_by_index[i] = [(start_s, stop_s) for start_s, stop_s, _sf, _ef in entries]

        try:
            excel_writer.write_manual_review(
                self.output_path, self.rat_id, self.session_label, bouts_by_index, self.object_names,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Failed to save", f"Could not write manual review:\n{exc}")
            return

        QMessageBox.information(
            self, "Saved",
            f"Manual review for '{self.session_label}' written to {self.output_path.name}.",
        )
        self.accept()

    # -- lifecycle --

    def closeEvent(self, event):
        self.player.release()
        super().closeEvent(event)

    def reject(self):
        if any(self.bouts.values()) or self.pending_start is not None:
            reply = QMessageBox.question(
                self, "Discard manual review?",
                "You've logged bouts that haven't been saved. Close without saving?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        super().reject()
