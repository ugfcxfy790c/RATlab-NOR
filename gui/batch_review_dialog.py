"""
Post-batch review screen: lists every video across a set of jobs, color
coded by review status --

  - red:    currently flagged by review_flags.py (low tracking
            confidence, or the nose dropping out of confident tracking
            near an object for a while) and not yet reviewed.
  - yellow: has a manual-review block in the group's Excel workbook
            (excel_writer.has_human_review) -- someone already hand-
            re-scored this exact video. Takes priority over red/green:
            it's the most specific, most reassuring status available.
  - green:  everything else -- never flagged, or flagged and since
            marked reviewed/cleared.

Opened automatically after a batch run finishes if it left anything
newly flagged (see main_window.py's _on_all_finished), or on demand via
the "Review" menu's "Review Videos..." action.

Double-clicking a row opens FlaggedVideoViewer: the already-annotated
validation video, played with the same transport controls as everywhere
else in this app (video_player_widget.py), plus "Mark as Reviewed"
(clears an active flag) and "Analyze Manually" (hands off to
manual_review_dialog.py for the underlying raw video). Its scrub bar
marks tracking-confidence dropouts (red) and each object's computed
exploration bouts (in that object's hitbox color -- see _OBJECT_COLORS,
matching manual_review_dialog.py/object_setup_dialog.py's own hitbox
colors) for quick navigation.
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QMessageBox, QMenu,
)

import excel_writer
import review_flags
import scoring
import sleap_inference
import validation_video
from job_queue import job_coords_path, job_predictions_folder, find_videos
from job_config import build_job_cfg
from manual_review_dialog import ManualReviewDialog
from video_player_widget import VideoPlayerWidget
from pose_utils import load_track_from_slp

_COLOR_FLAGGED = QColor(190, 0, 0)      # red -- still flagged, unreviewed
_COLOR_CLEAR = QColor(0, 120, 0)        # green -- not flagged, or flagged and reviewed/cleared
_COLOR_MANUAL = QColor(178, 134, 0)     # amber/yellow -- has a manual-review block already

# Scrub-bar marker colors -- same hitbox colors used everywhere else in
# the app (manual_review_dialog.py/object_setup_dialog.py's
# _OBJECT_COLORS, object_picker.LABEL_COLORS), duplicated here rather
# than imported since each of those already keeps its own local copy for
# its own toolkit (cv2 BGR tuples vs QColor) -- this is the QColor one.
_SCRUB_CONFIDENCE_COLOR = QColor(200, 0, 0)
_SCRUB_OBJECT_COLORS = {"novel": QColor(255, 140, 0), "original": QColor(0, 140, 255)}
_SCRUB_OBJECT_DEFAULT_COLOR = QColor(160, 160, 160)

# Distinct point-marker color for where scoring.py's confidence-aware
# merging (labels_to_bouts' merge_events) automatically bridged a
# low-confidence gap -- painted last (on top of the confidence/bout range
# markers) so these small markers stay visible even where they land
# inside a bout's own colored range.
_SCRUB_AUTO_MERGE_COLOR = QColor(180, 0, 220)


def _parse_filename(video_path, cfg):
    """Same rat_id/phase/session convention as main.py's
    parse_filename()/batch_worker_process.py's _parse_filename()."""
    stem = Path(video_path).stem
    pattern = getattr(cfg, "FILENAME_PATTERN", None)
    if pattern:
        m = re.match(pattern, stem)
        if m:
            return m.group("rat_id"), m.groupdict().get("phase", ""), m.groupdict().get("session", "")
    return stem, "", ""


def has_active_flags(jobs):
    """True if any of `jobs` currently has an unreviewed flagged video --
    used by main_window.py to decide whether to pop the review screen
    open right after a batch run."""
    return any(review_flags.active_flags(job.output_folder) for job in jobs)


def launch_manual_review(parent, job, video_path, app_data_dir, config_module, object_picker_module):
    """Load whatever manual_review_dialog.ManualReviewDialog needs for
    `video_path` (object coords, tracking predictions) and run it modally.
    Shared by FlaggedVideoViewer's "Analyze Manually" button and
    BatchReviewDialog's row context menu, so the two entry points can't
    drift apart.

    Returns True if the dialog was accepted (a manual re-score was
    finalized and written to the workbook), False if it couldn't be
    opened at all (missing coords/predictions -- a QMessageBox is shown
    to explain why) or the user canceled out of it.
    """
    video_path = Path(video_path)
    predictions_folder = job_predictions_folder(job, app_data_dir)
    coords_path = job_coords_path(job, app_data_dir)
    coords = object_picker_module.load_object_coords(coords_path)
    key = object_picker_module.video_key(video_path, job.input_folder)
    if key not in coords:
        QMessageBox.warning(
            parent, "No object coordinates",
            f"'{video_path.name}' has no object coordinates set -- "
            f"run Set Up Objects for this job first.",
        )
        return False
    object_coords = coords[key]
    object_names = list(object_coords.keys())

    slp_path = sleap_inference.predictions_path_for(video_path, predictions_folder)
    if not slp_path.exists():
        QMessageBox.warning(parent, "No tracking data", f"No tracking predictions found for '{video_path.name}'.")
        return False
    try:
        track = load_track_from_slp(slp_path, video_path=video_path)
    except Exception as exc:
        QMessageBox.critical(parent, "Couldn't load tracking data", f"Failed to load {slp_path.name}:\n{exc}")
        return False

    job_cfg = build_job_cfg(job, config_module)
    rat_id, _phase, _session = _parse_filename(video_path, job_cfg)
    output_path = job.output_folder / f"{job.group_name}_NOR_results.xlsx"

    try:
        dialog = ManualReviewDialog(
            video_path, track, object_coords, job_cfg, object_names,
            output_path, rat_id, video_path.stem, parent=parent,
        )
    except Exception as exc:
        QMessageBox.critical(parent, "Couldn't open video", f"Failed to open {video_path.name}:\n{exc}")
        return False

    return dialog.exec() == QDialog.Accepted


def swap_video_objects(parent, job, video_path, app_data_dir, config_module, object_picker_module):
    """Fix one video where novel/original got labeled backwards during
    Set Up Objects, without redoing pose inference or touching any other
    video in the job: swaps that video's novel/original coordinates,
    re-scores it against the existing .slp tracking data (no
    re-tracking), writes the corrected result into just its block in the
    workbook (excel_writer.write_computed_review -- discards any prior
    human review/reference for this video, since both were built on the
    wrong labels), and regenerates just its validation video.

    Shared by FlaggedVideoViewer's "Swap Novel/Original" button and
    BatchReviewDialog's row context menu action of the same name, so the
    two entry points can't drift apart -- same pairing as
    launch_manual_review() above.

    Returns True on success, False if it couldn't be done at all (missing
    coords/predictions, or export failed -- a QMessageBox explains why).
    Callers are expected to confirm with the user *before* calling this,
    since it's a destructive, no-undo edit to the workbook.
    """
    video_path = Path(video_path)
    coords_path = job_coords_path(job, app_data_dir)
    coords = object_picker_module.load_object_coords(coords_path)
    key = object_picker_module.video_key(video_path, job.input_folder)
    if key not in coords:
        QMessageBox.warning(
            parent, "No object coordinates",
            f"'{video_path.name}' has no object coordinates set -- "
            f"run Set Up Objects for this job first.",
        )
        return False

    predictions_folder = job_predictions_folder(job, app_data_dir)
    slp_path = sleap_inference.predictions_path_for(video_path, predictions_folder)
    if not slp_path.exists():
        QMessageBox.warning(parent, "No tracking data", f"No tracking predictions found for '{video_path.name}'.")
        return False
    try:
        track = load_track_from_slp(slp_path, video_path=video_path)
    except Exception as exc:
        QMessageBox.critical(parent, "Couldn't load tracking data", f"Failed to load {slp_path.name}:\n{exc}")
        return False

    object_picker_module.swap_labels(coords[key])
    object_picker_module.save_object_coords(coords_path, coords)
    object_coords = coords[key]
    object_names = list(object_coords.keys())

    job_cfg = build_job_cfg(job, config_module)
    rat_id, _phase, _session = _parse_filename(video_path, job_cfg)
    output_path = job.output_folder / f"{job.group_name}_NOR_results.xlsx"

    labels = scoring.score_frame_labels(track, object_coords, cfg=job_cfg)
    bouts_by_index, merge_events = scoring.labels_to_bouts(labels, track, cfg=job_cfg)
    bouts_to_write = {
        i: [(b.start_s, b.stop_s) for b in bouts_by_index.get(i, [])]
        for i in range(1, len(object_names) + 1)
    }
    try:
        excel_writer.write_computed_review(output_path, rat_id, video_path.stem, bouts_to_write, object_names)
    except Exception as exc:
        QMessageBox.critical(parent, "Couldn't save swap", f"Failed to write the re-scored result:\n{exc}")
        return False

    validation_path = job.output_folder / "validation_videos" / f"{video_path.stem}.validation.mp4"
    if validation_path.exists():
        try:
            validation_video.export_validation_video(video_path, track, object_coords, job_cfg, validation_path)
        except Exception as exc:
            QMessageBox.warning(
                parent, "Result saved, but validation video couldn't be updated",
                f"'{video_path.name}' was re-scored with the swapped objects, but its validation "
                f"video couldn't be regenerated:\n{exc}\n\nThe old validation video (with the "
                f"pre-swap labels) is still on disk.",
            )

    # Recompute review flags too, so a video that was flagged/cleared
    # purely because of the mislabeling doesn't keep nagging (or stay
    # silently wrong) under its new, corrected scoring.
    try:
        reasons = review_flags.compute_video_flags(track, object_coords, cfg=job_cfg, merge_events=merge_events)
        review_flags.update_video_flags(job.output_folder, video_path.stem, reasons)
    except Exception:
        pass

    return True


class FlaggedVideoViewer(QDialog):
    """Plays one video's validation video (already annotated -- no
    overlay drawing needed here), with Mark as Reviewed / Analyze
    Manually alongside the standard transport controls."""

    def __init__(
        self, job, video_path, validation_path, reasons, app_data_dir,
        config_module, object_picker_module, parent=None,
    ):
        super().__init__(parent)
        self.job = job
        self.video_path = Path(video_path)
        self.validation_path = Path(validation_path)
        self.reasons = reasons
        self.app_data_dir = app_data_dir
        self.config_module = config_module
        self.object_picker_module = object_picker_module
        self.marked_reviewed = False

        # Populated by _load_scrub_markers() -- kept around (rather than
        # local variables) so _on_merge_bout() can find the bout pair
        # straddling a dropout and, on confirm, patch just that one entry
        # and re-render the scrub bar without redoing all the scoring
        # work (which would just recompute the same split bouts again).
        self._job_cfg = None
        self._fps = None
        self._object_names = []
        self._bouts_by_index = {}
        self._merge_events_by_index = {}
        self._dropout_runs = []
        self._output_path = None
        self._rat_id = None

        self.setWindowTitle(f"Video Review -- {self.video_path.name}")

        self.player = VideoPlayerWidget(self.validation_path)
        self._load_scrub_markers()

        entry = review_flags.load_flags(self.job.output_folder).get(self.video_path.stem, {})
        is_active_flag = bool(entry.get("reasons")) and not entry.get("reviewed")
        if reasons:
            reasons_text = "\n".join(f"- {r}" for r in reasons)
            self.reasons_label = QLabel(f"Flagged for:\n{reasons_text}")
            self.reasons_label.setStyleSheet(f"color: {_COLOR_FLAGGED.name() if is_active_flag else _COLOR_CLEAR.name()};")
        else:
            self.reasons_label = QLabel("Not flagged -- nothing suspicious found automatically.")
            self.reasons_label.setStyleSheet(f"color: {_COLOR_CLEAR.name()};")
        self.reasons_label.setWordWrap(True)

        self.reviewed_btn = QPushButton("Mark as Reviewed")
        self.reviewed_btn.setEnabled(is_active_flag)
        self.merge_btn = QPushButton("Merge Bout at Playhead")
        self.merge_btn.setToolTip(
            "Park the playhead near a red confidence-dropout marker that splits one "
            "continuous bout in two, then click this to merge them back together."
        )
        self.merge_btn.setEnabled(bool(self._dropout_runs) and bool(self._bouts_by_index))
        self.swap_btn = QPushButton("Swap Novel/Original…")
        self.swap_btn.setToolTip(
            "Fix this video if the two objects were mislabeled during Set Up Objects -- "
            "re-scores it with novel/original swapped, using the existing tracking data "
            "(no re-tracking needed)."
        )
        self.manual_btn = QPushButton("Analyze Manually…")
        self.close_btn = QPushButton("Close")

        self.reviewed_btn.clicked.connect(self._on_mark_reviewed)
        self.merge_btn.clicked.connect(self._on_merge_bout)
        self.swap_btn.clicked.connect(self._on_swap_objects)
        self.manual_btn.clicked.connect(self._on_analyze_manually)
        self.close_btn.clicked.connect(self.accept)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.reviewed_btn)
        btn_row.addWidget(self.merge_btn)
        btn_row.addWidget(self.swap_btn)
        btn_row.addWidget(self.manual_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.close_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.reasons_label)
        layout.addWidget(self.player)
        layout.addLayout(btn_row)

    def _load_scrub_markers(self):
        """Best-effort: mark where the nose's own tracking confidence
        dropped out (red) and each object's computed exploration bouts
        (in that object's hitbox color) on the scrub bar, so a reviewer
        can jump straight to a point of interest instead of scrubbing
        blind. Silently does nothing it can't compute -- these are a
        navigation aid, not essential to viewing the validation video.
        Confidence markers are added first so bout markers (the more
        specific, more directly actionable of the two) paint on top
        wherever the two overlap -- see _ScrubBar's docstring.

        Also stashes the bouts/dropout runs it computed on self (see
        __init__'s comment) so _on_merge_bout() can act on exactly what's
        currently drawn, without re-deriving it and without needing a
        second, possibly-diverging pass over the tracking data."""
        try:
            self._job_cfg = build_job_cfg(self.job, self.config_module)
            predictions_folder = job_predictions_folder(self.job, self.app_data_dir)
            slp_path = sleap_inference.predictions_path_for(self.video_path, predictions_folder)
            if not slp_path.exists():
                return
            track = load_track_from_slp(slp_path, video_path=self.video_path)
            self._fps = track.fps
            self._dropout_runs = review_flags.nose_confidence_dropout_runs(track, self._job_cfg)
            self._output_path = self.job.output_folder / f"{self.job.group_name}_NOR_results.xlsx"
            self._rat_id, _phase, _session = _parse_filename(self.video_path, self._job_cfg)

            coords = self.object_picker_module.load_object_coords(
                job_coords_path(self.job, self.app_data_dir)
            )
            key = self.object_picker_module.video_key(self.video_path, self.job.input_folder)
            if key in coords:
                object_coords = coords[key]
                self._object_names = list(object_coords.keys())
                labels = scoring.score_frame_labels(track, object_coords, cfg=self._job_cfg)
                self._bouts_by_index, self._merge_events_by_index = scoring.labels_to_bouts(
                    labels, track, cfg=self._job_cfg
                )

            self._rebuild_scrub_markers()
        except Exception:
            pass

    def _rebuild_scrub_markers(self):
        """Redraw the scrub bar from self._dropout_runs/_bouts_by_index --
        called on initial load and again after a merge patches
        _bouts_by_index in place, so the bar reflects the merge
        immediately without re-running scoring (which would just
        reproduce the original split).

        Auto-merge markers (self._merge_events_by_index -- where
        scoring.py's confidence-aware merging bridged a low-confidence
        gap on its own, see labels_to_bouts()) are added last, in their
        own distinct color, so a reviewer can jump straight to each one
        and sanity-check the scorer's guess rather than trusting it
        blindly -- same reasoning as the REVIEW_MAX_AUTO_MERGES review
        flag."""
        markers = [(start, stop, _SCRUB_CONFIDENCE_COLOR) for start, stop in self._dropout_runs]
        for obj_i, name in enumerate(self._object_names, start=1):
            color = _SCRUB_OBJECT_COLORS.get(name, _SCRUB_OBJECT_DEFAULT_COLOR)
            for bout in self._bouts_by_index.get(obj_i, []):
                start_frame = round(bout.start_s * self._fps)
                stop_frame = round(bout.stop_s * self._fps)
                markers.append((start_frame, stop_frame, color))
        for events in self._merge_events_by_index.values():
            for start, stop in events:
                markers.append((start, stop, _SCRUB_AUTO_MERGE_COLOR))
        self.player.set_markers(markers)

    def _find_mergeable_pair(self, frame_idx):
        """Find the pair of consecutive same-object bouts, if any, whose
        gap is bridged by a confidence dropout at/near `frame_idx` --
        i.e. the isolated dropout that likely split one real bout of
        exploration into two. Returns (obj_index, obj_name, pair_index,
        bout_a, bout_b) or None if the playhead isn't near a qualifying
        split.

        "Near" means either inside a dropout run, or within 2 seconds of
        one -- close enough that clicking without frame-perfect scrubbing
        still finds it, but far enough away that a distant, unrelated
        dropout won't get matched by accident."""
        if not self._dropout_runs or not self._bouts_by_index or not self._fps:
            return None

        tolerance_frames = self._fps * 2.0
        best_run = None
        best_dist = None
        for start, stop in self._dropout_runs:
            dist = 0.0 if start <= frame_idx < stop else min(abs(frame_idx - start), abs(frame_idx - stop))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_run = (start, stop)
        if best_run is None or best_dist > tolerance_frames:
            return None
        dropout_start, dropout_stop = best_run

        for obj_i, name in enumerate(self._object_names, start=1):
            bouts = self._bouts_by_index.get(obj_i, [])
            for j in range(len(bouts) - 1):
                a, b = bouts[j], bouts[j + 1]
                gap_start_frame = a.stop_s * self._fps
                gap_stop_frame = b.start_s * self._fps
                # The dropout just needs to fall within (or overlap) the
                # gap between the two bouts -- it's what's plausibly
                # masking the continuation of one real bout as two.
                if gap_start_frame <= dropout_stop and gap_stop_frame >= dropout_start:
                    return obj_i, name, j, a, b
        return None

    def _on_merge_bout(self):
        match = self._find_mergeable_pair(self.player.current_frame_idx)
        if match is None:
            QMessageBox.information(
                self, "No split bout found",
                "Move the playhead near a red confidence-dropout marker that sits "
                "between two bouts of the same object, then try again.",
            )
            return
        obj_i, name, pair_idx, bout_a, bout_b = match

        answer = QMessageBox.question(
            self, "Merge bouts",
            f"Merge these two '{name}' bouts into one continuous bout?\n\n"
            f"  {bout_a.start_s:.2f}s – {bout_a.stop_s:.2f}s\n"
            f"  {bout_b.start_s:.2f}s – {bout_b.stop_s:.2f}s\n\n"
            f"becomes:\n\n"
            f"  {bout_a.start_s:.2f}s – {bout_b.stop_s:.2f}s\n\n"
            f"This is saved as a manual correction to the workbook and can't be undone "
            f"automatically (though the original computed result is kept alongside it, "
            f"and can be restored via \"Revert to Computed Data\").",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        merged = scoring.Bout(start_s=bout_a.start_s, stop_s=bout_b.stop_s)
        self._bouts_by_index[obj_i] = (
            self._bouts_by_index[obj_i][:pair_idx] + [merged] + self._bouts_by_index[obj_i][pair_idx + 2:]
        )

        bouts_to_write = {
            i: [(b.start_s, b.stop_s) for b in self._bouts_by_index.get(i, [])]
            for i in range(1, len(self._object_names) + 1)
        }
        try:
            excel_writer.write_manual_review(
                self._output_path, self._rat_id, self.video_path.stem,
                bouts_to_write, self._object_names,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Couldn't save merge", f"Failed to write the merged bout:\n{exc}")
            return

        self._rebuild_scrub_markers()
        self._clear_active_flag()

    def _clear_active_flag(self):
        review_flags.mark_reviewed(self.job.output_folder, self.video_path.stem)
        self.marked_reviewed = True

    def _on_mark_reviewed(self):
        self._clear_active_flag()
        self.accept()  # nothing left to look at here -- close the viewer too

    def _on_analyze_manually(self):
        self.player.pause()
        if launch_manual_review(
            self, self.job, self.video_path, self.app_data_dir,
            self.config_module, self.object_picker_module,
        ):
            # A manual re-score for this exact video was just written --
            # the automated flag has been addressed either way, so clear
            # it rather than leaving it to nag again next time.
            self._clear_active_flag()
        # Either way, the point of "Analyze Manually" is to move on from
        # the validation-video viewer to the manual-review workflow --
        # close it once that's done (whether finalized, canceled, or it
        # couldn't be opened at all).
        self.accept()

    def _on_swap_objects(self):
        self.player.pause()
        warning_extra = ""
        if self._output_path and self._rat_id is not None:
            if excel_writer.has_human_review(self._output_path, self._rat_id, self.video_path.stem):
                warning_extra = "\n\nThis will discard the existing manual review for this video."
            elif excel_writer.has_computed_reference(self._output_path, self._rat_id, self.video_path.stem):
                warning_extra = "\n\nThis will discard the displaced computed-reference data for this video."

        answer = QMessageBox.question(
            self, "Swap novel/original",
            f"Swap which object is novel vs. original for '{self.video_path.name}', then re-score "
            f"it and regenerate its validation video using the existing tracking data (no "
            f"re-tracking needed).{warning_extra}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        # Release the video file handle before regenerating it on disk,
        # and close the viewer afterward either way -- the validation
        # video on screen no longer matches whatever's now on disk (new
        # labels, possibly a re-rendered file), so there's nothing useful
        # left to keep displaying here.
        self.player.release()
        swap_video_objects(
            self, self.job, self.video_path, self.app_data_dir,
            self.config_module, self.object_picker_module,
        )
        self.accept()

    def closeEvent(self, event):
        self.player.release()
        super().closeEvent(event)


class BatchReviewDialog(QDialog):
    """Table of every video across `jobs`, color coded red/yellow/green
    by review status (see this module's docstring). Double-click (or
    "Open Video...") launches FlaggedVideoViewer for that row; the table
    refreshes when it closes, since that dialog may have cleared a flag
    or added a manual review."""

    def __init__(self, jobs, app_data_dir, config_module, object_picker_module, parent=None):
        super().__init__(parent)
        self.jobs = list(jobs)
        self.app_data_dir = app_data_dir
        self.config_module = config_module
        self.object_picker_module = object_picker_module
        self._rows = []  # row index -> dict, see _refresh()

        self.setWindowTitle("Video Review")
        self.resize(820, 440)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Job", "Video", "Status"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.doubleClicked.connect(self._on_open_selected)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_context_menu)

        self.open_btn = QPushButton("Open Video…")
        self.close_btn = QPushButton("Close")
        self.open_btn.clicked.connect(self._on_open_selected)
        self.close_btn.clicked.connect(self.accept)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.open_btn)
        btn_row.addWidget(self.close_btn)

        legend = QLabel(
            f'<span style="color:{_COLOR_FLAGGED.name()};">red = flagged, needs review</span> &nbsp;&nbsp; '
            f'<span style="color:{_COLOR_MANUAL.name()};">yellow = manually analyzed</span> &nbsp;&nbsp; '
            f'<span style="color:{_COLOR_CLEAR.name()};">green = clear</span>'
        )

        layout = QVBoxLayout(self)
        layout.addWidget(legend)
        layout.addWidget(self.table)
        layout.addLayout(btn_row)

        self._refresh()

    def _refresh(self):
        self._rows = []
        for job in self.jobs:
            job_cfg = build_job_cfg(job, self.config_module)
            flags = review_flags.load_flags(job.output_folder)
            output_path = job.output_folder / f"{job.group_name}_NOR_results.xlsx"
            validation_folder = job.output_folder / "validation_videos"

            for video_path in find_videos(job.input_folder):
                stem = video_path.stem
                entry = flags.get(stem, {})
                reasons = entry.get("reasons", [])
                flagged_active = bool(reasons) and not entry.get("reviewed")

                rat_id, _phase, _session = _parse_filename(video_path, job_cfg)
                manually_analyzed = excel_writer.has_human_review(output_path, rat_id, stem)
                has_reference = excel_writer.has_computed_reference(output_path, rat_id, stem)

                if manually_analyzed:
                    color = _COLOR_MANUAL
                elif flagged_active:
                    color = _COLOR_FLAGGED
                else:
                    color = _COLOR_CLEAR

                self._rows.append({
                    "job": job,
                    "video_path": video_path,
                    "validation_path": validation_folder / f"{stem}.validation.mp4",
                    "reasons": reasons,
                    "flagged_active": flagged_active,
                    "manually_analyzed": manually_analyzed,
                    "has_reference": has_reference,
                    "output_path": output_path,
                    "rat_id": rat_id,
                    "session_label": stem,
                    "color": color,
                })

        if not self._rows:
            self.table.setRowCount(1)
            self.table.setItem(0, 0, QTableWidgetItem(""))
            self.table.setItem(0, 1, QTableWidgetItem("No videos found."))
            self.table.setItem(0, 2, QTableWidgetItem(""))
            self.open_btn.setEnabled(False)
            return

        self.table.setRowCount(len(self._rows))
        for i, row in enumerate(self._rows):
            brush = QBrush(row["color"])
            values = (row["job"].group_name, row["video_path"].name, self._status_text(row))
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setForeground(brush)
                self.table.setItem(i, col, item)
        self.open_btn.setEnabled(True)

    @staticmethod
    def _status_text(row):
        if row["manually_analyzed"]:
            if row["reasons"]:
                return "Manually analyzed (was flagged: " + "; ".join(row["reasons"]) + ")"
            return "Manually analyzed"
        if row["flagged_active"]:
            return "; ".join(row["reasons"])
        if row["reasons"]:
            return "Cleared -- " + "; ".join(row["reasons"])
        return "No issues found"

    def _selected_row(self):
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        if not rows or not self._rows or rows[0] >= len(self._rows):
            return None
        return self._rows[rows[0]]

    def _on_open_selected(self):
        row = self._selected_row()
        if row is not None:
            self._open_row(row)

    def _open_row(self, row):
        if not row["validation_path"].exists():
            QMessageBox.information(
                self, "No validation video",
                f"No validation video found yet for '{row['video_path'].name}' -- run the "
                f"batch for this job (with validation video export enabled) first.",
            )
            return
        viewer = FlaggedVideoViewer(
            row["job"], row["video_path"], row["validation_path"], row["reasons"], self.app_data_dir,
            self.config_module, self.object_picker_module, parent=self,
        )
        viewer.exec()
        self._refresh()

    def _analyze_row_manually(self, row):
        launch_manual_review(
            self, row["job"], row["video_path"], self.app_data_dir,
            self.config_module, self.object_picker_module,
        )
        # Whether it was finalized or canceled, the workbook or flag state
        # may have changed (a finalize captures a fresh computed
        # reference the first time), so always refresh.
        self._refresh()

    def _swap_row_objects(self, row):
        warning_extra = ""
        if row["manually_analyzed"]:
            warning_extra = "\n\nThis will discard the existing manual review for this video."
        elif row["has_reference"]:
            warning_extra = "\n\nThis will discard the displaced computed-reference data for this video."

        answer = QMessageBox.question(
            self, "Swap novel/original",
            f"Swap which object is novel vs. original for '{row['video_path'].name}', then re-score "
            f"it and regenerate its validation video using the existing tracking data (no "
            f"re-tracking needed).{warning_extra}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        swap_video_objects(
            self, row["job"], row["video_path"], self.app_data_dir,
            self.config_module, self.object_picker_module,
        )
        self._refresh()

    def _revert_row(self, row):
        answer = QMessageBox.question(
            self, "Revert to computed data",
            f"This will discard the manual review for '{row['video_path'].name}' and "
            f"restore the original automated result. This can't be undone. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            excel_writer.revert_to_computed(row["output_path"], row["rat_id"], row["session_label"])
        except ValueError as exc:
            QMessageBox.warning(self, "Couldn't revert", str(exc))
            return
        self._refresh()

    def _on_table_context_menu(self, pos):
        index = self.table.indexAt(pos)
        if not index.isValid():
            return
        self.table.selectRow(index.row())
        row = self._selected_row()
        if row is None:
            return

        menu = QMenu(self)
        menu.addAction("Open", lambda: self._open_row(row))
        menu.addAction("Analyze Manually…", lambda: self._analyze_row_manually(row))
        menu.addAction("Swap Novel/Original…", lambda: self._swap_row_objects(row))
        revert_action = menu.addAction("Revert to Computed Data", lambda: self._revert_row(row))
        revert_action.setEnabled(row["has_reference"])
        menu.exec(self.table.viewport().mapToGlobal(pos))
