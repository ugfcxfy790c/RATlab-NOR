"""
Main window shell for the NOR classifier GUI.

Job queue table, add/remove controls, run/stop controls, and a log panel.
"Add Job" (task #3) uses native OS folder pickers via AddJobDialog.
"Run Queue" / "Stop" and "Set Up Objects" are still stubs -- tasks #4/#5
wire up the real object-picker launch and background batch runner.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QModelIndex
from PySide6.QtGui import QActionGroup, QTextCursor
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTableView,
    QPushButton, QPlainTextEdit, QDialog, QMessageBox, QAbstractItemView,
    QSplitter, QApplication, QFileDialog, QProgressBar, QLabel, QInputDialog, QMenu,
)

from job_queue import (
    Job, JobQueue, JobStatus, refresh_job_readiness,
    job_coords_path, job_predictions_folder, find_videos,
)
from job_table_model import JobTableModel
from add_job_dialog import AddJobDialog
from job_config import build_job_cfg
from object_setup_dialog import run_object_setup
from manual_review_dialog import ManualReviewDialog
import batch_review_dialog
from batch_runner import BatchRunner
from crop_setup_dialog import CropSetupDialog
from settings_dialog import SettingsDialog
import reset as reset_module
import app_settings
import sleap_inference
import video_crop
from pose_utils import load_track_from_slp

# Jobs in these statuses haven't run yet, so it's safe to change their
# settings. Anything past that (running/done/failed/canceled) is left
# alone -- editing a job that already produced output, or is mid-run,
# would be confusing at best.
EDITABLE_STATUSES = (JobStatus.NEEDS_SETUP, JobStatus.READY)

# Jobs that ended without producing a usable result (crashed, hit an
# error mid-run, interrupted by the app closing, or stopped by the
# user). "Run Queue" offers to retry these -- reset back to NEEDS_SETUP
# and immediately re-checked, so a job whose object coordinates are
# still on disk goes straight back to READY without redoing setup.
RETRIABLE_STATUSES = (JobStatus.FAILED, JobStatus.CANCELED)


# --- Main window -------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, job_queue: JobQueue, object_picker_module, default_output_base: str, config_module):
        super().__init__()
        self.job_queue = job_queue
        self.object_picker_module = object_picker_module
        self.default_output_base = default_output_base
        self.config_module = config_module
        self.runner: BatchRunner | None = None

        # Overall-queue progress tracking (across every job in the current
        # run, not just the job currently executing) -- see _start_run/
        # _on_job_progress/_update_progress_ui. Keyed by job id so a job's
        # count can be corrected once its real video total is known
        # (job.videos_total is a readiness-time estimate; the worker's
        # first job_progress message for that job carries the actual
        # count, which can differ slightly, e.g. videos removed from disk
        # after the last readiness check).
        self._run_total_by_job: dict[str, int] = {}
        self._run_done_by_job: dict[str, int] = {}
        self._run_start_time: float | None = None
        # Wall-clock time of the last video completion (or run start, if
        # none yet) -- see _update_progress_ui for why this is tracked
        # separately from _run_start_time.
        self._run_last_progress_time: float | None = None

        # Ticks _update_progress_ui() once a second while a run is in
        # flight, independent of job_progress events -- those only arrive
        # when a whole video finishes, which for a slow model can be
        # minutes apart, making the ETA look frozen/stuck in between even
        # though time (and therefore the ETA estimate) is moving. Started
        # in _start_run, stopped in _on_runner_thread_finished.
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(1000)
        self._progress_timer.timeout.connect(self._update_progress_ui)

        # Whether the last line written to the log panel is a live
        # progress-bar redraw (as opposed to a discrete log message) --
        # see _on_progress_line/_log. Lets repeated redraws of "the same"
        # tqdm line overwrite each other in place instead of each one
        # adding a new line.
        self._progress_line_open = False

        self.setWindowTitle("RATlab NOR")
        self.resize(980, 560)

        self.model = JobTableModel(job_queue)

        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_context_menu)

        self.log_panel = QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setMaximumBlockCount(5000)
        self.log_panel.setPlaceholderText("Run log will appear here…")

        self.add_btn = QPushButton("Add Job…")
        self.edit_btn = QPushButton("Edit Job…")
        self.remove_btn = QPushButton("Remove Selected")
        self.setup_btn = QPushButton("Set Up Objects…")
        self.run_btn = QPushButton("Run Queue")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        self.add_btn.clicked.connect(self._on_add_job)
        self.edit_btn.clicked.connect(self._on_edit_job)
        self.remove_btn.clicked.connect(self._on_remove_selected)
        self.setup_btn.clicked.connect(self._on_setup_objects)
        self.run_btn.clicked.connect(self._on_run_queue)
        self.stop_btn.clicked.connect(self._on_stop_queue)
        self.table.doubleClicked.connect(lambda _index: self._on_edit_job())

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.edit_btn)
        btn_row.addWidget(self.remove_btn)
        btn_row.addWidget(self.setup_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)

        # Overall-queue progress bar + ETA, shown only while a batch run is
        # in flight (see _start_run / _on_runner_thread_finished). Counts
        # videos across every job in the run, not just the running job --
        # a per-job bar alone doesn't tell you how much of an overnight
        # batch is actually left.
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m videos (%p%)")
        self.progress_bar.setVisible(False)
        self.eta_label = QLabel("")
        self.eta_label.setVisible(False)

        progress_row = QHBoxLayout()
        progress_row.addWidget(self.progress_bar, stretch=1)
        progress_row.addWidget(self.eta_label)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.addLayout(btn_row)
        top_layout.addLayout(progress_row)
        top_layout.addWidget(self.table)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(self.log_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.addWidget(splitter)
        self.setCentralWidget(central)

        self.statusBar().showMessage(f"{len(self.job_queue.jobs)} job(s) queued")
        self._log(f"Loaded queue from {self.job_queue.path}")

        self._build_menu_bar()

        # Deferred rather than called directly here so the popup (if any)
        # appears after the main window is actually visible on screen --
        # this fires on the first event-loop iteration, which happens
        # after app.py's window.show() but before anything else can steal
        # focus.
        QTimer.singleShot(0, self._check_sleap_available)

    # -- startup checks --

    def _check_sleap_available(self):
        """Every batch run needs a usable sleap-nn/sleap/sleap-track on
        PATH -- nag about it once at startup rather than letting someone
        queue up a whole overnight batch and only find out it's missing
        when the first job immediately fails."""
        try:
            sleap_inference.check_sleap_track_available()
        except sleap_inference.SleapTrackNotFound:
            QMessageBox.warning(
                self, "SLEAP not found",
                "No usable SLEAP install (sleap-nn, sleap, or sleap-track) was "
                "found on PATH -- batch runs will fail until one's installed.\n\n"
                + sleap_inference.sleap_install_instructions(),
            )

    # -- menu bar --

    def _build_menu_bar(self):
        """Menu order follows the rough workflow: "Model" (which model to
        run inference with), "Settings" (output/scoring defaults --
        see settings_dialog.py), "Video" (crop footage to what that model
        expects), "Run" (menu equivalents of the Run Queue/Stop buttons),
        then "Reset": per-selected-job resets (the usual case -- each
        job's own coordinate/prediction cache and output folder under
        nor_classifier/app_data/), plus a "Legacy (CLI)" submenu wrapping
        reset.py's original single-folder cache clearing (config.py's
        PREDICTIONS_FOLDER/OUTPUT_FOLDER/OBJECT_COORDS_FILE, used by
        main.py's setup/run commands, not by any GUI job)."""
        self._build_model_menu()
        self._build_settings_menu()
        self._build_video_menu()
        self._build_run_menu()
        self._build_review_menu()

        reset_menu = self.menuBar().addMenu("Reset")

        self.reset_job_predictions_action = reset_menu.addAction(
            "Reset Selected Job(s): Predictions + Output…",
            lambda: self._on_reset_job(clear_predictions=True, clear_output=True, clear_coords=False),
        )
        self.reset_job_output_action = reset_menu.addAction(
            "Reset Selected Job(s): Output Only…",
            lambda: self._on_reset_job(clear_predictions=False, clear_output=True, clear_coords=False),
        )
        self.reset_job_coords_action = reset_menu.addAction(
            "Reset Selected Job(s): Object Coordinates…",
            lambda: self._on_reset_job(clear_predictions=False, clear_output=False, clear_coords=True),
        )
        self.reset_job_all_action = reset_menu.addAction(
            "Reset Selected Job(s): Everything…",
            lambda: self._on_reset_job(clear_predictions=True, clear_output=True, clear_coords=True),
        )

        reset_menu.addSeparator()
        legacy_menu = reset_menu.addMenu("Legacy (CLI)")
        self.reset_predictions_action = legacy_menu.addAction(
            "Reset Predictions + Output…",
            lambda: self._on_reset(clear_predictions=True, clear_output=True, clear_coords=False),
        )
        self.reset_output_action = legacy_menu.addAction(
            "Reset Output Only…",
            lambda: self._on_reset(clear_predictions=False, clear_output=True, clear_coords=False),
        )
        self.reset_coords_action = legacy_menu.addAction(
            "Reset Object Coordinates…",
            lambda: self._on_reset(clear_predictions=False, clear_output=False, clear_coords=True),
        )
        self.reset_all_action = legacy_menu.addAction(
            "Reset Everything…",
            lambda: self._on_reset(clear_predictions=True, clear_output=True, clear_coords=True),
        )

        self._reset_actions = [
            self.reset_job_predictions_action, self.reset_job_output_action,
            self.reset_job_coords_action, self.reset_job_all_action,
            self.reset_predictions_action, self.reset_output_action,
            self.reset_coords_action, self.reset_all_action,
        ]

    def _build_model_menu(self):
        """"Model" menu: pick which model subfolder under RATlab/models/
        new inference runs use. Applies app-wide to any job that doesn't
        set its own MODEL_PATHS override (see job_config.build_job_cfg's
        precedence order); persisted to app_data/settings.json so it
        survives restarts."""
        model_menu = self.menuBar().addMenu("Model")
        models = app_settings.list_available_models(self.config_module.RATLAB_DIR)

        if not models:
            action = model_menu.addAction("(no models found in RATlab/models/)")
            action.setEnabled(False)
            return

        current = app_settings.get_selected_model(self.config_module.NOR_CLASSIFIER_DIR)
        if current not in models:
            # Nothing saved yet (or the saved choice no longer exists) --
            # default to whichever model config.py currently points at.
            configured_names = {Path(p).name for p in self.config_module.MODEL_PATHS}
            current = next((m for m in models if m in configured_names), models[0])

        self._model_action_group = QActionGroup(self)
        self._model_action_group.setExclusive(True)
        for name in models:
            action = model_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == current)
            action.triggered.connect(lambda checked, n=name: self._on_select_model(n))
            self._model_action_group.addAction(action)

    def _on_select_model(self, model_name: str):
        app_settings.set_selected_model(self.config_module.NOR_CLASSIFIER_DIR, model_name)
        self._log(f"Model set to '{model_name}' -- applies to any job that doesn't set its own model override.")

    def _build_settings_menu(self):
        settings_menu = self.menuBar().addMenu("Settings")
        settings_menu.addAction("Preferences…", self._on_open_settings)

    def _on_open_settings(self):
        dialog = SettingsDialog(self.config_module, parent=self)
        if dialog.exec() == QDialog.Accepted:
            self.default_output_base = app_settings.get_default_output_base(
                self.config_module.NOR_CLASSIFIER_DIR, self.config_module,
            )
            self._log("Settings saved.")

    def _build_video_menu(self):
        """"Video" menu: tools for getting footage into the pixel size a
        model actually expects (config.py's CROP_TARGET_WIDTH/HEIGHT) --
        "Check Resolutions" is just a quick read-only scan; "Crop Videos"
        opens the per-video crop walkthrough (video_crop.py +
        crop_setup_dialog.py)."""
        video_menu = self.menuBar().addMenu("Video")
        video_menu.addAction("Check Resolutions…", self._on_check_resolutions)
        video_menu.addAction("Crop Videos…", self._on_crop_videos)

    def _on_check_resolutions(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder of videos to check", str(Path.home()))
        if not folder:
            return
        groups = video_crop.scan_resolutions(folder)
        if not groups:
            QMessageBox.information(self, "No videos found", f"No .mp4 files found under:\n{folder}")
            return

        target = (self.config_module.CROP_TARGET_WIDTH, self.config_module.CROP_TARGET_HEIGHT)
        lines = []
        for (w, h), videos in sorted(groups.items()):
            if (w, h) == (-1, -1):
                lines.append(f"  - unreadable: {len(videos)} video(s)")
            else:
                tag = "  <- matches model target" if (w, h) == target else ""
                lines.append(f"  - {w}x{h}: {len(videos)} video(s){tag}")

        message = f"Model expects {target[0]}x{target[1]}.\n\nFound under {folder}:\n\n" + "\n".join(lines)
        self._log(f"Checked resolutions under {folder}:")
        for line in lines:
            self._log(line)
        QMessageBox.information(self, "Video resolutions", message)

    def _on_crop_videos(self):
        input_folder = QFileDialog.getExistingDirectory(self, "Choose the folder of videos to crop", str(Path.home()))
        if not input_folder:
            return
        videos = video_crop.find_videos(input_folder)
        if not videos:
            QMessageBox.information(self, "No videos found", f"No .mp4 files found under:\n{input_folder}")
            return

        output_folder = QFileDialog.getExistingDirectory(self, "Choose where cropped videos should be written", str(Path.home()))
        if not output_folder:
            return
        if Path(input_folder).resolve() == Path(output_folder).resolve():
            QMessageBox.warning(self, "Same folder", "Output folder must be different from the input folder.")
            return

        dialog = CropSetupDialog(
            videos, input_folder, output_folder, self.config_module.NOR_CLASSIFIER_DIR,
            self.config_module.CROP_TARGET_WIDTH, self.config_module.CROP_TARGET_HEIGHT,
            parent=self,
        )
        dialog.exec()

    def _build_run_menu(self):
        """Menu equivalents of the Run Queue/Stop buttons, plus "Run
        Selected Job(s)" -- runs only the jobs currently selected in the
        table, rather than every Ready job in the queue."""
        run_menu = self.menuBar().addMenu("Run")

        self.run_queue_menu_action = run_menu.addAction("Run Queue", self._on_run_queue)
        self.run_selected_menu_action = run_menu.addAction("Run Selected Job(s)", self._on_run_selected)
        run_menu.addSeparator()
        self.stop_menu_action = run_menu.addAction("Stop", self._on_stop_queue)
        self.stop_menu_action.setEnabled(False)

        self._run_actions = [self.run_queue_menu_action, self.run_selected_menu_action]

    def _build_review_menu(self):
        """"Review" menu: "Review Videos..." lists every video across
        every job in the queue, color coded (red = review_flags.py
        flagged it and it's unreviewed; yellow = already manually
        analyzed; green = clear -- see batch_review_dialog.py). Also
        opens automatically right after a batch run if it left anything
        newly flagged -- see _on_all_finished. "Manual Review Video..."
        is the same one-video re-scoring utility (manual_review_dialog.py)
        but reachable directly, for a specific video without going
        through the review table first. Both are interactive,
        one-video-at-a-time workflows -- separate from the queued/
        unattended Run menu."""
        review_menu = self.menuBar().addMenu("Review")
        review_menu.addAction("Review Videos…", self._on_review_videos)
        review_menu.addAction("Manual Review Video…", self._on_manual_review)

    def _on_review_videos(self):
        self._open_batch_review(list(self.job_queue.jobs))

    def _open_batch_review(self, jobs):
        dialog = batch_review_dialog.BatchReviewDialog(
            jobs, self.job_queue.app_data_dir, self.config_module, self.object_picker_module, parent=self,
        )
        dialog.exec()

    def _on_manual_review(self):
        jobs = self._selected_jobs()
        if len(jobs) != 1:
            QMessageBox.information(
                self, "Select one job",
                "Select exactly one job in the table first, to review a video from.",
            )
            return
        job = jobs[0]

        videos = find_videos(job.input_folder)
        if not videos:
            QMessageBox.information(self, "No videos", f"No .mp4 files found in {job.input_folder}.")
            return

        predictions_folder = job_predictions_folder(job, self.job_queue.app_data_dir)
        coords_path = job_coords_path(job, self.job_queue.app_data_dir)
        coords = self.object_picker_module.load_object_coords(coords_path)

        reviewable = []
        for v in videos:
            slp_path = sleap_inference.predictions_path_for(v, predictions_folder)
            key = self.object_picker_module.video_key(v, job.input_folder)
            if slp_path.exists() and key in coords:
                reviewable.append(v)

        if not reviewable:
            QMessageBox.information(
                self, "Nothing to review",
                f"'{job.group_name}' has no videos with both tracking predictions and "
                f"object coordinates yet. Run the batch (or at least Set Up Objects, then "
                f"a run) for this job before manually reviewing one of its videos.",
            )
            return

        names = [v.name for v in reviewable]
        chosen_name, ok = QInputDialog.getItem(
            self, "Manual Review", "Video to review:", names, editable=False,
        )
        if not ok:
            return
        video_path = reviewable[names.index(chosen_name)]

        job_cfg = build_job_cfg(job, self.config_module)
        key = self.object_picker_module.video_key(video_path, job.input_folder)
        object_coords = coords[key]
        object_names = list(object_coords.keys())

        slp_path = sleap_inference.predictions_path_for(video_path, predictions_folder)
        try:
            track = load_track_from_slp(slp_path, video_path=video_path)
        except Exception as exc:
            QMessageBox.critical(self, "Couldn't load tracking data", f"Failed to load {slp_path.name}:\n{exc}")
            return

        rat_id, _phase, _session = self._parse_filename(video_path, job_cfg)
        output_path = job.output_folder / f"{job.group_name}_NOR_results.xlsx"

        try:
            dialog = ManualReviewDialog(
                video_path, track, object_coords, job_cfg, object_names,
                output_path, rat_id, video_path.stem, parent=self,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Couldn't open video", f"Failed to open {video_path.name}:\n{exc}")
            return

        if dialog.exec() == QDialog.Accepted:
            self._log(f"Manual review saved for '{video_path.name}' -> {output_path.name}")

    @staticmethod
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

    def _on_reset_job(self, clear_predictions: bool, clear_output: bool, clear_coords: bool):
        jobs = self._selected_jobs()
        if not jobs:
            QMessageBox.information(self, "No selection", "Select one or more jobs first.")
            return

        targetable = [j for j in jobs if j.status != JobStatus.RUNNING]
        for j in jobs:
            if j.status == JobStatus.RUNNING:
                self._log(f"Skipping '{j.group_name}' -- it's currently running.")
        if not targetable:
            return

        lines = []
        for job in targetable:
            lines.append(f"'{job.group_name}':")
            lines.extend(
                f"    {t}" for t in reset_module.describe_job_targets(
                    job, self.job_queue.app_data_dir, clear_predictions, clear_output, clear_coords,
                )
            )
        message = "This will permanently delete:\n\n" + "\n".join(lines)
        reply = QMessageBox.warning(
            self, "Confirm reset", message,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        for job in targetable:
            reset_module.perform_job_reset(
                job, self.job_queue.app_data_dir, clear_predictions, clear_output, clear_coords, log=self._log,
            )
            job.error_message = ""
            job.videos_done = 0
            job.status = JobStatus.NEEDS_SETUP  # force refresh_job_readiness to recompute cleanly below
            refresh_job_readiness(job, self.job_queue.app_data_dir, self.object_picker_module)
            self.job_queue.update(job)
            self._log(f"'{job.group_name}' reset -- now {job.status.value}.")

        self.model.refresh()
        self._update_status_bar()

    def _on_reset(self, clear_predictions: bool, clear_output: bool, clear_coords: bool):
        lines = reset_module.describe_targets(self.config_module, clear_predictions, clear_output, clear_coords)
        message = "This will permanently delete:\n\n" + "\n".join(f"  - {line}" for line in lines)
        message += "\n\nThis affects the legacy single-folder CLI workflow's cache, not any GUI job's own data."
        reply = QMessageBox.warning(
            self, "Confirm reset", message,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._log(f"Resetting (predictions={clear_predictions}, output={clear_output}, coords={clear_coords})…")
        reset_module.perform_reset(self.config_module, clear_predictions, clear_output, clear_coords, log=self._log)
        self._log("Reset complete.")

    # -- logging --

    def _log(self, message: str):
        self.log_panel.appendPlainText(message)
        # A discrete message always ends any progress-bar redraw in
        # progress -- the next one (a different video's tracking, most
        # likely) should start its own new line rather than overwriting
        # whatever was just logged here.
        self._progress_line_open = False

    def _on_progress_line(self, message: str):
        """Live tracking-progress updates (see sleap_inference.py's
        `progress` callback) -- redraws the same line in place, the same
        way the tqdm bar this mirrors does in a real terminal, instead of
        adding a new line to the log for every redraw."""
        if self._progress_line_open:
            cursor = self.log_panel.textCursor()
            cursor.movePosition(QTextCursor.End)
            # Selects from the end of the document back to the start of
            # the last line (not the whole document) -- KeepAnchor stops
            # the selection from also eating the newline before it, so
            # only that one line's text gets replaced.
            cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertText(message)
            self.log_panel.setTextCursor(cursor)
        else:
            self.log_panel.appendPlainText(message)
            self._progress_line_open = True
        self.log_panel.ensureCursorVisible()

    # -- selection helpers --

    def _selected_jobs(self) -> list[Job]:
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        return [self.model.job_at(r) for r in rows]

    def _on_table_context_menu(self, pos):
        """Right-click menu on the job table: Edit / Set Up Objects / Run
        / Review / Reset, all scoped to whichever job(s) the click
        applies to -- same underlying handlers as the toolbar buttons and
        menu bar actions, just reachable without leaving the row."""
        index = self.table.indexAt(pos)
        if not index.isValid():
            return

        # Right-clicking a row outside the current selection selects just
        # that row first (standard table convention) -- right-clicking
        # within an existing multi-row selection leaves it alone, so a
        # batch action (e.g. Reset) can still apply to all of them.
        selection_model = self.table.selectionModel()
        if not selection_model.isRowSelected(index.row(), QModelIndex()):
            self.table.selectRow(index.row())

        jobs = self._selected_jobs()
        if not jobs:
            return

        menu = self._build_job_context_menu(jobs)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _build_job_context_menu(self, jobs: list[Job]) -> QMenu:
        """Split out from _on_table_context_menu so it can be constructed
        (and its contents inspected) without also invoking the blocking,
        modal QMenu.exec() -- handy for tests, and keeps the "what's in
        the menu" logic separate from "where/when it pops up"."""
        menu = QMenu(self)
        menu.addAction("Edit Job…", self._on_edit_job)
        menu.addAction("Set Up Objects…", self._on_setup_objects)
        menu.addSeparator()
        menu.addAction("Run Selected Job(s)", self._on_run_selected)
        menu.addAction("Review Videos…", lambda: self._open_batch_review(jobs))
        menu.addSeparator()

        reset_menu = menu.addMenu("Reset")
        reset_menu.addAction(self.reset_job_predictions_action)
        reset_menu.addAction(self.reset_job_output_action)
        reset_menu.addAction(self.reset_job_coords_action)
        reset_menu.addAction(self.reset_job_all_action)

        return menu

    # -- actions --

    def _on_add_job(self):
        existing_names = {j.group_name for j in self.job_queue.jobs}
        default_skip_validation = app_settings.get_skip_validation_default(self.config_module.NOR_CLASSIFIER_DIR)
        dialog = AddJobDialog(
            self.default_output_base, existing_names, parent=self,
            default_skip_validation=default_skip_validation,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        job = Job(**dialog.result_job_kwargs)
        refresh_job_readiness(job, self.job_queue.app_data_dir, self.object_picker_module)
        self.job_queue.add(job)
        self.model.refresh()
        self._log(f"Added job '{job.group_name}' ({job.status.value}).")
        self._update_status_bar()

    def _on_edit_job(self):
        jobs = self._selected_jobs()
        if len(jobs) != 1:
            QMessageBox.information(self, "Select one job", "Select exactly one job to edit.")
            return
        job = jobs[0]
        if job.status not in EDITABLE_STATUSES:
            QMessageBox.warning(
                self, "Can't edit this job",
                f"'{job.group_name}' is {job.status.value} -- only jobs that haven't run yet "
                f"(Needs setup / Ready) can be edited.",
            )
            return

        existing_names = {j.group_name for j in self.job_queue.jobs if j.id != job.id}
        dialog = AddJobDialog(self.default_output_base, existing_names, parent=self, existing_job=job)
        if dialog.exec() != QDialog.Accepted:
            return

        kwargs = dialog.result_job_kwargs
        old_group_name = job.group_name
        old_input_folder = job.input_folder
        old_coords_path = job_coords_path(job, self.job_queue.app_data_dir)
        old_predictions_folder = job_predictions_folder(job, self.job_queue.app_data_dir)

        job.group_name = kwargs["group_name"]
        job.input_folder = kwargs["input_folder"]
        job.output_base_folder = kwargs["output_base_folder"]
        job.skip_validation = kwargs["skip_validation"]

        # The coords/predictions cache is namespaced by job id + a slug of
        # the group name (see job_queue.py) so two jobs never collide --
        # but that means renaming the group also moves that cache. Move
        # the files rather than losing already-clicked object positions.
        if job.group_name != old_group_name:
            new_coords_path = job_coords_path(job, self.job_queue.app_data_dir)
            if old_coords_path.exists() and not new_coords_path.exists():
                new_coords_path.parent.mkdir(parents=True, exist_ok=True)
                old_coords_path.rename(new_coords_path)

            new_predictions_folder = job_predictions_folder(job, self.job_queue.app_data_dir)
            if old_predictions_folder.exists() and not new_predictions_folder.exists():
                new_predictions_folder.parent.mkdir(parents=True, exist_ok=True)
                old_predictions_folder.rename(new_predictions_folder)

        # If the input folder changed, previously-set coordinates may no
        # longer apply to the new set of videos -- force a re-check rather
        # than trusting the old READY status.
        if job.input_folder != old_input_folder:
            job.status = JobStatus.NEEDS_SETUP
        refresh_job_readiness(job, self.job_queue.app_data_dir, self.object_picker_module)

        self.job_queue.update(job)
        self.model.refresh()
        self._log(f"Updated job '{job.group_name}' ({job.status.value}).")
        self._update_status_bar()

    def _retry_job(self, job: Job):
        """Reset a Failed/Canceled job back to NEEDS_SETUP and
        immediately re-check it -- if its object coordinates are still
        on disk (the usual case; a run failing doesn't erase setup
        work), it lands straight back on READY."""
        job.error_message = ""
        job.videos_done = 0
        job.status = JobStatus.NEEDS_SETUP  # so refresh_job_readiness below will recompute it
        refresh_job_readiness(job, self.job_queue.app_data_dir, self.object_picker_module)
        self.job_queue.update(job)

    def _on_remove_selected(self):
        jobs = self._selected_jobs()
        if not jobs:
            return
        for job in jobs:
            if job.status == JobStatus.RUNNING:
                QMessageBox.warning(self, "Job running", f"Can't remove '{job.group_name}' while it's running.")
                continue
            self.job_queue.remove(job.id)
            self._log(f"Removed job '{job.group_name}'.")
        self.model.refresh()
        self._update_status_bar()

    def _on_setup_objects(self):
        jobs = self._selected_jobs()
        if not jobs:
            QMessageBox.information(self, "No selection", "Select one or more jobs first.")
            return

        runnable = [j for j in jobs if j.status != JobStatus.RUNNING]
        for j in jobs:
            if j.status == JobStatus.RUNNING:
                self._log(f"Skipping '{j.group_name}' -- it's currently running.")
        if not runnable:
            return

        # Object setup opens a QDialog (see object_setup_dialog.py) -- a
        # real Qt window sharing the app's event loop, not a separate
        # cv2 window, so it behaves as a normal modal dialog rather than
        # freezing the rest of the app.
        self.setup_btn.setEnabled(False)
        try:
            for job in runnable:
                videos = find_videos(job.input_folder)
                if not videos:
                    self._log(f"'{job.group_name}': no .mp4 files found in {job.input_folder} -- skipping.")
                    continue

                self._log(f"Opening object setup for '{job.group_name}' ({len(videos)} video(s))…")
                self.log_panel.repaint()
                QApplication.processEvents()

                job_cfg = build_job_cfg(job, self.config_module)
                coords_path = job_coords_path(job, self.job_queue.app_data_dir)
                try:
                    run_object_setup(videos, coords_path, job_cfg, parent=self)
                except Exception as exc:
                    self._log(f"Object setup for '{job.group_name}' hit an error: {exc}")
                    QMessageBox.warning(
                        self, "Setup error",
                        f"Object setup for '{job.group_name}' hit an error:\n{exc}",
                    )
                    continue

                refresh_job_readiness(job, self.job_queue.app_data_dir, self.object_picker_module)
                self.job_queue.update(job)
                self.model.refresh()
                self._log(f"'{job.group_name}' is now {job.status.value}.")
        finally:
            self.setup_btn.setEnabled(True)
            self._update_status_bar()

    def _on_run_queue(self):
        self._start_run(list(self.job_queue.jobs))

    def _on_run_selected(self):
        jobs = self._selected_jobs()
        if not jobs:
            QMessageBox.information(self, "No selection", "Select one or more jobs to run first.")
            return
        self._start_run(jobs)

    def _start_run(self, candidate_jobs: list[Job]):
        """Shared by "Run Queue" (candidate_jobs = every job) and "Run
        Selected Job(s)" (candidate_jobs = just the current table
        selection) -- everything past the candidate list is identical:
        offer to retry any failed/canceled jobs among them, then start
        the batch runner on whichever of them are Ready."""
        if self.runner is not None:
            return  # already running

        retriable = [j for j in candidate_jobs if j.status in RETRIABLE_STATUSES]
        if retriable:
            lines = "\n".join(f"  - '{j.group_name}': {j.error_message or j.status.value}" for j in retriable)
            reply = QMessageBox.warning(
                self, "Retry failed/canceled jobs?",
                f"{len(retriable)} job(s) previously failed or were canceled:\n\n{lines}\n\n"
                f"If whatever caused that hasn't been fixed, retrying will likely fail the same way. "
                f"Retry them as part of this run?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                for job in retriable:
                    self._retry_job(job)
                    self._log(f"'{job.group_name}' reset for retry -- now {job.status.value}.")
                self.model.refresh()

        runnable = [j for j in candidate_jobs if j.status == JobStatus.READY]
        if not runnable:
            QMessageBox.information(
                self, "Nothing to run",
                "None of these jobs are marked Ready. Finish object setup on any 'Needs setup' jobs first.",
            )
            return

        self._set_queue_editable(False)
        self._set_run_actions_enabled(False)
        self._set_stop_enabled(True)
        self._log(f"Starting batch run of {len(runnable)} job(s): {', '.join(j.group_name for j in runnable)}")

        # Seed per-job totals from the readiness-time video count so the
        # bar shows a sensible range immediately, before any job_progress
        # message has arrived for a given job -- _on_job_progress corrects
        # each job's entry to the worker's actual count as soon as it starts.
        self._run_total_by_job = {j.id: j.videos_total for j in runnable}
        self._run_done_by_job = {j.id: 0 for j in runnable}
        self._run_start_time = time.monotonic()
        self._run_last_progress_time = self._run_start_time
        self.progress_bar.setRange(0, max(sum(self._run_total_by_job.values()), 1))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.eta_label.setVisible(True)
        self.eta_label.setText("ETA: calculating…")
        self._progress_timer.start()
        self._progress_line_open = False

        # Remembered so _on_all_finished knows which jobs to check for
        # freshly flagged videos once the run completes.
        self._last_run_jobs = list(runnable)

        self.runner = BatchRunner(runnable, self.job_queue.app_data_dir, self.config_module.NOR_CLASSIFIER_DIR)
        self.runner.log.connect(self._log)
        self.runner.progress.connect(self._on_progress_line)
        self.runner.job_started.connect(self._on_job_started)
        self.runner.job_progress.connect(self._on_job_progress)
        self.runner.job_finished.connect(self._on_job_finished)
        self.runner.finished.connect(self._on_runner_thread_finished)  # QThread's own signal
        self.runner.all_finished.connect(self._on_all_finished)
        self.runner.start()

    def _on_stop_queue(self):
        if self.runner is None:
            return
        self.runner.request_stop()
        self._set_stop_enabled(False)
        self._log("Stop requested -- finishing the current video, then halting.")

    def _set_stop_enabled(self, enabled: bool):
        self.stop_btn.setEnabled(enabled)
        if hasattr(self, "stop_menu_action"):
            self.stop_menu_action.setEnabled(enabled)

    # -- batch runner signal handlers (always run on the GUI thread) --

    def _on_job_started(self, job_id: str):
        job = self.job_queue.get(job_id)
        if job is None:
            return
        job.status = JobStatus.RUNNING
        job.error_message = ""
        self.job_queue.update(job)
        self.model.refresh()
        self._update_status_bar()

    def _on_job_progress(self, job_id: str, done: int, total: int):
        job = self.job_queue.get(job_id)
        if job is None:
            return
        job.videos_done = done
        job.videos_total = total
        self.job_queue.update(job)
        self.model.refresh()

        if job_id in self._run_done_by_job:
            if done != self._run_done_by_job[job_id]:
                # A video just finished -- this is the one and only
                # moment avg-per-video should be allowed to move (see
                # _update_progress_ui). Recorded even if the periodic
                # timer would've ticked anyway, so the "time into the
                # next video" clock starts fresh from right now.
                self._run_last_progress_time = time.monotonic()
            self._run_done_by_job[job_id] = done
            self._run_total_by_job[job_id] = total
            self._update_progress_ui()

    def _update_progress_ui(self):
        total = sum(self._run_total_by_job.values())
        done = sum(self._run_done_by_job.values())

        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(done)

        if self._run_start_time is None or self._run_last_progress_time is None:
            return
        now = time.monotonic()
        elapsed_at_last_completion = self._run_last_progress_time - self._run_start_time
        # Require a little elapsed time and at least one finished video
        # before trusting a rate -- otherwise the first video's own
        # (possibly atypical, e.g. cold model load) duration dominates
        # the estimate.
        if done > 0 and elapsed_at_last_completion > 2:
            # Average pace is only recomputed at the instant a video
            # finishes (elapsed_at_last_completion), not against the
            # freely-running clock -- otherwise, every second that ticks
            # by *within* a still-running video shrinks the apparent
            # rate (same done, bigger elapsed) and the ETA visibly counts
            # up instead of down. Instead, the average pace stays fixed
            # between completions, and only the "how far into the
            # current video are we" term ticks down live.
            avg_per_video = elapsed_at_last_completion / done
            time_into_current = now - self._run_last_progress_time
            remaining_after_current = max(total - done - 1, 0)
            remaining_current = max(avg_per_video - time_into_current, 0)
            eta_seconds = remaining_after_current * avg_per_video + remaining_current
            self.eta_label.setText(
                f"{done}/{total} videos -- ETA {self._format_duration(eta_seconds)} remaining"
            )
        else:
            self.eta_label.setText(f"{done}/{total} videos -- ETA calculating…")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes}m"
        if minutes:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    def _on_job_finished(self, job_id: str, status_value: str, error_message: str):
        job = self.job_queue.get(job_id)
        if job is None:
            return
        job.status = JobStatus(status_value)
        job.error_message = error_message
        self.job_queue.update(job)
        self.model.refresh()
        suffix = f" -- {error_message}" if error_message else ""
        self._log(f"Job '{job.group_name}' finished: {job.status.value}{suffix}")
        self._update_status_bar()

    def _on_all_finished(self):
        self._log("Batch run finished.")
        jobs = getattr(self, "_last_run_jobs", [])
        if jobs and batch_review_dialog.has_active_flags(jobs):
            self._log("Some videos were flagged for review -- opening the review screen.")
            self._open_batch_review(jobs)

    def _on_runner_thread_finished(self):
        # QThread.finished fires once run() has fully returned -- safe
        # point to drop our reference and restore normal controls.
        self.runner = None
        self._set_run_actions_enabled(True)
        self._set_stop_enabled(False)
        self._set_queue_editable(True)
        self._update_status_bar()

        self._progress_timer.stop()
        self.progress_bar.setVisible(False)
        self.eta_label.setVisible(False)
        self._run_total_by_job = {}
        self._run_done_by_job = {}
        self._run_start_time = None
        self._run_last_progress_time = None

    def _set_run_actions_enabled(self, editable: bool):
        self.run_btn.setEnabled(editable)
        for action in getattr(self, "_run_actions", []):
            action.setEnabled(editable)

    def _set_queue_editable(self, editable: bool):
        """Disable queue-mutating controls while a batch run is in
        flight, so a job can't be removed/edited/re-set-up out from under
        the runner mid-pass."""
        self.add_btn.setEnabled(editable)
        self.edit_btn.setEnabled(editable)
        self.remove_btn.setEnabled(editable)
        self.setup_btn.setEnabled(editable)
        for action in getattr(self, "_reset_actions", []):
            action.setEnabled(editable)

    def closeEvent(self, event):
        if self.runner is not None:
            reply = QMessageBox.question(
                self, "Batch run in progress",
                "A batch run is still in progress. Quit anyway? The current video will be "
                "interrupted and that job will be left incomplete.",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self.runner.request_stop()
            self.runner.wait(5000)
        event.accept()

    def _update_status_bar(self):
        n = len(self.job_queue.jobs)
        ready = len(self.job_queue.runnable_jobs())
        self.statusBar().showMessage(f"{n} job(s) queued, {ready} ready to run")
