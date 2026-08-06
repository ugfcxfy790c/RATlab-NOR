"""
Add-job dialog: native OS folder pickers for the input video folder and
output destination, a live "N videos found" readout, and validation
against filesystem-unsafe group names / duplicate group names already in
the queue.

Object-coordinate setup is deliberately NOT triggered from here -- per the
design decision (see job_queue.py docstring), a newly added job lands in
NEEDS_SETUP and setup happens as its own explicit step (task #4), so you
can queue several jobs first and then walk through setup for all of them.
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QLabel, QDialogButtonBox, QMessageBox, QCheckBox,
    QFileDialog, QWidget,
)

from job_queue import find_videos

# Characters that can't safely appear in a folder name on macOS or Windows.
_UNSAFE_NAME_CHARS = re.compile(r'[\\/:*?"<>|]')


def _folder_row(initial_text: str = "") -> tuple[QWidget, QLineEdit, QPushButton]:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    edit = QLineEdit(initial_text)
    edit.setReadOnly(True)
    browse_btn = QPushButton("Browse…")
    layout.addWidget(edit)
    layout.addWidget(browse_btn)
    return row, edit, browse_btn


class AddJobDialog(QDialog):
    def __init__(
        self, default_output_base: str, existing_group_names: set[str], parent=None, existing_job=None,
        default_skip_validation: bool = False,
    ):
        """If `existing_job` is given, the dialog opens pre-filled for
        editing that job (title/button text change accordingly) instead of
        building a new one. `existing_group_names` should exclude the job
        being edited's own current name, so it isn't flagged as a
        duplicate of itself. `default_skip_validation` (from the Settings
        dialog's "Paths & Output Defaults") only applies when adding a new
        job -- an existing job's own saved value always wins."""
        super().__init__(parent)
        self.existing_job = existing_job
        self.setWindowTitle("Edit Job" if existing_job else "Add Job")
        self.setMinimumWidth(480)

        self._existing_group_names = {n.lower() for n in existing_group_names}
        self.result_job_kwargs: dict | None = None

        self.group_name_edit = QLineEdit()
        self.group_name_edit.setPlaceholderText("e.g. Cohort 1 (saline)")

        initial_input = existing_job.input_folder if existing_job else ""
        initial_output = existing_job.output_base_folder if existing_job else default_output_base
        input_row, self.input_folder_edit, input_browse_btn = _folder_row(initial_input)
        output_row, self.output_base_edit, output_browse_btn = _folder_row(initial_output)

        self.video_count_label = QLabel("Pick an input folder to see how many videos it contains.")
        self.video_count_label.setStyleSheet("color: #666;")

        self.skip_validation_check = QCheckBox("Skip validation videos (faster; you can still spot-check later)")

        if existing_job:
            self.group_name_edit.setText(existing_job.group_name)
            self.skip_validation_check.setChecked(existing_job.skip_validation)
            if initial_input:
                self._update_video_count_label(initial_input)
        else:
            self.skip_validation_check.setChecked(default_skip_validation)

        form = QFormLayout()
        form.addRow("Group name:", self.group_name_edit)
        form.addRow("Input video folder:", input_row)
        form.addRow("", self.video_count_label)
        form.addRow("Output base folder:", output_row)
        out_hint = QLabel("Results will be written to <output base>/<group name>/")
        out_hint.setStyleSheet("color: #666; font-size: 11px;")
        form.addRow("", out_hint)
        form.addRow("", self.skip_validation_check)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Save Changes" if existing_job else "Add Job")
        self.buttons.accepted.connect(self._on_accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.buttons)

        input_browse_btn.clicked.connect(self._browse_input_folder)
        output_browse_btn.clicked.connect(self._browse_output_folder)

    # -- browsing --

    def _update_video_count_label(self, folder: str):
        videos = find_videos(folder)
        if not videos:
            self.video_count_label.setText("No .mp4 files found in this folder (it's searched recursively).")
            self.video_count_label.setStyleSheet("color: #a94442;")
        else:
            self.video_count_label.setText(f"{len(videos)} video(s) found.")
            self.video_count_label.setStyleSheet("color: #2e7d32;")

    def _browse_input_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose the folder of videos for this group", self.input_folder_edit.text() or str(Path.home()),
        )
        if not folder:
            return
        self.input_folder_edit.setText(folder)
        self._update_video_count_label(folder)

    def _browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose where this group's results should be written",
            self.output_base_edit.text() or str(Path.home()),
        )
        if folder:
            self.output_base_edit.setText(folder)

    # -- validation + accept --

    def _on_accept(self):
        group_name = self.group_name_edit.text().strip()
        input_folder = self.input_folder_edit.text().strip()
        output_base = self.output_base_edit.text().strip()

        if not group_name or not input_folder or not output_base:
            QMessageBox.warning(self, "Missing info", "Group name, input folder, and output folder are all required.")
            return

        if _UNSAFE_NAME_CHARS.search(group_name):
            QMessageBox.warning(
                self, "Invalid group name",
                'Group name can\'t contain any of: \\ / : * ? " < > |\n'
                "(it's used directly as an output folder name).",
            )
            return

        if group_name.lower() in self._existing_group_names:
            QMessageBox.warning(self, "Duplicate group", f"A job named '{group_name}' is already in the queue.")
            return

        if not Path(input_folder).is_dir():
            QMessageBox.warning(self, "Folder not found", f"Input folder doesn't exist:\n{input_folder}")
            return

        if not find_videos(input_folder):
            reply = QMessageBox.question(
                self, "No videos found",
                f"No .mp4 files were found under:\n{input_folder}\n\nAdd this job anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self.result_job_kwargs = dict(
            group_name=group_name,
            input_folder=input_folder,
            output_base_folder=output_base,
            skip_validation=self.skip_validation_check.isChecked(),
        )
        self.accept()
