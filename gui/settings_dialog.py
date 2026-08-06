"""
Settings dialog -- opened via the "Settings" menu's "Preferences…" action.
Two sections:

  - "Paths & Output Defaults": low-risk, only affects what a newly-added
    job is pre-filled with (default output folder, skip-validation
    checkbox). Doesn't touch jobs already in the queue.
  - "Scoring / Calibration Constants": affects scoring for every future
    run across every job (see app_settings.SCORING_OVERRIDE_SPEC and
    job_config.build_job_cfg). Per established practice these should be
    measured from footage, not eyeballed -- the dialog says so.

Each scoring field is pre-filled from the saved override in
app_data/settings.json if one exists, else config.py's hardcoded default.
Saving only persists a value that actually differs from config.py's
default (see app_settings.get_scoring_overrides' docstring for why), so
untouched fields keep tracking config.py even if config.py itself
changes later.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QCheckBox, QDoubleSpinBox, QSpinBox,
    QDialogButtonBox, QFileDialog, QWidget,
)

import app_settings


class SettingsDialog(QDialog):
    def __init__(self, config_module, parent=None):
        super().__init__(parent)
        self.config_module = config_module
        self.nor_classifier_dir = config_module.NOR_CLASSIFIER_DIR
        self.setWindowTitle("Settings")
        self.setMinimumWidth(560)

        self._scoring_fields: dict[str, QDoubleSpinBox | QSpinBox | QCheckBox] = {}

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_paths_group())
        layout.addWidget(self._build_scoring_group())

        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._on_save)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    # -- paths & output defaults --

    def _build_paths_group(self) -> QGroupBox:
        group = QGroupBox("Paths && Output Defaults")
        form = QFormLayout(group)

        initial_output = app_settings.get_default_output_base(self.nor_classifier_dir, self.config_module)
        self.output_edit = QLineEdit(initial_output)
        self.output_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_output)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(self.output_edit)
        row_layout.addWidget(browse_btn)
        form.addRow("Default output folder for new jobs:", row)

        self.skip_validation_check = QCheckBox("New jobs default to skipping validation videos")
        self.skip_validation_check.setChecked(app_settings.get_skip_validation_default(self.nor_classifier_dir))
        form.addRow("", self.skip_validation_check)

        return group

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose the default output folder", self.output_edit.text() or str(Path.home()),
        )
        if folder:
            self.output_edit.setText(folder)

    # -- scoring/calibration constants --

    def _build_scoring_group(self) -> QGroupBox:
        group = QGroupBox("Scoring / Calibration Constants")
        outer = QVBoxLayout(group)

        warning = QLabel(
            "These affect scoring for every future run, across every job. "
            "Measure calibration constants (like pixels-per-cm) from footage "
            "-- don't eyeball them -- and check the literature before "
            "re-tuning a criterion like the sniff angle."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #a94442;")
        outer.addWidget(warning)

        overrides = app_settings.get_scoring_overrides(self.nor_classifier_dir)
        form = QFormLayout()

        for name, value_type, decimals, label, help_text in app_settings.SCORING_OVERRIDE_SPEC:
            default_value = getattr(self.config_module, name)
            current_value = overrides.get(name, default_value)

            if value_type is bool:
                field = QCheckBox()
                field.setChecked(bool(current_value))
            elif value_type is int:
                field = QSpinBox()
                field.setRange(0, 1_000_000)
                field.setValue(int(current_value))
            else:
                field = QDoubleSpinBox()
                field.setDecimals(decimals)
                field.setRange(-1_000_000.0, 1_000_000.0)
                field.setSingleStep(10 ** (-decimals) if decimals else 1)
                field.setValue(float(current_value))

            field.setToolTip(f"{help_text}\n\nconfig.py default: {default_value}")
            self._scoring_fields[name] = field
            if value_type is bool:
                form.addRow("", field)
                field.setText(label)
            else:
                form.addRow(f"{label}:", field)

        outer.addLayout(form)

        reset_btn = QPushButton("Reset All Scoring Constants to config.py Defaults")
        reset_btn.clicked.connect(self._reset_scoring_defaults)
        outer.addWidget(reset_btn)

        return group

    def _reset_scoring_defaults(self):
        for name, field in self._scoring_fields.items():
            default_value = getattr(self.config_module, name)
            if isinstance(field, QCheckBox):
                field.setChecked(bool(default_value))
            else:
                field.setValue(default_value)

    # -- save --

    def _on_save(self):
        app_settings.set_default_output_base(self.nor_classifier_dir, self.output_edit.text().strip())
        app_settings.set_skip_validation_default(self.nor_classifier_dir, self.skip_validation_check.isChecked())

        overrides = {}
        for name, value_type, decimals, label, help_text in app_settings.SCORING_OVERRIDE_SPEC:
            default_value = getattr(self.config_module, name)
            field = self._scoring_fields[name]

            if value_type is bool:
                field_value = field.isChecked()
                differs = field_value != bool(default_value)
            elif value_type is int:
                field_value = int(field.value())
                differs = field_value != default_value
            else:
                field_value = round(float(field.value()), decimals)
                differs = abs(field_value - float(default_value)) > 1e-9

            if differs:
                overrides[name] = field_value

        app_settings.set_scoring_overrides(self.nor_classifier_dir, overrides)
        self.accept()
