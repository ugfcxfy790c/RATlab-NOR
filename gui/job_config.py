"""
Builds a per-job config object for the pipeline modules (object_picker,
sleap_inference, scoring, etc.) that all expect a `cfg`-like object with
config.py's attribute names.

Rather than mutating the shared config module (which would race across
jobs / GUI state), each job gets its own lightweight copy with
VIDEO_FOLDER pointed at that job's input folder, the app-wide selected
model and scoring overrides (if any -- see app_settings.py) applied, and
any of the job's own config_overrides layered on top of that.
"""

from __future__ import annotations

import types
from pathlib import Path

from app_settings import get_scoring_overrides, get_selected_model


def build_job_cfg(job, base_config_module):
    """Return a namespace object usable anywhere `cfg` is expected,
    reflecting `job`'s input folder and per-job overrides on top of the
    installed config.py defaults.

    Precedence (highest first) for both MODEL_PATHS and the scoring
    constants in app_settings.SCORING_OVERRIDE_SPEC: job.config_overrides
    -> app-wide settings from the GUI's Model/Settings menus ->
    config.py's hardcoded defaults.
    """
    cfg = types.SimpleNamespace()
    for name in dir(base_config_module):
        if name.isupper():
            setattr(cfg, name, getattr(base_config_module, name))

    cfg.VIDEO_FOLDER = Path(job.input_folder)

    selected_model = get_selected_model(base_config_module.NOR_CLASSIFIER_DIR)
    if selected_model:
        model_path = Path(base_config_module.RATLAB_DIR) / "models" / selected_model
        if model_path.is_dir():
            cfg.MODEL_PATHS = [model_path]

    scoring_overrides = get_scoring_overrides(base_config_module.NOR_CLASSIFIER_DIR)
    for key, value in scoring_overrides.items():
        setattr(cfg, key, value)

    # CLIMBING_TORSO_DISTANCE_PX is derived from OBJECT_SIZE_CM/PX_PER_CM
    # in config.py rather than being independently settable (see
    # app_settings.SCORING_OVERRIDE_SPEC's docstring) -- recompute it here
    # so it stays consistent with whichever of those two ends up in effect.
    if "OBJECT_SIZE_CM" in scoring_overrides or "PX_PER_CM" in scoring_overrides:
        cfg.CLIMBING_TORSO_DISTANCE_PX = (cfg.OBJECT_SIZE_CM / 2.0) * cfg.PX_PER_CM

    for key, value in (job.config_overrides or {}).items():
        setattr(cfg, key, value)

    return cfg
