"""
Persisted app-level settings -- which model (a subfolder of
RATlab/models/) new jobs should use for inference, the default
output/validation choices new jobs are pre-filled with, and any
scoring/calibration constants overridden via the Settings dialog. Stored
at app_data/settings.json, separate from queue.json since it's app-wide
rather than per-job.

Precedence (highest first), applied in job_config.build_job_cfg():
  1. a job's own config_overrides (explicit per-job override)
  2. app-wide settings from here (model selection, scoring overrides --
     applies to every job that doesn't override them itself)
  3. config.py's hardcoded defaults
"""

from __future__ import annotations

import json
from pathlib import Path

from job_queue import default_app_data_dir

# Scoring/calibration constants exposed in the Settings dialog, with enough
# metadata to build a field for each (a numeric spinbox for float/int,
# a checkbox for bool -- see settings_dialog.py's _build_scoring_group)
# and to explain what it does -- see config.py for the authoritative
# descriptions these are summarized from. Deliberately a short, curated
# list rather than "every uppercase name in config.py": these are the
# constants that directly shape scoring output, as opposed to paths,
# node names, or the filename-parsing regex.
#
# CLIMBING_TORSO_DISTANCE_PX is intentionally NOT in this list -- it's
# derived from OBJECT_SIZE_CM and PX_PER_CM in config.py, and
# job_config.build_job_cfg() recomputes it from whichever of those two
# ends up in effect, so overriding it separately here would risk the two
# silently drifting apart.
SCORING_OVERRIDE_SPEC = [
    ("PX_PER_CM", float, 2, "Pixels per real-world cm",
     "Measure from footage (e.g. a known object's pixel width / its cm width) -- don't eyeball."),
    ("OBJECT_SIZE_CM", float, 2, "Object footprint size (cm)",
     "Real-world size of the objects being explored."),
    ("OBJECT_PADDING_CM", float, 2, "Object hitbox padding (cm)",
     "Padding added to the object footprint to form the nose-proximity hitbox."),
    ("OBJECT_FOOTPRINT_GROW_PX", float, 1, "Object footprint grow (px)",
     "Grows the footprint square used for the cone-intersection test and picker outline."),
    ("SNIFF_CONE_HALF_ANGLE_DEG", float, 1, "Sniff cone half-angle (deg)",
     "How far off dead-on the head direction can be and still count as \"oriented toward\" the object."),
    ("SNIFF_RAY_ORIGIN_BACKSET_RATIO", float, 2, "Sniff ray origin backset (ratio)",
     "Pulls the cone's apex back from the nose tip, as a fraction of neck-to-nose distance."),
    ("MIN_NODE_CONFIDENCE", float, 2, "Min keypoint confidence",
     "SLEAP confidence below this is treated as an untracked keypoint."),
    ("NOSE_SWAP_BODY_LENGTH_RATIO", float, 2, "Nose swap: body-length ratio",
     "Flags a frame if nose-neck distance exceeds this fraction of neck-to-tail-base distance."),
    ("NOSE_SWAP_DISTANCE_RATIO", float, 2, "Nose swap: fallback distance ratio",
     "Fallback cue when tail_base isn't confidently tracked (multiple of the video's median nose-neck distance)."),
    ("NOSE_SWAP_ISOLATED_MAX_RUN_FRAMES", int, 0, "Nose swap: max run length (frames)",
     "Only isolated runs up to this many consecutive frames get nulled out."),
    ("MERGE_GAP_S", float, 2, "Bout merge gap (s)",
     "Bouts closer together than this are merged into one -- for a confidently-tracked gap "
     "(real evidence of a break). See confidence-aware gap merging below for low-confidence gaps."),
    ("MIN_BOUT_DURATION_S", float, 2, "Min bout duration (s)",
     "Bouts shorter than this are discarded as noise."),
    ("CONFIDENCE_AWARE_MERGE_ENABLED", bool, None, "Confidence-aware gap merging",
     "Merge low-confidence-tracking gaps between bouts even if longer than the merge gap above, "
     "since there's no real evidence the animal left (usually self-occlusion from close sniffing). "
     "Disable to fall back to one flat threshold (the merge gap above) for every gap."),
    ("LOW_CONFIDENCE_GAP_MERGE_S", float, 2, "Low-confidence gap merge cap (s)",
     "How long a low-confidence-tracking gap can be and still get merged automatically. Longer "
     "dropouts are left as separate bouts and the video is flagged for review instead."),
]


def settings_path(nor_classifier_dir) -> Path:
    return default_app_data_dir(nor_classifier_dir) / "settings.json"


def load_settings(nor_classifier_dir) -> dict:
    path = settings_path(nor_classifier_dir)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_settings(nor_classifier_dir, settings: dict) -> None:
    path = settings_path(nor_classifier_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2))


def list_available_models(ratlab_dir) -> list[str]:
    """Names of model subfolders under RATlab/models/ -- each one is
    something main.py/config.py's MODEL_PATHS could point at. Skips
    hidden entries (e.g. .DS_Store) and anything that isn't a
    directory."""
    models_dir = Path(ratlab_dir) / "models"
    if not models_dir.is_dir():
        return []
    return sorted(
        p.name for p in models_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def get_selected_model(nor_classifier_dir) -> str | None:
    return load_settings(nor_classifier_dir).get("selected_model")


def set_selected_model(nor_classifier_dir, model_name: str) -> None:
    settings = load_settings(nor_classifier_dir)
    settings["selected_model"] = model_name
    save_settings(nor_classifier_dir, settings)


# --- Add-job defaults (Settings dialog "Paths & Output Defaults") --------

def get_default_output_base(nor_classifier_dir, config_module) -> str:
    """Pre-fills the Add Job dialog's output folder. Falls back to
    config.py's OUTPUT_FOLDER until the user sets one via Settings."""
    saved = load_settings(nor_classifier_dir).get("default_output_base")
    return saved if saved else str(config_module.OUTPUT_FOLDER)


def set_default_output_base(nor_classifier_dir, folder: str) -> None:
    settings = load_settings(nor_classifier_dir)
    settings["default_output_base"] = folder
    save_settings(nor_classifier_dir, settings)


def get_skip_validation_default(nor_classifier_dir) -> bool:
    """Whether a newly-added job's "skip validation videos" checkbox
    starts checked. Doesn't affect jobs already in the queue."""
    return bool(load_settings(nor_classifier_dir).get("skip_validation_default", False))


def set_skip_validation_default(nor_classifier_dir, value: bool) -> None:
    settings = load_settings(nor_classifier_dir)
    settings["skip_validation_default"] = bool(value)
    save_settings(nor_classifier_dir, settings)


# --- Scoring/calibration overrides (Settings dialog "Scoring / Calibration") --

def get_scoring_overrides(nor_classifier_dir) -> dict:
    """Only ever contains keys from SCORING_OVERRIDE_SPEC that the user has
    explicitly changed away from config.py's default (see
    set_scoring_overrides) -- so a value not present here just means "keep
    tracking whatever config.py says," including future edits to config.py
    made outside the GUI."""
    raw = load_settings(nor_classifier_dir).get("scoring_overrides", {})
    known = {name for name, *_ in SCORING_OVERRIDE_SPEC}
    return {k: v for k, v in raw.items() if k in known}


def set_scoring_overrides(nor_classifier_dir, overrides: dict) -> None:
    settings = load_settings(nor_classifier_dir)
    settings["scoring_overrides"] = overrides
    save_settings(nor_classifier_dir, settings)
