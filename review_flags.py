"""
Heuristic flags for "this video's automated result might be wrong, take a
look" -- computed once per video at the end of the batch pipeline (see
gui/batch_worker_process.py and main.py's cmd_run), persisted alongside
that job's other output (<output_folder>/review_flags.json), and
surfaced by the GUI's post-batch review screen
(gui/batch_review_dialog.py) so a person can skim straight to the videos
most likely to need a manual re-score instead of opening every
validation video.

Three independent heuristics:
  - low overall tracking confidence -- few frames had trustworthy
    nose+neck tracking to begin with (via scoring.diagnose(), which
    requires *both* points confident).
  - the nose's *own* tracking confidence drops out for a sustained,
    continuous stretch while it's sitting right in an object's hitbox --
    long enough that it could plausibly be masking a real (or a false)
    exploration bout.
  - an unusual number of bouts got automatically merged across a
    low-confidence gap (scoring.py's confidence-aware merging -- see
    labels_to_bouts()/_merge_bout_runs()). Each individual merge is the
    scorer's best guess given no real evidence the animal left, but a
    video with an unusually high count of them is worth a person's
    spot-check rather than trusting every one blindly.

The second one deliberately does NOT reuse scoring.diagnose()'s
"suppressed" count: that figure is keyed off the same joint nose+neck
"valid" flag as the first heuristic, so a frame where the nose itself is
tracked perfectly fine but the neck alone dips in confidence still counts
as "suppressed". That's the *normal* case during real close-up sniffing
(head angled down into the object, neck partly self-occluded), not a
sign of bad data -- counting it made nearly every video with genuine
exploration trip this flag. Checking the nose's own confidence, and
requiring one continuous dropout rather than a handful of scattered
single-frame blips summed across the whole video, targets the actually
suspicious case: the tracker specifically losing the nose right where
scoring needs it most.

This intentionally doesn't try to guess at bout-level correctness --
just "was the input this video's score is built on trustworthy enough to
take at face value".
"""

import json
from pathlib import Path

import numpy as np

import config as _default_config
import scoring
from geometry import hitbox_half_width_px

# Fallback defaults if `cfg` doesn't carry these (e.g. an older
# config.py) -- normally overridden by config.py's
# REVIEW_MIN_VALID_FRAMES_PCT / REVIEW_NOSE_DROPOUT_NEAR_OBJECT_S, which
# job_config.build_job_cfg copies through like every other config.py
# constant, so per-job GUI overrides work the same way as the scoring
# thresholds do.
MIN_VALID_FRAMES_PCT = getattr(_default_config, "REVIEW_MIN_VALID_FRAMES_PCT", 70.0)
NOSE_DROPOUT_NEAR_OBJECT_S = getattr(_default_config, "REVIEW_NOSE_DROPOUT_NEAR_OBJECT_S", 1.0)
MAX_AUTO_MERGES = getattr(_default_config, "REVIEW_MAX_AUTO_MERGES", 2)


def _mask_to_runs(mask):
    """1-D boolean array -> list of (start_frame, stop_frame_exclusive)
    contiguous True runs -- same (start, stop_exclusive) convention as
    scoring.py's own _mask_to_runs, so frame ranges from this module
    compose cleanly with the rest of the pipeline."""
    if not np.any(mask):
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0]
    return list(zip(starts.tolist(), stops.tolist()))


def _longest_run_frames(mask):
    """Length, in frames, of the longest contiguous stretch of True in a
    1-D boolean array (0 if none)."""
    runs = _mask_to_runs(mask)
    return max((stop - start for start, stop in runs), default=0)


def nose_confidence_dropout_runs(track, cfg):
    """List of (start_frame, stop_frame_exclusive) ranges where the
    nose's own tracking confidence was below cfg.MIN_NODE_CONFIDENCE
    (including frames where it's missing outright) -- independent of
    proximity to any object, unlike compute_video_flags's near-object
    check below. Used to mark confidence drops on the GUI's scrub bar
    (video_player_widget.py's set_markers, via
    batch_review_dialog.py's FlaggedVideoViewer) so a reviewer can jump
    straight to a suspicious stretch instead of scrubbing blind."""
    nose_conf = track.conf(cfg.NODE_NOSE)
    low = np.nan_to_num(nose_conf, nan=0.0) < cfg.MIN_NODE_CONFIDENCE
    return _mask_to_runs(low)


def compute_video_flags(track, object_coords, cfg, merge_events=None):
    """Return a list of human-readable reason strings -- empty if
    nothing about this video's tracking data looks suspicious.

    merge_events: the merge_events_by_label dict scoring.labels_to_bouts()
    returns alongside its bouts (i.e. {label_value: [(gap_start_frame,
    gap_stop_frame_exclusive), ...]}) -- optional, since not every caller
    already has bout data computed nearby. When omitted, the auto-merge
    count check below is simply skipped rather than recomputed here, to
    avoid this function silently doing its own (possibly diverging) pass
    of score_frame_labels()/labels_to_bouts()."""
    reasons = []
    fps = track.fps or 30.0

    info = scoring.diagnose(track, object_coords, cfg=cfg)
    min_valid_pct = getattr(cfg, "REVIEW_MIN_VALID_FRAMES_PCT", MIN_VALID_FRAMES_PCT)
    pct_valid = info["pct_valid"]
    if pct_valid < min_valid_pct:
        reasons.append(
            f"Low tracking confidence -- only {pct_valid:.0f}% of frames had "
            f"trustworthy nose/neck tracking."
        )

    dropout_s_threshold = getattr(cfg, "REVIEW_NOSE_DROPOUT_NEAR_OBJECT_S", NOSE_DROPOUT_NEAR_OBJECT_S)
    half_width = hitbox_half_width_px(cfg)
    nose = track.xy(cfg.NODE_NOSE)
    nose_conf = track.conf(cfg.NODE_NOSE)
    nose_present = ~np.isnan(nose).any(axis=1)
    nose_low_conf = nose_present & (np.nan_to_num(nose_conf, nan=0.0) < cfg.MIN_NODE_CONFIDENCE)

    for obj_name, obj_xy in object_coords.items():
        obj_xy = np.asarray(obj_xy, dtype=np.float64)
        to_obj = obj_xy[None, :] - nose
        in_hitbox = nose_present & (np.abs(to_obj[:, 0]) <= half_width) & (np.abs(to_obj[:, 1]) <= half_width)
        suppressed = in_hitbox & nose_low_conf

        longest_s = _longest_run_frames(suppressed) / fps
        if longest_s >= dropout_s_threshold:
            reasons.append(
                f"Nose tracking confidence dropped out for a continuous {longest_s:.1f}s "
                f"while in '{obj_name}''s hitbox -- possible missed or miscounted exploration."
            )

    if merge_events:
        max_merges = getattr(cfg, "REVIEW_MAX_AUTO_MERGES", MAX_AUTO_MERGES)
        total_merges = sum(len(events) for events in merge_events.values())
        if total_merges > max_merges:
            reasons.append(
                f"{total_merges} bout(s) were automatically merged across a low-confidence "
                f"tracking gap -- review recommended."
            )

    return reasons


def _flags_path(output_folder):
    return Path(output_folder) / "review_flags.json"


def load_flags(output_folder):
    path = _flags_path(output_folder)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_flags(output_folder, flags):
    path = _flags_path(output_folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(flags, f, indent=2)
    tmp.replace(path)


def update_video_flags(output_folder, video_stem, reasons):
    """Store this video's freshly computed flag reasons -- called once
    per video, once per batch run. Preserves an existing "reviewed"
    acknowledgement only if the reasons are exactly unchanged from last
    time; if the situation actually changed (different reasons, or newly
    flagged), the video needs a fresh look, so any previous
    acknowledgement is cleared rather than silently carried forward."""
    flags = load_flags(output_folder)
    existing = flags.get(video_stem)
    reviewed = bool(existing and existing.get("reasons") == reasons and existing.get("reviewed"))
    flags[video_stem] = {"reasons": reasons, "reviewed": reviewed}
    save_flags(output_folder, flags)
    return flags[video_stem]


def mark_reviewed(output_folder, video_stem):
    flags = load_flags(output_folder)
    if video_stem in flags:
        flags[video_stem]["reviewed"] = True
        save_flags(output_folder, flags)


def active_flags(output_folder):
    """{video_stem: entry} for every video in this output folder that's
    currently flagged (has reasons) and hasn't been marked reviewed."""
    flags = load_flags(output_folder)
    return {
        stem: entry for stem, entry in flags.items()
        if entry.get("reasons") and not entry.get("reviewed")
    }
