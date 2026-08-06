"""
Filters for cleaning up SLEAP-inference artifacts before scoring.

Handles one specific artifact: a single-frame "nose/tail swap", where the
nose prediction jumps to roughly where tail_base/torso actually is for one
isolated frame, while neck and ear nodes stay correctly placed. Only
isolated frames (bounded by clean frames on both sides) are touched --
sustained bad runs are a different, more serious tracking failure and are
left alone.

Two independent cues flag a candidate swap frame:
  (a) nose-neck distance exceeds NOSE_SWAP_BODY_LENGTH_RATIO times that
      frame's own neck-to-tail_base distance (a magnitude test, not an
      ordering test, so a real curled-up/grooming posture doesn't
      false-positive).
  (b) nose-neck distance exceeds NOSE_SWAP_DISTANCE_RATIO times the
      video's own median nose-neck distance -- the fallback when
      tail_base isn't confidently tracked.

Both cues also require neck and at least one ear node to be confidently
tracked on that frame.
"""

import numpy as np

import config
from pose_utils import Track


def _distance(a, b):
    return np.linalg.norm(a - b, axis=1)


def filter_nose_tail_swaps(track, cfg=config, verbose=False):
    """Return (new_track, n_frames_filtered).

    new_track is a copy of `track` with the nose keypoint nulled out on
    frames identified as isolated nose/tail-base swaps; everything else
    is untouched. A nulled frame is transparently excluded from scoring
    via scoring.py's existing `valid` mask. Only runs up to
    cfg.NOSE_SWAP_ISOLATED_MAX_RUN_FRAMES long, bounded by non-flagged
    frames, are nulled -- longer runs are left alone.
    """
    n = track.n_frames
    nose = track.xy(cfg.NODE_NOSE)
    neck = track.xy(cfg.NODE_NECK)
    left_ear = track.xy(cfg.NODE_LEFT_EAR)
    right_ear = track.xy(cfg.NODE_RIGHT_EAR)

    # Degrade gracefully to the video-median fallback cue if this
    # skeleton doesn't have a tail_base node under that exact name.
    has_tail_node = cfg.NODE_TAIL_BASE in track.node_names
    if has_tail_node:
        tail_base = track.xy(cfg.NODE_TAIL_BASE)
        tail_conf = track.conf(cfg.NODE_TAIL_BASE)
    else:
        tail_base = np.full((n, 2), np.nan)
        tail_conf = np.full(n, np.nan)

    neck_conf = track.conf(cfg.NODE_NECK)
    left_ear_conf = track.conf(cfg.NODE_LEFT_EAR)
    right_ear_conf = track.conf(cfg.NODE_RIGHT_EAR)

    min_conf = cfg.MIN_NODE_CONFIDENCE

    nose_present = ~np.isnan(nose).any(axis=1)
    neck_present = ~np.isnan(neck).any(axis=1)
    tail_present = has_tail_node & ~np.isnan(tail_base).any(axis=1)

    neck_ok = neck_present & (np.nan_to_num(neck_conf, nan=0.0) >= min_conf)
    ear_ok = (
        (~np.isnan(left_ear).any(axis=1) & (np.nan_to_num(left_ear_conf, nan=0.0) >= min_conf))
        | (~np.isnan(right_ear).any(axis=1) & (np.nan_to_num(right_ear_conf, nan=0.0) >= min_conf))
    )
    tail_ok = tail_present & (np.nan_to_num(tail_conf, nan=0.0) >= min_conf)

    other_head_nodes_trustworthy = neck_ok & ear_ok

    valid_neck_pair = nose_present & neck_present
    d_nose_neck = np.full(n, np.nan)
    d_nose_neck[valid_neck_pair] = _distance(nose[valid_neck_pair], neck[valid_neck_pair])

    valid_neck_tail_pair = neck_present & tail_present
    d_neck_tail = np.full(n, np.nan)
    d_neck_tail[valid_neck_tail_pair] = _distance(neck[valid_neck_tail_pair], tail_base[valid_neck_tail_pair])

    # (a) nose-neck distance is a large fraction of this frame's own body length.
    near_body_length = (
        other_head_nodes_trustworthy
        & tail_ok
        & valid_neck_pair
        & valid_neck_tail_pair
        & (d_neck_tail > 0)
        & (d_nose_neck > cfg.NOSE_SWAP_BODY_LENGTH_RATIO * d_neck_tail)
    )

    # (b) inordinately large nose-neck distance vs. this video's typical
    # value, computed from non-flagged frames so outliers can't skew it.
    reference_mask = other_head_nodes_trustworthy & valid_neck_pair & ~near_body_length
    reference_dist = (
        float(np.median(d_nose_neck[reference_mask])) if reference_mask.sum() >= 10 else None
    )

    if reference_dist and reference_dist > 0:
        outlier = (
            other_head_nodes_trustworthy
            & valid_neck_pair
            & (d_nose_neck > cfg.NOSE_SWAP_DISTANCE_RATIO * reference_dist)
        )
    else:
        outlier = np.zeros(n, dtype=bool)

    flagged = near_body_length | outlier

    # Only null out ISOLATED runs -- short spikes bounded by clean frames.
    max_run = cfg.NOSE_SWAP_ISOLATED_MAX_RUN_FRAMES
    to_null = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if flagged[i]:
            j = i
            while j < n and flagged[j]:
                j += 1
            if (j - i) <= max_run:
                to_null[i:j] = True
            i = j
        else:
            i += 1

    n_filtered = int(to_null.sum())
    if n_filtered and verbose:
        frames = np.nonzero(to_null)[0].tolist()
        print(f"Nose/tail-swap filter: nulled {n_filtered} isolated frame(s): {frames}")

    new_points = track.points.copy()
    new_scores = track.scores.copy()
    nose_idx = track.node_index(cfg.NODE_NOSE)
    new_points[to_null, nose_idx, :] = np.nan
    new_scores[to_null, nose_idx] = np.nan

    new_track = Track(fps=track.fps, node_names=track.node_names, points=new_points, scores=new_scores)
    return new_track, n_filtered
