"""
Core NOR exploration-bout scoring logic.

Standard criteria used here (see e.g. Antunes & Biala, 2012, "The novel
object recognition memory: neurobiology, test procedure and its
modifications", for a review of common scoring conventions):

  - A frame counts as "exploring" an object if the animal's nose falls
    inside the object's square hitbox (the object's real-world footprint
    plus a padding margin -- see OBJECT_SIZE_CM / OBJECT_PADDING_CM /
    PX_PER_CM in config.py; this is the same hitbox shown and dragged in
    the object picker) AND the head is oriented toward it. "Oriented
    toward it" is a geometric test: a ray from the nose in the head
    direction (neck -> nose), or any ray within +/-SNIFF_CONE_HALF_ANGLE_DEG
    of it, must cross the object's real footprint square. See
    _head_ray_intersects_object() and _head_cone_intersects_object().
  - Frames where the torso is essentially on top of the object (climbing /
    sitting on it) are excluded, since climbing is not scored as
    exploration under standard NOR conventions.
  - If both criteria are met for both objects in the same frame (rare,
    only possible if objects are very close together), the nearer object
    wins.
  - Short gaps between consecutive exploring frames are bridged (tracking
    jitter) and very short bouts are dropped as noise. A gap that's
    mostly low-confidence/untracked (the scorer never actually saw the
    animal look away -- usually self-occlusion from close sniffing, not
    a real break) is bridged more liberally than a gap the tracker
    confidently saw as "not exploring" the whole time (real evidence of
    a break). See CONFIDENCE_AWARE_MERGE_ENABLED in config.py and
    _merge_bout_runs() below.

This module is independent of SLEAP/video I/O -- it only needs a `Track`
(see pose_utils.py) and the clicked object coordinates, so it can be
tested with synthetic data.
"""

from dataclasses import dataclass
import numpy as np

import config
from geometry import hitbox_half_width_px, object_half_width_px


@dataclass
class Bout:
    start_s: float
    stop_s: float

    @property
    def duration_s(self):
        return self.stop_s - self.start_s


def _head_ray_intersects_object(nose, head_vec, obj_xy, object_half_width):
    """Whether the forward ray from the nose, in the head direction
    (neck -> nose), crosses the object's square footprint anywhere.

    Uses the standard ray/AABB slab method: parametrize the ray as
    point(t) = nose + t * head_vec for t >= 0, compute the entry/exit t
    range on each axis where the ray is within the box's bounds, then
    intersect those ranges. If the nose is already inside the footprint,
    the ray trivially "hits" at t=0 regardless of direction.

    Returns a boolean array, one per frame.
    """
    lo = obj_xy - object_half_width
    hi = obj_xy + object_half_width

    n = nose.shape[0]
    tmin = np.full(n, -np.inf)
    tmax = np.full(n, np.inf)
    hit = np.ones(n, dtype=bool)

    for axis in range(2):
        o = nose[:, axis]
        d = head_vec[:, axis]
        lo_a, hi_a = lo[axis], hi[axis]

        nonzero = np.abs(d) > 1e-9
        d_safe = np.where(nonzero, d, 1.0)  # placeholder to avoid /0; masked out below
        t1 = (lo_a - o) / d_safe
        t2 = (hi_a - o) / d_safe
        axis_tmin = np.minimum(t1, t2)
        axis_tmax = np.maximum(t1, t2)
        tmin = np.where(nonzero, np.maximum(tmin, axis_tmin), tmin)
        tmax = np.where(nonzero, np.minimum(tmax, axis_tmax), tmax)

        # Ray parallel to this axis: it never crosses the lo/hi planes on
        # this axis, so the box only counts as hit if the nose already
        # sits within [lo_a, hi_a] on this axis -- otherwise no value of
        # t brings the ray into the box along this axis.
        parallel_outside = ~nonzero & ((o < lo_a) | (o > hi_a))
        hit = hit & ~parallel_outside

    intersects = hit & (tmax >= tmin) & (tmax >= 0)
    return intersects


def _wrap_to_pi(angle):
    """Wrap an angle (radians) to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _head_cone_intersects_object(nose, head_vec, obj_xy, object_half_width, cfg=config):
    """Widened version of _head_ray_intersects_object(): true if any ray
    within +/-cfg.SNIFF_CONE_HALF_ANGLE_DEG of the head vector, cast from a
    point pulled back from the nose tip, crosses the object's footprint.

    The cone's apex is set back from the tracked nose tip by
    cfg.SNIFF_RAY_ORIGIN_BACKSET_RATIO * |head_vec| along -head_vec, rather
    than the nose tip itself. A cone rooted at the tip can never reach an
    object touched by the SIDE of the snout, since the object then sits
    off to the side of (not roughly in front of) the tip regardless of
    cone width; pulling the apex back models the snout's own width, so the
    same angular cone sweeps a wider swath at the nose's actual position.
    Real tip contact (nose inside the footprint) still always counts,
    independent of this.

    The angular test itself is exact, not sampled: for a convex box seen
    from an external point, the directions that hit it form the interval
    between its two extreme (tangent) corners -- found here via all 4
    corners -- checked against the cone's [-half_angle, +half_angle]
    interval around the head vector.

    Returns a boolean array, one per frame.
    """
    half_angle_rad = np.radians(cfg.SNIFF_CONE_HALF_ANGLE_DEG)
    lo = obj_xy - object_half_width
    hi = obj_xy + object_half_width

    tip_inside = (
        (nose[:, 0] >= lo[0]) & (nose[:, 0] <= hi[0])
        & (nose[:, 1] >= lo[1]) & (nose[:, 1] <= hi[1])
    )

    origin = nose - cfg.SNIFF_RAY_ORIGIN_BACKSET_RATIO * head_vec

    head_angle = np.arctan2(head_vec[:, 1], head_vec[:, 0])
    n = nose.shape[0]
    min_delta = np.full(n, np.inf)
    max_delta = np.full(n, -np.inf)
    for cx, cy in ((lo[0], lo[1]), (hi[0], lo[1]), (lo[0], hi[1]), (hi[0], hi[1])):
        to_corner = np.stack([cx - origin[:, 0], cy - origin[:, 1]], axis=1)
        corner_angle = np.arctan2(to_corner[:, 1], to_corner[:, 0])
        delta = _wrap_to_pi(corner_angle - head_angle)
        min_delta = np.minimum(min_delta, delta)
        max_delta = np.maximum(max_delta, delta)

    overlap = (min_delta <= half_angle_rad) & (max_delta >= -half_angle_rad)
    return tip_inside | overlap


def _valid_mask(track, cfg=config):
    """Whether the nose AND neck are both present and confidently tracked,
    frame by frame -- the "trustworthy enough to score" test shared by
    score_frame_labels(), diagnose(), and compute_frame_details(), and
    also used by _merge_bout_runs() to tell a genuine "animal looked
    away" gap from a "tracker lost the nose" gap (see this module's
    docstring)."""
    nose = track.xy(cfg.NODE_NOSE)
    neck = track.xy(cfg.NODE_NECK)
    nose_conf = track.conf(cfg.NODE_NOSE)
    neck_conf = track.conf(cfg.NODE_NECK)
    return (
        ~np.isnan(nose).any(axis=1)
        & ~np.isnan(neck).any(axis=1)
        & (np.nan_to_num(nose_conf, nan=0.0) >= cfg.MIN_NODE_CONFIDENCE)
        & (np.nan_to_num(neck_conf, nan=0.0) >= cfg.MIN_NODE_CONFIDENCE)
    )


def score_frame_labels(track, object_coords, cfg=config):
    """Return an array of length n_frames with values in {0, 1, 2} meaning
    "not exploring", "exploring object 1", "exploring object 2".

    object_coords: dict like {"novel": (x, y), "original": (x, y)}
    """
    half_width = hitbox_half_width_px(cfg)
    obj_half_width = object_half_width_px(cfg)

    nose = track.xy(cfg.NODE_NOSE)
    neck = track.xy(cfg.NODE_NECK)
    torso = track.xy(cfg.NODE_TORSO)

    valid = _valid_mask(track, cfg)

    head_vec = nose - neck  # neck -> nose

    n_frames = track.n_frames
    labels = np.zeros(n_frames, dtype=np.int8)
    best_dist = np.full(n_frames, np.inf)

    for obj_i, (obj_name, obj_xy) in enumerate(object_coords.items(), start=1):
        obj_xy = np.asarray(obj_xy, dtype=np.float64)
        to_obj = obj_xy[None, :] - nose  # nose -> object center (hitbox/distance still use this)
        dist = np.linalg.norm(to_obj, axis=1)
        oriented = _head_cone_intersects_object(nose, head_vec, obj_xy, obj_half_width, cfg=cfg)

        in_hitbox = (np.abs(to_obj[:, 0]) <= half_width) & (np.abs(to_obj[:, 1]) <= half_width)

        meets_criteria = valid & in_hitbox & oriented

        if not np.all(np.isnan(torso)):
            torso_dist = np.linalg.norm(obj_xy[None, :] - torso, axis=1)
            climbing = torso_dist <= cfg.CLIMBING_TORSO_DISTANCE_PX
            meets_criteria = meets_criteria & ~climbing

        # nearer object wins if both match this frame
        take = meets_criteria & (dist < best_dist)
        labels[take] = obj_i
        best_dist[take] = dist[take]

    return labels


def diagnose(track, object_coords, cfg=config):
    """Break down why a track is or isn't producing exploration bouts --
    for debugging a video that scores zero bouts unexpectedly. Returns a
    dict of stats: how many frames have trustworthy nose/neck tracking,
    where the nose actually was, and per-object distance/angle/hitbox
    numbers so you can tell whether the problem is (a) bad/low-confidence
    tracking, (b) object coordinates that don't line up with where the
    nose actually goes, or (c) thresholds that are too strict.
    """
    half_width = hitbox_half_width_px(cfg)
    obj_half_width = object_half_width_px(cfg)

    nose = track.xy(cfg.NODE_NOSE)
    neck = track.xy(cfg.NODE_NECK)
    nose_conf = track.conf(cfg.NODE_NOSE)

    valid = _valid_mask(track, cfg)

    n_frames = track.n_frames
    n_valid = int(valid.sum())

    # Per-node breakdown -- "valid" above requires BOTH nose and neck to be
    # present AND confident, which can hide which node is actually the
    # bottleneck. This separates "the model never predicted this point at
    # all" (present=False) from "it predicted a point but wasn't confident
    # in it" (present but below MIN_NODE_CONFIDENCE), for every node in the
    # skeleton -- not just nose/neck -- so a low overall valid-frame count
    # can be traced back to a specific node.
    node_breakdown = {}
    for node_name in track.node_names:
        node_xy = track.xy(node_name)
        node_conf = track.conf(node_name)
        present = ~np.isnan(node_xy).any(axis=1)
        conf_vals = node_conf[present]
        confident = present & (np.nan_to_num(node_conf, nan=0.0) >= cfg.MIN_NODE_CONFIDENCE)
        node_breakdown[node_name] = {
            "pct_present": (100.0 * present.sum() / n_frames) if n_frames else 0.0,
            "pct_confident": (100.0 * confident.sum() / n_frames) if n_frames else 0.0,
            "mean_confidence": float(np.mean(conf_vals)) if conf_vals.size else None,
        }

    info = {
        "n_frames": n_frames,
        "n_valid_frames": n_valid,
        "pct_valid": (100.0 * n_valid / n_frames) if n_frames else 0.0,
        "hitbox_half_width_px": half_width,
        "nose_x_range": None,
        "nose_y_range": None,
        "node_breakdown": node_breakdown,
    }

    if n_valid > 0:
        vx = nose[valid, 0]
        vy = nose[valid, 1]
        info["nose_x_range"] = (float(np.min(vx)), float(np.max(vx)))
        info["nose_y_range"] = (float(np.min(vy)), float(np.max(vy)))

    head_vec = nose - neck
    torso = track.xy(cfg.NODE_TORSO)
    fps = track.fps
    min_bout_frames = max(1, round(cfg.MIN_BOUT_DURATION_S * fps))

    per_object = {}

    for obj_name, obj_xy in object_coords.items():
        obj_xy = np.asarray(obj_xy, dtype=np.float64)
        to_obj = obj_xy[None, :] - nose
        dist = np.linalg.norm(to_obj, axis=1)
        oriented = _head_cone_intersects_object(nose, head_vec, obj_xy, obj_half_width, cfg=cfg)
        in_hitbox = (np.abs(to_obj[:, 0]) <= half_width) & (np.abs(to_obj[:, 1]) <= half_width)

        dist_where_valid = np.where(valid, dist, np.inf)
        if n_valid > 0 and np.isfinite(dist_where_valid).any():
            closest_frame = int(np.argmin(dist_where_valid))
            closest_dist = float(dist_where_valid[closest_frame])
            closest_oriented = bool(oriented[closest_frame])
        else:
            closest_frame = None
            closest_dist = None
            closest_oriented = None

        # Distinguish two different reasons a real approach might not turn
        # into a scored bout: (a) tracking confidence drops specifically
        # while the rat is near this object (e.g. head-down sniffing angle
        # the model wasn't trained on much, or partial self-occlusion),
        # which silently drops those frames from `valid` before they ever
        # reach the hitbox/angle test; vs (b) the approach genuinely gets
        # scored frame-by-frame but each contiguous run is shorter than
        # MIN_BOUT_DURATION_S and gets discarded as noise. Both look like
        # "missing bouts" from the outside but need different fixes.
        in_hitbox_raw = (
            ~np.isnan(nose).any(axis=1)
            & (np.abs(to_obj[:, 0]) <= half_width)
            & (np.abs(to_obj[:, 1]) <= half_width)
        )
        n_in_hitbox_raw = int(in_hitbox_raw.sum())
        n_in_hitbox_valid = int((in_hitbox_raw & valid).sum())
        suppressed = in_hitbox_raw & ~valid
        n_suppressed_by_confidence = int(suppressed.sum())
        mean_conf_when_suppressed = (
            float(np.nanmean(np.nan_to_num(nose_conf[suppressed], nan=0.0))) if suppressed.any() else None
        )

        meets_criteria = valid & in_hitbox & oriented
        if not np.all(np.isnan(torso)):
            torso_dist = np.linalg.norm(obj_xy[None, :] - torso, axis=1)
            climbing = torso_dist <= cfg.CLIMBING_TORSO_DISTANCE_PX
            meets_criteria = meets_criteria & ~climbing

        runs = _mask_to_runs(meets_criteria)
        merged_runs, merge_events = _merge_bout_runs(runs, valid, fps, cfg=cfg)
        surviving = [r for r in merged_runs if (r[1] - r[0]) >= min_bout_frames]
        dropped = [r for r in merged_runs if (r[1] - r[0]) < min_bout_frames]
        dropped_lengths_s = sorted(((stop - start) / fps for start, stop in dropped))

        per_object[obj_name] = {
            "object_xy": [float(obj_xy[0]), float(obj_xy[1])],
            "n_confidence_merges": len(merge_events),
            "closest_frame": closest_frame,
            "closest_distance_px": closest_dist,
            "oriented_at_closest": closest_oriented,
            "n_frames_in_hitbox": int((in_hitbox & valid).sum()),
            "n_frames_oriented": int((oriented & valid).sum()),
            "n_frames_both": int((in_hitbox & oriented & valid).sum()),
            "n_in_hitbox_raw": n_in_hitbox_raw,
            "n_in_hitbox_valid": n_in_hitbox_valid,
            "n_suppressed_by_confidence": n_suppressed_by_confidence,
            "mean_conf_when_suppressed": mean_conf_when_suppressed,
            "n_candidate_bouts": len(merged_runs),
            "n_candidate_bouts_surviving": len(surviving),
            "n_candidate_bouts_dropped_too_short": len(dropped),
            "dropped_bout_lengths_s": dropped_lengths_s,
        }

    info["per_object"] = per_object
    return info


def compute_frame_details(track, object_coords, cfg=config):
    """Per-frame, per-object detail for building the annotated validation
    video (see validation_video.py): not just the final raw label, but
    *why* each frame did or didn't count -- valid tracking, in-hitbox,
    within-angle, climbing-excluded, and whether it landed inside an
    actually-counted bout versus a candidate run that got filtered out.

    Reuses score_frame_labels() and the same frame-index run/merge/filter
    helpers labels_to_bouts() uses, so a validation video's "counted"
    status always matches the Excel output exactly.
    """
    half_width = hitbox_half_width_px(cfg)
    obj_half_width = object_half_width_px(cfg)

    nose = track.xy(cfg.NODE_NOSE)
    neck = track.xy(cfg.NODE_NECK)
    torso = track.xy(cfg.NODE_TORSO)

    valid = _valid_mask(track, cfg)
    head_vec = nose - neck

    n_frames = track.n_frames
    raw_labels = score_frame_labels(track, object_coords, cfg=cfg)

    fps = track.fps
    min_bout_frames = max(1, round(cfg.MIN_BOUT_DURATION_S * fps))

    counted_label = np.zeros(n_frames, dtype=np.int8)
    merge_events_by_label = {}
    for lab in sorted(set(raw_labels.tolist()) - {0}):
        mask = raw_labels == lab
        runs = _mask_to_runs(mask)
        runs, merge_events = _merge_bout_runs(runs, valid, fps, cfg=cfg)
        runs = [r for r in runs if (r[1] - r[0]) >= min_bout_frames]
        merge_events_by_label[lab] = merge_events
        for start, stop in runs:
            counted_label[start:stop] = lab

    per_object = {}
    for obj_i, (obj_name, obj_xy) in enumerate(object_coords.items(), start=1):
        obj_xy = np.asarray(obj_xy, dtype=np.float64)
        to_obj = obj_xy[None, :] - nose
        oriented = _head_cone_intersects_object(nose, head_vec, obj_xy, obj_half_width, cfg=cfg)
        in_hitbox = (np.abs(to_obj[:, 0]) <= half_width) & (np.abs(to_obj[:, 1]) <= half_width)

        if not np.all(np.isnan(torso)):
            torso_dist = np.linalg.norm(obj_xy[None, :] - torso, axis=1)
            climbing = torso_dist <= cfg.CLIMBING_TORSO_DISTANCE_PX
        else:
            climbing = np.zeros(n_frames, dtype=bool)

        per_object[obj_name] = {
            "obj_i": obj_i,
            "in_hitbox": in_hitbox,
            "within_angle": oriented,
            "climbing": climbing,
        }

    return {
        "valid": valid,
        "raw_labels": raw_labels,
        "counted_label": counted_label,
        "merge_events_by_label": merge_events_by_label,
        "per_object": per_object,
    }


def labels_to_bouts(labels, track, cfg=config):
    """Convert a per-frame label array into bouts per object label value.

    `track` supplies fps and the tracking-confidence data
    _merge_bout_runs() uses to tell a genuine break in contact from a
    tracking dropout that merely looks like one -- see this module's
    docstring's note on confidence-aware merging.

    Returns (bouts_by_label, merge_events_by_label):
      - bouts_by_label: {label_value: [Bout, ...]} (label_value 0 excluded).
      - merge_events_by_label: {label_value: [(gap_start_frame,
        gap_stop_frame_exclusive), ...]} -- the low-confidence gaps that
        got bridged (always empty per label if
        cfg.CONFIDENCE_AWARE_MERGE_ENABLED is False). See
        review_flags.py's REVIEW_MAX_AUTO_MERGES flag and
        gui/batch_review_dialog.py's scrub-bar markers, both of which
        exist so a person can spot-check these rather than trusting them
        blindly.
    """
    fps = track.fps
    valid = _valid_mask(track, cfg)
    min_bout_frames = max(1, round(cfg.MIN_BOUT_DURATION_S * fps))

    bouts_by_label = {}
    merge_events_by_label = {}
    unique_labels = sorted(set(labels.tolist()) - {0})

    for lab in unique_labels:
        mask = labels == lab
        runs = _mask_to_runs(mask)
        merged_runs, merge_events = _merge_bout_runs(runs, valid, fps, cfg=cfg)
        merged_runs = [r for r in merged_runs if (r[1] - r[0]) >= min_bout_frames]
        bouts_by_label[lab] = [
            Bout(start_s=start / fps, stop_s=stop / fps) for start, stop in merged_runs
        ]
        merge_events_by_label[lab] = merge_events

    return bouts_by_label, merge_events_by_label


def _mask_to_runs(mask):
    """Boolean array -> list of (start_idx, stop_idx_exclusive) runs of True."""
    runs = []
    in_run = False
    start = None
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run = True
            start = i
        elif not v and in_run:
            in_run = False
            runs.append((start, i))
    if in_run:
        runs.append((start, len(mask)))
    return runs


def _merge_bout_runs(runs, valid, fps, cfg=config):
    """Merge nearby same-label runs into bouts, using a looser threshold
    for gaps the tracker didn't actually see confidently (no real
    evidence the animal left) than for gaps it confidently tracked as
    "not exploring" (real evidence of a break) -- see this module's
    docstring and config.py's CONFIDENCE_AWARE_MERGE_ENABLED section.

    valid: the same per-frame confident-tracking mask used to build
    `runs` in the first place (see _valid_mask) -- NOT restricted to the
    gap; this function does the slicing itself.

    Falls back to a single flat cfg.MERGE_GAP_S threshold for every gap
    (the old behavior) if cfg.CONFIDENCE_AWARE_MERGE_ENABLED is False.

    Returns (merged_runs, merge_events) -- merge_events is the list of
    (gap_start_frame, gap_stop_frame_exclusive) low-confidence gaps that
    got bridged *because* of the looser confidence threshold -- i.e.
    gaps longer than cfg.MERGE_GAP_S, which would NOT have merged under
    the plain confident-gap threshold. A low-confidence gap short enough
    to merge either way isn't a judgment call worth surfacing. Always []
    when confidence-aware merging is off, or when no gap qualified.
    """
    if not runs:
        return runs, []

    confident_gap_frames = max(1, round(cfg.MERGE_GAP_S * fps))
    confidence_aware = getattr(cfg, "CONFIDENCE_AWARE_MERGE_ENABLED", True)
    low_confidence_gap_frames = max(
        1, round(getattr(cfg, "LOW_CONFIDENCE_GAP_MERGE_S", cfg.MERGE_GAP_S) * fps)
    )
    low_confidence_fraction = getattr(cfg, "LOW_CONFIDENCE_GAP_FRACTION", 0.9)

    merged = [runs[0]]
    merge_events = []
    for start, stop in runs[1:]:
        prev_start, prev_stop = merged[-1]
        gap = start - prev_stop

        is_low_confidence_gap = False
        if confidence_aware and gap > 0:
            gap_slice = valid[prev_stop:start]
            if gap_slice.size > 0:
                invalid_fraction = 1.0 - (gap_slice.sum() / gap_slice.size)
                is_low_confidence_gap = invalid_fraction >= low_confidence_fraction

        threshold = low_confidence_gap_frames if is_low_confidence_gap else confident_gap_frames
        if gap <= threshold:
            merged[-1] = (prev_start, stop)
            # Only count this as an "auto-merge" event if the confidence
            # slack is actually what bridged it -- a low-confidence gap
            # short enough to merge under the plain MERGE_GAP_S threshold
            # anyway isn't a judgment call this feature made; it's a gap
            # that would've merged regardless, and shouldn't count toward
            # REVIEW_MAX_AUTO_MERGES or show up as a scrub-bar marker.
            if is_low_confidence_gap and gap > confident_gap_frames:
                merge_events.append((prev_stop, start))
        else:
            merged.append((start, stop))

    return merged, merge_events
