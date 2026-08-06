"""
Compares the model+scoring pipeline's bouts against human-coded ground
truth, for validating/retuning scoring.py's thresholds (SNIFF_CONE_HALF_ANGLE_DEG,
SNIFF_RAY_ORIGIN_BACKSET_RATIO, MERGE_GAP_S, MIN_BOUT_DURATION_S, etc.)
against real behavioral data instead of guesses.

Usage:
    python benchmark.py human_bouts.json \\
        --predictions-dir app_data/predictions/<job_id>_<group> \\
        --coords app_data/coords/<job_id>_<group>.json \\
        --video-dir "/path/to/28 days"

Or, for the CLI pipeline's own predictions/coords (config.py defaults):
    python benchmark.py human_bouts.json --video-dir "/path/to/28 days"

human_bouts.json format: {"<rat_id>_<session>": {"novel": [[start_s, stop_s], ...],
"original": [[start_s, stop_s], ...], "novel_shape": "Square"|"Hexagon"}, ...}
-- see parse script/README for how this is produced from a human coder's notes.

Matching is event-level, not 1:1: a human bout counts as "detected" if any
model bout overlaps it within --tolerance seconds, and vice versa for model
bouts counted as "real". This is simple and standard for this kind of
comparison, but can look generous if the model fragments one long human
bout into several short ones (each fragment "detects" the human bout, and
also each counts as a valid model bout) -- the per-video bout-count columns
in the report make that visible even though the score doesn't penalize it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import config
import object_picker
import pose_filters
import scoring
from pose_utils import load_track_from_slp


def _find_video(video_dir, rat_id, session):
    video_dir = Path(video_dir)
    matches = list(video_dir.glob(f"{rat_id} novel {session}*.mp4")) + list(
        video_dir.glob(f"{rat_id} Novel {session}*.mp4")
    )
    if not matches:
        return None
    return matches[0]


def _overlaps(a, b, tolerance):
    a_start, a_stop = a
    b_start, b_stop = b
    return (a_start - tolerance) <= b_stop and (b_start - tolerance) <= a_stop


def match_bouts(human, model, tolerance=1.0):
    """human, model: lists of (start_s, stop_s). Event-level overlap
    matching (see module docstring) -- not a strict 1:1 assignment.
    """
    human_hit = [False] * len(human)
    model_hit = [False] * len(model)
    for i, h in enumerate(human):
        for j, m in enumerate(model):
            if _overlaps(h, m, tolerance):
                human_hit[i] = True
                model_hit[j] = True

    n_human, n_model = len(human), len(model)
    tp_recall = sum(human_hit)
    tp_precision = sum(model_hit)
    recall = (tp_recall / n_human) if n_human else None
    precision = (tp_precision / n_model) if n_model else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "n_human": n_human, "n_model": n_model,
        "human_detected": tp_recall, "model_real": tp_precision,
        "precision": precision, "recall": recall, "f1": f1,
    }


def run_benchmark(human_bouts_path, video_dir, predictions_dir, coords_path, tolerance=1.0, cfg=config):
    human_data = json.loads(Path(human_bouts_path).read_text())
    coords = object_picker.load_object_coords(coords_path)

    rows = []
    skipped = []

    for key, entry in sorted(human_data.items()):
        rat_id, session = key.split("_")
        video_path = _find_video(video_dir, rat_id, session)
        if video_path is None:
            skipped.append((key, "video not found"))
            continue

        video_key = object_picker.video_key(video_path, video_dir)
        if video_key not in coords:
            skipped.append((key, "no object coordinates set"))
            continue
        obj_coords = coords[video_key]

        slp_path = Path(predictions_dir) / f"{video_path.stem}.predictions.slp"
        if not slp_path.exists():
            skipped.append((key, f"no cached prediction at {slp_path}"))
            continue

        track = load_track_from_slp(slp_path, video_path=video_path)
        track, _ = pose_filters.filter_nose_tail_swaps(track, cfg=cfg)
        labels = scoring.score_frame_labels(track, obj_coords, cfg=cfg)
        model_bouts, _merge_events = scoring.labels_to_bouts(labels, track, cfg=cfg)

        object_names = list(obj_coords.keys())  # [novel_name, original_name], in click order
        for obj_i, obj_kind in ((1, "novel"), (2, "original")):
            obj_name = object_names[obj_i - 1] if obj_i - 1 < len(object_names) else obj_kind
            human_list = [tuple(x) for x in entry.get(obj_kind, [])]
            model_list = [(b.start_s, b.stop_s) for b in model_bouts.get(obj_i, [])]
            m = match_bouts(human_list, model_list, tolerance=tolerance)
            m.update({
                "rat_id": rat_id, "session": session, "object": obj_kind,
                "object_label": obj_name,
                "human_total_s": sum(e - s for s, e in human_list),
                "model_total_s": sum(e - s for s, e in model_list),
            })
            rows.append(m)

    return rows, skipped


def print_report(rows, skipped):
    if skipped:
        print(f"Skipped {len(skipped)} video-session(s):")
        for key, reason in skipped:
            print(f"  {key}: {reason}")
        print()

    if not rows:
        print("No videos scored -- nothing to report.")
        return

    print(f"{'rat':>5} {'sess':>4} {'obj':>8} {'n_h':>4} {'n_m':>4} {'prec':>6} {'rec':>6} {'f1':>6} {'human_s':>8} {'model_s':>8}")
    for r in rows:
        prec = f"{r['precision']:.2f}" if r["precision"] is not None else "n/a"
        rec = f"{r['recall']:.2f}" if r["recall"] is not None else "n/a"
        f1 = f"{r['f1']:.2f}" if r["f1"] is not None else "n/a"
        print(
            f"{r['rat_id']:>5} {r['session']:>4} {r['object']:>8} "
            f"{r['n_human']:>4} {r['n_model']:>4} {prec:>6} {rec:>6} {f1:>6} "
            f"{r['human_total_s']:>8.1f} {r['model_total_s']:>8.1f}"
        )

    precisions = [r["precision"] for r in rows if r["precision"] is not None]
    recalls = [r["recall"] for r in rows if r["recall"] is not None]
    human_totals = np.array([r["human_total_s"] for r in rows])
    model_totals = np.array([r["model_total_s"] for r in rows])

    print()
    print(f"Overall bout-level: mean precision={np.mean(precisions):.3f}, mean recall={np.mean(recalls):.3f}")
    print(f"Total exploration time: mean absolute error={np.mean(np.abs(human_totals - model_totals)):.2f}s, "
          f"correlation r={np.corrcoef(human_totals, model_totals)[0, 1]:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("human_bouts", type=Path, help="path to human_bouts.json")
    parser.add_argument("--video-dir", type=Path, required=True, help="folder containing the benchmark videos")
    parser.add_argument("--predictions-dir", type=Path, default=None, help="defaults to config.PREDICTIONS_FOLDER")
    parser.add_argument("--coords", type=Path, default=None, help="defaults to config.OBJECT_COORDS_FILE")
    parser.add_argument("--tolerance", type=float, default=1.0, help="overlap tolerance in seconds (default 1.0)")
    args = parser.parse_args()

    predictions_dir = args.predictions_dir or config.PREDICTIONS_FOLDER
    coords_path = args.coords or config.OBJECT_COORDS_FILE

    rows, skipped = run_benchmark(
        args.human_bouts, args.video_dir, predictions_dir, coords_path, tolerance=args.tolerance,
    )
    print_report(rows, skipped)


if __name__ == "__main__":
    main()
