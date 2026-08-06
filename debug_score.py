"""
Fast iteration loop for debugging scoring.py / pose_utils.py without
waiting on SLEAP inference: scores one video using its cached prediction
in predictions/ (from a previous `main.py run`) and prints the resulting
bouts to the console.

Usage:
    python debug_score.py "359 novel a"            # score, matched by filename substring
    python debug_score.py "359 novel a" --xlsx     # also write output/debug_<name>.xlsx
    python debug_score.py "359 novel a" --infer    # run inference first if nothing is cached
    python debug_score.py "359 novel a" --validate # also write the annotated validation mp4

Requires a cached predictions/<video>.predictions.slp by default; pass
--infer to run inference for just this one video instead.
"""

import argparse
import sys
from pathlib import Path

import config
import object_picker
import sleap_inference
import scoring
import excel_writer
import frame_export
import validation_video
import pose_filters
from main import validation_video_path_for
from pose_utils import load_track_from_slp


def find_matching_video(query):
    videos = sorted(Path(config.VIDEO_FOLDER).rglob("*.mp4"))
    matches = [v for v in videos if query.lower() in v.name.lower()]
    if not matches:
        print(f"No video under {config.VIDEO_FOLDER} matches '{query}'.")
        print("Available videos:")
        for v in videos:
            print(f"  {v.name}")
        sys.exit(1)
    if len(matches) > 1:
        print(f"'{query}' matches multiple videos, be more specific:")
        for v in matches:
            print(f"  {v.name}")
        sys.exit(1)
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="substring to match against a video filename in VIDEO_FOLDER")
    parser.add_argument("--infer", action="store_true", help="run inference if no cached prediction exists yet")
    parser.add_argument(
        "--reinfer", action="store_true",
        help="force a fresh inference run even if a prediction is already cached -- use after "
             "retraining/switching MODEL_PATHS, since the cache is keyed only by video filename",
    )
    parser.add_argument("--xlsx", action="store_true", help="also write a one-video workbook to output/debug_<name>.xlsx")
    parser.add_argument(
        "--frames", nargs=2, type=float, metavar=("START_S", "END_S"),
        help="save annotated frames covering this time range (seconds) to output/frames_<video>_<start>-<end>s/, "
             "so you can see exactly what the model predicted during a specific bout",
    )
    parser.add_argument(
        "--slp", type=Path, default=None,
        help="score against this .slp file directly instead of the cached/inferred prediction -- e.g. a file "
             "exported from the SLEAP GUI, to A/B test the model/weights against a known-good prediction",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="write the full annotated validation video for this video to the same place "
             "`main.py run` would (output/validation_videos/<video>.validation.mp4), without "
             "running the whole batch. Always overwrites, unlike main.py run's skip-if-fresh "
             "caching, since you're explicitly asking to (re)generate this one.",
    )
    parser.add_argument(
        "--no-swap-filter", action="store_true",
        help="skip pose_filters.filter_nose_tail_swaps() entirely -- use this to A/B the "
             "nose/tail-swap filter's effect on bout counts/durations for this video (run once "
             "with, once without, and diff the printed bout lists).",
    )

    args = parser.parse_args()

    video_path = find_matching_video(args.query)
    print(f"Video: {video_path}")

    coords = object_picker.load_object_coords(config.OBJECT_COORDS_FILE)
    key = object_picker.video_key(video_path, config.VIDEO_FOLDER)
    if key not in coords:
        print(f"No object positions set for this video yet. Run `python main.py setup` first.")
        sys.exit(1)
    obj_coords = coords[key]
    print(f"Object positions: {obj_coords}")

    if args.slp is not None:
        if not args.slp.exists():
            print(f"--slp path does not exist: {args.slp}")
            sys.exit(1)
        slp_path = args.slp
        print(f"Using explicit --slp override (skipping cache/inference): {slp_path}")
    else:
        slp_path = sleap_inference.predictions_path_for(video_path, config.PREDICTIONS_FOLDER)

    if args.slp is not None:
        pass  # already resolved above -- skip all cache/inference logic below
    elif args.reinfer:
        if slp_path.exists():
            print(f"--reinfer: forcing a fresh run, ignoring cached {slp_path}")
        base_cmd = sleap_inference.check_sleap_track_available()
        slp_path = sleap_inference.run_inference(
            video_path,
            config.MODEL_PATHS,
            config.PREDICTIONS_FOLDER,
            base_cmd=base_cmd,
            overwrite=True,
            max_height=config.INFERENCE_MAX_HEIGHT,
            max_width=config.INFERENCE_MAX_WIDTH,
        )
    elif not slp_path.exists():
        if not args.infer:
            print(f"No cached prediction at {slp_path}.")
            print("Run `python main.py run` first (or once, then Ctrl+C after this video),")
            print("or pass --infer to run inference for just this video now.")
            sys.exit(1)
        print("No cached prediction -- running inference for this video only...")
        base_cmd = sleap_inference.check_sleap_track_available()
        slp_path = sleap_inference.run_inference(
            video_path,
            config.MODEL_PATHS,
            config.PREDICTIONS_FOLDER,
            base_cmd=base_cmd,
            max_height=config.INFERENCE_MAX_HEIGHT,
            max_width=config.INFERENCE_MAX_WIDTH,
        )
    else:
        # Cache hit -- warn if it's older than the currently configured model.
        model_mtime = None
        for model_path in config.MODEL_PATHS:
            ckpt = Path(model_path) / "best.ckpt"
            if ckpt.exists():
                model_mtime = max(model_mtime or 0, ckpt.stat().st_mtime)
        if model_mtime is not None and slp_path.stat().st_mtime < model_mtime:
            print(f"WARNING: cached prediction at {slp_path} is OLDER than the model")
            print(f"currently configured in MODEL_PATHS ({config.MODEL_PATHS[0]}).")
            print("This cache was almost certainly produced by a different/older model.")
            print("Re-run with --reinfer to force a fresh prediction from the current model.")

    print(f"Loading track from: {slp_path}")
    track = load_track_from_slp(slp_path, video_path=video_path)
    print(f"  {track.n_frames} frames @ {track.fps} fps, nodes: {track.node_names}")
    print(f"  ({track.n_frames / track.fps:.1f}s implied by frame count / fps -- sanity check "
          f"this against the video's actual known duration; a mismatch means fps is wrong and "
          f"every merge-gap/min-duration threshold below is off.)")

    if args.no_swap_filter:
        print("  --no-swap-filter: skipping nose/tail-swap filtering (A/B mode)")
    else:
        track, n_swap_frames = pose_filters.filter_nose_tail_swaps(track, cfg=config, verbose=True)
        if n_swap_frames:
            print(f"  Filtered {n_swap_frames} isolated nose/tail-swap frame(s) (see above)")
        else:
            print("  No isolated nose/tail-swap frames detected.")

    labels = scoring.score_frame_labels(track, obj_coords, cfg=config)
    bouts, merge_events = scoring.labels_to_bouts(labels, track, cfg=config)

    object_names = list(obj_coords.keys())
    print()
    for obj_i, obj_name in enumerate(object_names, start=1):
        obj_bouts = bouts.get(obj_i, [])
        total_time = sum(b.duration_s for b in obj_bouts)
        print(f"{obj_name}: {len(obj_bouts)} bout(s), {total_time:.2f}s total")
        for b in obj_bouts:
            print(f"    {b.start_s:.2f}s -> {b.stop_s:.2f}s  ({b.duration_s:.2f}s)")
        events = merge_events.get(obj_i, [])
        if events:
            print(f"    ({len(events)} auto-merge(s) across a low-confidence gap:")
            for start, stop in events:
                print(f"       {start / track.fps:.2f}s -> {stop / track.fps:.2f}s)")

    diag = scoring.diagnose(track, obj_coords, cfg=config)
    print()
    print("--- Diagnostics ---")
    print(f"Valid (trustworthy) tracking frames: {diag['n_valid_frames']}/{diag['n_frames']} ({diag['pct_valid']:.1f}%)")
    if diag['pct_valid'] < 50:
        print("  ^ LOW. Most frames have missing/low-confidence nose or neck points.")
        print("    Check MIN_NODE_CONFIDENCE in config.py, or whether the model is tracking this video well at all.")
        print("    Per-node breakdown below shows which node(s) are the bottleneck.")
    print()
    print("Per-node tracking quality (pct_present = model predicted a point at all;")
    print(f"pct_confident = present AND confidence >= MIN_NODE_CONFIDENCE ({config.MIN_NODE_CONFIDENCE})):")
    for node_name, stats in diag["node_breakdown"].items():
        mean_conf_str = f"{stats['mean_confidence']:.2f}" if stats["mean_confidence"] is not None else "n/a"
        print(f"  {node_name:12s} present: {stats['pct_present']:5.1f}%   confident: {stats['pct_confident']:5.1f}%   mean conf (when present): {mean_conf_str}")
    if diag["nose_x_range"]:
        print(f"Nose x range: {diag['nose_x_range'][0]:.1f} - {diag['nose_x_range'][1]:.1f} px")
        print(f"Nose y range: {diag['nose_y_range'][0]:.1f} - {diag['nose_y_range'][1]:.1f} px")
    print(f"Hitbox half-width: {diag['hitbox_half_width_px']:.1f} px")
    for obj_name, stats in diag["per_object"].items():
        print(f"\n  {obj_name} @ {stats['object_xy']}:")
        if stats["closest_frame"] is None:
            print("    No valid frames to compare against.")
            continue
        oriented_str = "yes" if stats["oriented_at_closest"] else "no"
        print(f"    Closest approach: {stats['closest_distance_px']:.1f} px at frame {stats['closest_frame']} "
              f"(head ray hit the object at that frame: {oriented_str})")
        print(f"    Frames within hitbox (any orientation): {stats['n_frames_in_hitbox']}")
        print(f"    Frames with head ray hitting the object (any distance): {stats['n_frames_oriented']}")
        print(f"    Frames meeting BOTH (before climbing exclusion/min-duration): {stats['n_frames_both']}")

        obj_idx = object_names.index(obj_name) + 1
        n_bouts_scored = len(bouts.get(obj_idx, []))

        if stats["closest_distance_px"] > diag["hitbox_half_width_px"] * 3:
            print("    ^ Nose never gets remotely close to this object -- the clicked object")
            print("      position likely doesn't match where the object actually is in this")
            print("      video (wrong click, or video/object coordinates don't align).")
        elif stats["n_frames_in_hitbox"] == 0:
            print("    ^ Nose gets somewhat close but never inside the hitbox --")
            print("      consider increasing OBJECT_PADDING_CM/OBJECT_SIZE_CM or re-checking PX_PER_CM.")
        elif stats["n_frames_oriented"] == 0:
            print("    ^ Nose is near the object but the head-direction ray never crosses it --")
            print("      double check the object coordinates and OBJECT_FOOTPRINT_GROW_PX in config.py.")

        if stats["n_suppressed_by_confidence"] > 0:
            pct_suppressed = (
                100.0 * stats["n_suppressed_by_confidence"] / stats["n_in_hitbox_raw"]
                if stats["n_in_hitbox_raw"] else 0.0
            )
            print(f"    Frames where nose was near this object but tracking wasn't trustworthy: "
                  f"{stats['n_suppressed_by_confidence']}/{stats['n_in_hitbox_raw']} ({pct_suppressed:.0f}%), "
                  f"mean confidence {stats['mean_conf_when_suppressed']:.2f}" if stats["mean_conf_when_suppressed"] is not None
                  else f"    Frames where nose was near this object but tracking wasn't trustworthy: {stats['n_suppressed_by_confidence']}")
            if pct_suppressed > 30:
                print("      ^ Confidence drops specifically while the rat is near this object --")
                print("        likely head-down/occluded poses under-represented in training data.")

        if stats["n_candidate_bouts_dropped_too_short"] > 0:
            lens = stats["dropped_bout_lengths_s"]
            print(f"    Candidate bouts dropped as too short (< MIN_BOUT_DURATION_S={config.MIN_BOUT_DURATION_S}s): "
                  f"{stats['n_candidate_bouts_dropped_too_short']} (lengths: "
                  + ", ".join(f"{l:.2f}s" for l in lens) + ")")
            print(f"      ^ These would have added up to {sum(lens):.2f}s more exploration time if not")
            print(f"        filtered -- consider lowering MIN_BOUT_DURATION_S if this looks like real sniffing")
            print(f"        rather than tracking jitter.")

        print(f"    Bouts actually scored: {n_bouts_scored}")

    if args.frames:
        start_s, end_s = args.frames
        frames_dir = Path(config.OUTPUT_FOLDER) / f"frames_{video_path.stem}_{start_s:g}-{end_s:g}s"
        print(f"\nSaving annotated frames ({start_s:g}s-{end_s:g}s) to {frames_dir} ...")
        saved = frame_export.export_annotated_frames(video_path, track, obj_coords, config, start_s, end_s, frames_dir)
        print(f"Saved {len(saved)} frames. Green nose = trusted, red nose = below MIN_NODE_CONFIDENCE, cyan = neck.")
        print("If the nose marker is confidently (green) landing somewhere far from the object during")
        print("a bout you know is real, that's a misprediction, not a confidence/duration filtering issue --")
        print("likely the object is blocking the camera's view of the nose during close contact.")

    if args.validate:
        validation_folder = Path(config.OUTPUT_FOLDER) / "validation_videos"
        validation_path = validation_video_path_for(video_path, validation_folder)
        print(f"\nWriting validation video: {validation_path} ...")
        validation_video.export_validation_video(video_path, track, obj_coords, config, validation_path)
        print(f"Wrote {validation_path}")

    if args.xlsx:
        config.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        out_path = Path(config.OUTPUT_FOLDER) / f"debug_{video_path.stem}.xlsx"
        rat_id = video_path.stem
        results = {
            rat_id: [{
                "session_label": video_path.stem,
                "video_name": video_path.name,
                "bouts": bouts,
            }]
        }
        excel_writer.write_workbook(results, out_path, object_names=object_names)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
