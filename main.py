"""
NOR classifier app -- entry point.

Usage:
    python main.py setup      # interactively click object locations for
                               # any new videos in VIDEO_FOLDER (one-time
                               # per video; skips ones already configured)
    python main.py run        # unattended batch: run SLEAP inference +
                               # scoring on every video that has object
                               # coordinates set, write Excel output(s),
                               # and write an annotated validation video
                               # for every video (see below)
    python main.py all        # setup, then run

Add --skip-validation to `run`/`all` to skip writing the per-video
annotated validation videos (they take noticeably longer than scoring
alone) -- useful while iterating quickly; leave it off for a real run
meant to back a publication.

Edit config.py first to point at your videos, model, and skeleton node
names, and to tune the exploration-scoring thresholds.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

import config
import object_picker
import sleap_inference
import scoring
import excel_writer
import validation_video
import pose_filters
import review_flags
from pose_utils import load_track_from_slp

# gui/ isn't on sys.path by default -- only cmd_setup() needs it (to reuse
# the same Qt object-setup dialog the GUI uses; see cmd_setup()'s docstring).
sys.path.insert(0, str(Path(__file__).resolve().parent / "gui"))


def find_videos(folder):
    folder = Path(folder)
    return sorted(folder.rglob("*.mp4"))


def parse_filename(video_path):
    """Pull rat_id / phase / session out of the filename using
    config.FILENAME_PATTERN. Falls back to using the whole stem as the
    rat ID if the pattern doesn't match."""
    stem = Path(video_path).stem
    m = re.match(config.FILENAME_PATTERN, stem)
    if m:
        return m.group("rat_id"), m.groupdict().get("phase", ""), m.groupdict().get("session", "")
    return stem, "", ""


def cmd_setup():
    """Object setup used to open its own standalone cv2 window
    (object_picker.setup_missing_videos()); that's gone now -- this opens
    the same PySide6 dialog the GUI's "Set Up Objects..." button uses
    (gui/object_setup_dialog.py), just run standalone with a throwaway
    QApplication rather than as a child of the main window."""
    videos = find_videos(config.VIDEO_FOLDER)
    if not videos:
        print(f"No .mp4 files found under {config.VIDEO_FOLDER}")
        return

    from PySide6.QtWidgets import QApplication
    from object_setup_dialog import run_object_setup

    print(f"Found {len(videos)} video(s). Launching object setup for any not yet configured...")
    app = QApplication.instance() or QApplication(sys.argv)
    run_object_setup(videos, config.OBJECT_COORDS_FILE, config)
    print("Setup complete.")


def validation_video_path_for(video_path, validation_folder):
    video_path = Path(video_path)
    validation_folder = Path(validation_folder)
    return validation_folder / f"{video_path.stem}.validation.mp4"


def cmd_run(skip_validation=False):
    videos = find_videos(config.VIDEO_FOLDER)
    coords = object_picker.load_object_coords(config.OBJECT_COORDS_FILE)

    ready_videos = []
    for v in videos:
        key = object_picker.video_key(v, config.VIDEO_FOLDER)
        if key not in coords:
            print(f"Skipping (no object coordinates set): {v}")
            continue
        ready_videos.append(v)

    if not ready_videos:
        print("No videos ready to process. Run `python main.py setup` first.")
        return

    base_cmd = sleap_inference.check_sleap_track_available()
    print(f"Using tracking command: {' '.join(base_cmd)}")
    config.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    out_path = Path(config.OUTPUT_FOLDER) / "NOR_results.xlsx"
    object_names = list(next(iter(coords.values())).keys()) if coords else ["novel", "original"]

    # Inference + scoring + workbook write happen per-video so the Excel
    # file stays up to date if the run is interrupted partway through.
    results = defaultdict(list)
    total = len(ready_videos)
    for i, video_path in enumerate(ready_videos, start=1):
        print(f"[{i}/{total}] Running SLEAP inference: {video_path}")
        slp_path = sleap_inference.run_inference(
            video_path,
            config.MODEL_PATHS,
            config.PREDICTIONS_FOLDER,
            base_cmd=base_cmd,
            max_height=config.INFERENCE_MAX_HEIGHT,
            max_width=config.INFERENCE_MAX_WIDTH,
        )

        rat_id, phase, session = parse_filename(video_path)
        key = object_picker.video_key(video_path, config.VIDEO_FOLDER)
        obj_coords = coords[key]

        print(f"  Scoring: {video_path}")
        track = load_track_from_slp(slp_path, video_path=video_path)
        track, n_swap_frames = pose_filters.filter_nose_tail_swaps(track, cfg=config)
        if n_swap_frames:
            print(f"  Filtered {n_swap_frames} isolated nose/tail-swap frame(s)")
        labels = scoring.score_frame_labels(track, obj_coords, cfg=config)
        bouts, merge_events = scoring.labels_to_bouts(labels, track, cfg=config)

        reasons = review_flags.compute_video_flags(track, obj_coords, cfg=config, merge_events=merge_events)
        review_flags.update_video_flags(config.OUTPUT_FOLDER, video_path.stem, reasons)
        if reasons:
            print(f"  FLAGGED for review: {reasons[0]}")
            for extra in reasons[1:]:
                print(f"    also: {extra}")

        results[rat_id].append({
            "session_label": Path(video_path).stem,
            "video_name": Path(video_path).name,
            "bouts": bouts,
        })

        excel_writer.write_workbook(results, out_path, object_names=object_names)
        print(f"  Updated {out_path} ({i}/{total} videos)")

        if not skip_validation:
            validation_folder = Path(config.OUTPUT_FOLDER) / "validation_videos"
            validation_path = validation_video_path_for(video_path, validation_folder)
            # Skip regenerating if it's already up to date with the .slp.
            if validation_path.exists() and validation_path.stat().st_mtime >= Path(slp_path).stat().st_mtime:
                print(f"  Validation video up to date: {validation_path}")
            else:
                print(f"  Writing validation video: {validation_path}")
                validation_video.export_validation_video(video_path, track, obj_coords, config, validation_path)
                print(f"  Wrote {validation_path}")

    print(f"Done. Wrote {out_path}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("setup", "run", "all"):
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]
    skip_validation = "--skip-validation" in sys.argv[2:]

    if mode in ("setup", "all"):
        cmd_setup()
    if mode in ("run", "all"):
        cmd_run(skip_validation=skip_validation)


if __name__ == "__main__":
    main()
