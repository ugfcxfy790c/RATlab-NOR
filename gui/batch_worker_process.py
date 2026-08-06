"""
Batch pipeline execution -- runs in a completely separate OS process
(via multiprocessing, spawned fresh) rather than a thread inside the GUI
process.

Why a *process* and not a QThread: this pipeline decodes/encodes video
through OpenCV and shells out to SLEAP/ffmpeg while the GUI's own Qt
event loop is running on the main thread. On macOS, doing OpenCV video
I/O on a background thread while a Cocoa/Qt run loop is active
elsewhere in the *same process* is a known source of hard crashes
(segfaults) -- OpenCV's build bundles its own windowing/Cocoa
integration that isn't safe to touch concurrently with PySide6's.
Running the pipeline in its own process sidesteps this entirely: the
GUI process only ever does Qt work, the worker process only ever does
OpenCV/SLEAP work, and there's no native state shared between them.

Communicates back to the GUI over a multiprocessing.Queue of small,
plain-data messages (tuples of str/int) -- deliberately not pickling
Job/JobStatus objects across the process boundary, so there's no
import-order dependency during unpickling in the freshly spawned
interpreter (which starts with a blank sys.path until we set it below).

Message kinds put on the queue:
    ("log", str)
    ("job_started", job_id)
    ("job_progress", job_id, done, total)
    ("job_finished", job_id, status_value, error_message)
    ("all_finished",)
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path


def _parse_filename(video_path, cfg):
    """Same convention as main.py's parse_filename()."""
    stem = Path(video_path).stem
    pattern = getattr(cfg, "FILENAME_PATTERN", None)
    if pattern:
        m = re.match(pattern, stem)
        if m:
            return m.group("rat_id"), m.groupdict().get("phase", ""), m.groupdict().get("session", "")
    return stem, "", ""


def run_batch_worker(nor_classifier_dir: str, job_dicts: list, app_data_dir: str, queue, stop_event):
    """Entry point run in the spawned worker process."""
    gui_dir = str(Path(nor_classifier_dir) / "gui")
    for p in (nor_classifier_dir, gui_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    import config
    import object_picker as op
    import sleap_inference
    import scoring
    import excel_writer
    import validation_video
    import pose_filters
    import review_flags
    from pose_utils import load_track_from_slp
    from job_queue import Job, find_videos, job_coords_path, job_predictions_folder
    from job_config import build_job_cfg
    from frozen_config import is_frozen, patch_config_for_ratlab_dir

    # This is a fresh interpreter (multiprocessing spawn), so if the GUI
    # process patched config.py's paths to point at the real RATlab
    # folder (see app.py/frozen_config.py -- only relevant for a
    # packaged .app/.exe/AppImage), that patch didn't carry over here.
    # Redo it using the same nor_classifier_dir the GUI resolved to.
    if is_frozen():
        patch_config_for_ratlab_dir(config, nor_classifier_dir)

    def log(msg):
        queue.put(("log", msg))

    try:
        base_cmd = sleap_inference.check_sleap_track_available()
    except Exception as exc:
        log(f"Can't find a SLEAP tracking command on PATH: {exc}")
        queue.put(("all_finished",))
        return

    log(f"Using tracking command: {' '.join(base_cmd)}")

    for job_dict in job_dicts:
        if stop_event.is_set():
            log("Stop requested -- not starting any further jobs.")
            break
        _run_job(
            job_dict, base_cmd, app_data_dir, queue, stop_event,
            config, op, sleap_inference, scoring, excel_writer, validation_video,
            pose_filters, review_flags, load_track_from_slp, Job, find_videos, job_coords_path,
            job_predictions_folder, build_job_cfg,
        )

    queue.put(("all_finished",))


def _run_job(
    job_dict, base_cmd, app_data_dir, queue, stop_event,
    config, op, sleap_inference, scoring, excel_writer, validation_video,
    pose_filters, review_flags, load_track_from_slp, Job, find_videos, job_coords_path,
    job_predictions_folder, build_job_cfg,
):
    # Reconstruct a real Job (rather than a dict/stand-in) now that
    # job_queue.Job is importable in this process -- keeps slug/
    # output_folder logic in the one place that defines it
    # (job_queue.py), instead of duplicating it here.
    # Local aliases so run_inference's log=/progress=... can forward its
    # streamed subprocess output through the same queue everything else
    # in this function uses -- run_batch_worker() has its own `log`
    # closure, but that's local to that function and isn't visible in
    # this one. Kept as two separate message kinds (not just "log" for
    # both) because the GUI redraws "progress" messages in place over
    # the previous one, but always appends "log" messages as new lines
    # -- see main_window.py's _on_progress_line / _log.
    def log(msg):
        queue.put(("log", msg))

    def progress(msg):
        queue.put(("progress", msg))

    job = Job.from_dict(job_dict)
    queue.put(("job_started", job.id))
    queue.put(("log", f"--- Starting '{job.group_name}' ---"))

    job_cfg = build_job_cfg(job, config)
    coords = op.load_object_coords(job_coords_path(job, Path(app_data_dir)))

    videos = find_videos(job.input_folder)
    ready_videos = []
    for v in videos:
        key = op.video_key(v, job.input_folder)
        if key not in coords:
            queue.put(("log", f"  Skipping (no object coordinates set): {v.name}"))
            continue
        ready_videos.append(v)

    if not ready_videos:
        queue.put(("job_finished", job.id, "failed", "No videos with object coordinates set."))
        return

    output_folder = job.output_folder
    try:
        output_folder.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        queue.put(("job_finished", job.id, "failed", f"Couldn't create output folder: {exc}"))
        return

    predictions_folder = job_predictions_folder(job, Path(app_data_dir))
    out_path = output_folder / f"{job.group_name}_NOR_results.xlsx"
    object_names = list(next(iter(coords.values())).keys()) if coords else ["novel", "original"]

    results = defaultdict(list)
    total = len(ready_videos)
    queue.put(("job_progress", job.id, 0, total))

    for i, video_path in enumerate(ready_videos, start=1):
        if stop_event.is_set():
            queue.put(("log", f"  Stopping '{job.group_name}' after {i - 1}/{total} video(s) (Stop was clicked)."))
            queue.put(("job_finished", job.id, "canceled", "Stopped by user."))
            return

        try:
            queue.put(("log", f"[{job.group_name} {i}/{total}] Tracking: {video_path.name}"))
            slp_path = sleap_inference.run_inference(
                video_path,
                job_cfg.MODEL_PATHS,
                predictions_folder,
                base_cmd=base_cmd,
                max_height=getattr(job_cfg, "INFERENCE_MAX_HEIGHT", None),
                max_width=getattr(job_cfg, "INFERENCE_MAX_WIDTH", None),
                log=log,
                progress=progress,
            )

            rat_id, phase, session = _parse_filename(video_path, job_cfg)
            key = op.video_key(video_path, job.input_folder)
            obj_coords = coords[key]

            queue.put(("log", f"  Scoring: {video_path.name}"))
            track = load_track_from_slp(slp_path, video_path=video_path)
            track, n_swap_frames = pose_filters.filter_nose_tail_swaps(track, cfg=job_cfg)
            if n_swap_frames:
                queue.put(("log", f"  Filtered {n_swap_frames} isolated nose/tail-swap frame(s)"))
            labels = scoring.score_frame_labels(track, obj_coords, cfg=job_cfg)
            bouts, merge_events = scoring.labels_to_bouts(labels, track, cfg=job_cfg)

            reasons = review_flags.compute_video_flags(track, obj_coords, cfg=job_cfg, merge_events=merge_events)
            review_flags.update_video_flags(output_folder, video_path.stem, reasons)
            if reasons:
                queue.put(("log", f"  FLAGGED for review: {reasons[0]}"))
                for extra in reasons[1:]:
                    queue.put(("log", f"    also: {extra}"))

            results[rat_id].append({
                "session_label": video_path.stem,
                "video_name": video_path.name,
                "bouts": bouts,
            })
            excel_writer.write_workbook(results, out_path, object_names=object_names)
            queue.put(("log", f"  Updated {out_path.name} ({i}/{total})"))

            if not job.skip_validation:
                validation_folder = output_folder / "validation_videos"
                validation_path = validation_folder / f"{video_path.stem}.validation.mp4"
                if validation_path.exists() and validation_path.stat().st_mtime >= Path(slp_path).stat().st_mtime:
                    queue.put(("log", f"  Validation video up to date: {validation_path.name}"))
                else:
                    queue.put(("log", f"  Writing validation video: {validation_path.name}"))
                    validation_video.export_validation_video(video_path, track, obj_coords, job_cfg, validation_path)
                    queue.put(("log", f"  Wrote {validation_path.name}"))

            queue.put(("job_progress", job.id, i, total))

        except Exception as exc:
            queue.put(("log", f"  ERROR on {video_path.name}: {exc}"))
            queue.put(("job_finished", job.id, "failed", f"{video_path.name}: {exc}"))
            return

    queue.put(("log", f"--- Finished '{job.group_name}' ({total} video(s)) -> {out_path} ---"))
    queue.put(("job_finished", job.id, "done", ""))
