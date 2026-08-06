"""
Build a "negative frames" video: real frames sampled from one or more
input videos with the rat digitally erased -- composited out using that
video's own static background -- for use as negative training examples.

Locates the rat via background subtraction (temporal median of sampled
frames as the rat-free background, then contours in the diff as the rat's
silhouette), so it doesn't need a SLEAP prediction to run.

Usage:
    python build_negative_frames.py "360 novel a" "360 novel b"
    python build_negative_frames.py "360 novel a" "360 novel b" \\
        --max-per-video 25 --hold-s 0.6 -o output/negative_frames.mp4

Each matched video contributes up to --max-per-video frames, sampled
evenly across the whole video. For each sampled frame, the detected rat
silhouette (dilated, edges feathered) is replaced with that video's own
background pixels at the same location. Selected frames are concatenated
in the order given, each held for --hold-s seconds in the output video.
"""

import argparse
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

import config

# How many frames to sample (evenly spaced) when building the temporal-
# median background for a video. More samples = cleaner background (less
# chance of the rat's own motion leaving a ghost), but slower.
N_BACKGROUND_SAMPLES = 150

# Frame-diff-from-background threshold (0-255) and morphology used to
# isolate the rat's silhouette. Tuned for this project's red-tinted
# lighting, where the rat's tail can be only a few grayscale levels
# different from the background -- lower this if the tail is left
# un-erased, raise it if lighting noise gets picked up as false positives.
DIFF_THRESHOLD = 12
MIN_CONTOUR_AREA_PX = 60
# The tail often shows up as its own small, disconnected blob in the diff
# mask -- any contour at or above this (smaller) area is unioned into the
# mask too, not just the largest, so a thin tail isn't left un-erased.
MIN_SECONDARY_CONTOUR_AREA_PX = 35
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Bridges the gap between a faint tail segment and the main body blob
# before contour extraction, so they merge into one contour.
_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

# Grown past the detected silhouette and Gaussian-blurred, so the erased
# region covers anti-aliased/motion-blurred edges and blends smoothly.
MASK_DILATE_PX = 10
MASK_FEATHER_KERNEL = (25, 25)

_FFMPEG_CRF = 18


def _get_ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def _find_matching_video(query):
    videos = sorted(Path(config.VIDEO_FOLDER).rglob("*.mp4"))
    matches = [v for v in videos if query.lower() in v.name.lower()]
    if not matches:
        print(f"No video under {config.VIDEO_FOLDER} matches '{query}'.")
        sys.exit(1)
    if len(matches) > 1:
        print(f"'{query}' matches multiple videos, be more specific:")
        for v in matches:
            print(f"  {v.name}")
        sys.exit(1)
    return matches[0]


def _build_background(cap, n_frames, n_samples=N_BACKGROUND_SAMPLES):
    """Temporal-median background image -- the rat moves between sampled
    frames, so at each pixel the median value is overwhelmingly the
    static arena/objects, not the rat. Returns (background_bgr,
    background_gray)."""
    sample_idxs = np.linspace(0, max(n_frames - 1, 0), num=min(n_samples, n_frames), dtype=int)
    color_samples = []
    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        color_samples.append(frame)
    if not color_samples:
        raise RuntimeError("Could not read any frames to build a background image.")
    stack = np.stack(color_samples, axis=0)
    background_bgr = np.median(stack, axis=0).astype(np.uint8)
    background_gray = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY)
    return background_bgr, background_gray


def _rat_mask(frame_gray, background_gray):
    """Binary mask covering the rat's full silhouette in this frame
    (body plus any disconnected low-contrast bits like a faint tail tip,
    dilated for a safety margin), or None if nothing significant enough
    to be the rat was found (e.g. the rat is very still and has
    partially merged into the background)."""
    diff = cv2.absdiff(frame_gray, background_gray)
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_area = max(cv2.contourArea(c) for c in contours)
    if largest_area < MIN_CONTOUR_AREA_PX:
        return None
    # Union every contour past the secondary threshold, not just the
    # largest, so a disconnected tail segment isn't dropped.
    keep = [c for c in contours if cv2.contourArea(c) >= MIN_SECONDARY_CONTOUR_AREA_PX]
    rat_only = np.zeros_like(mask)
    cv2.drawContours(rat_only, keep, -1, 255, thickness=cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_DILATE_PX * 2 + 1,) * 2)
    return cv2.dilate(rat_only, kernel)


def _erase_rat(frame_bgr, background_bgr, mask):
    """Replace the masked (rat) region with this video's own background
    pixels at the same location, feathering the mask edges so the
    replacement blends in rather than leaving a hard-edged cutout."""
    alpha = cv2.GaussianBlur(mask, MASK_FEATHER_KERNEL, 0).astype(np.float32) / 255.0
    alpha = alpha[:, :, None]
    composited = frame_bgr.astype(np.float32) * (1 - alpha) + background_bgr.astype(np.float32) * alpha
    return composited.astype(np.uint8)


def select_negative_frames(video_path, cfg=config, max_frames=20, min_gap_s=2.0):
    """Return a list of (frame_bgr, frame_idx) tuples, sampled across the
    whole video at least min_gap_s apart, with the rat digitally erased
    from each (composited out using that video's own background). The
    rat's on-screen position no longer matters for selection since it's
    removed either way -- frames are just spread across the timeline for
    lighting/background variety."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Building background from {video_path.name} ({n_frames} frames @ {fps:.2f} fps)...")
    background_bgr, background_gray = _build_background(cap, n_frames)

    sample_idxs = np.linspace(0, max(n_frames - 1, 0), num=min(max_frames, n_frames), dtype=int)

    selected = []
    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = _rat_mask(gray, background_gray)
        erased = _erase_rat(frame, background_bgr, mask) if mask is not None else frame
        selected.append((erased, int(idx)))

    cap.release()
    print(f"  Produced {len(selected)} rat-erased frame(s), spread across the video.")
    return selected


def write_negative_video(all_selected, out_path, output_fps=10, hold_s=0.75):
    """all_selected: list of (video_name, [(frame_bgr, frame_idx), ...])."""
    if not all_selected or not any(frames for _, frames in all_selected):
        raise RuntimeError("No qualifying frames were found in any input video.")

    ffmpeg_exe = _get_ffmpeg_exe()
    if ffmpeg_exe is None:
        raise RuntimeError("Needs the `imageio_ffmpeg` package (bundles its own ffmpeg binary).")

    # Letterbox/crop every frame to a common size (centered, not stretched)
    # in case source videos have different resolutions.
    first_frame = next(f for _, frames in all_selected for f, _ in frames)
    out_h, out_w = first_frame.shape[:2]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hold_frames = max(1, round(hold_s * output_fps))

    ffmpeg_cmd = [
        ffmpeg_exe, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", str(output_fps),
        "-i", "-",
        "-c:v", "libx264", "-crf", str(_FFMPEG_CRF), "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr_chunks = []

    def _drain_stderr():
        for chunk in iter(lambda: proc.stderr.read(4096), b""):
            stderr_chunks.append(chunk)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    def _fit(frame):
        h, w = frame.shape[:2]
        if (h, w) == (out_h, out_w):
            return frame
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        y0 = max(0, (out_h - h) // 2)
        x0 = max(0, (out_w - w) // 2)
        ch, cw = min(h, out_h), min(w, out_w)
        canvas[y0:y0 + ch, x0:x0 + cw] = frame[:ch, :cw]
        return canvas

    # Source video/frame-index tracking is kept out of the pixels (no
    # burned-in text) and written to a sidecar manifest instead, so a
    # classifier can't learn a shortcut label instead of rat absence.
    manifest_lines = ["output_frame_start\toutput_frame_end\tsource_video\tsource_frame_idx"]

    total_written = 0
    try:
        for video_name, frames in all_selected:
            for frame, frame_idx in frames:
                fitted = _fit(frame)
                manifest_lines.append(
                    f"{total_written}\t{total_written + hold_frames - 1}\t{video_name}\t{frame_idx}"
                )
                for _ in range(hold_frames):
                    proc.stdin.write(fitted.tobytes())
                    total_written += 1
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        proc.wait()
        stderr_thread.join(timeout=5)

    if proc.returncode != 0:
        stderr = b"".join(stderr_chunks).decode(errors="replace")
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}:\n{stderr}")

    manifest_path = out_path.with_suffix(".manifest.tsv")
    manifest_path.write_text("\n".join(manifest_lines) + "\n")

    print(f"Wrote {out_path} ({total_written} output frames @ {output_fps} fps, "
          f"~{total_written / output_fps:.1f}s)")
    print(f"Wrote {manifest_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queries", nargs="+", help="substring(s) matching video filename(s) in VIDEO_FOLDER")
    parser.add_argument("--max-per-video", type=int, default=20, help="number of rat-erased frames to produce per video")
    parser.add_argument("--min-gap-s", type=float, default=2.0, help="unused now that frames are erased rather than avoided, kept for CLI compatibility")
    parser.add_argument("--hold-s", type=float, default=0.75, help="how long each selected frame is held in the output video")
    parser.add_argument("--output-fps", type=int, default=10, help="frame rate of the output video")
    parser.add_argument("-o", "--out", type=Path, default=None, help="output video path (default: output/negative_frames.mp4)")
    args = parser.parse_args()

    all_selected = []
    for query in args.queries:
        video_path = _find_matching_video(query)
        print(f"{video_path.name}:")
        frames = select_negative_frames(video_path, cfg=config, max_frames=args.max_per_video)
        all_selected.append((video_path.stem, frames))

    out_path = args.out or (Path(config.OUTPUT_FOLDER) / "negative_frames.mp4")
    write_negative_video(all_selected, out_path, output_fps=args.output_fps, hold_s=args.hold_s)


if __name__ == "__main__":
    main()
