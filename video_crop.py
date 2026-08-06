"""
Crops videos down to a fixed target size -- for footage recorded at some
resolution other than what a model was trained on (see config.py's
CROP_TARGET_WIDTH/HEIGHT). Deliberately explicit/visual (via the GUI's
crop dialog) rather than letting sleap-nn's own --max_height/--max_width
inference-time preprocessing pad or resize oversized frames implicitly,
since a resize (as opposed to a pixel-for-pixel crop) would distort the
PX_PER_CM calibration scoring.py and the object-picker both depend on.

No GUI/Qt dependency -- usable from the CLI, tests, or the GUI's worker
process alike. Pipes raw frames to ffmpeg directly (like
validation_video.py does), rather than cv2.VideoWriter, for the same
reason noted there: cv2.VideoWriter's built-in encoders were confirmed
elsewhere in this project to visibly degrade quality across a full
sequence.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path

import cv2

# Quality knob for the crop's re-encode. Cropped output feeds back into
# inference (not just human review, unlike validation_video.py's output),
# so bias toward less lossy than that module's default.
_FFMPEG_CRF = 12


def _get_ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def find_videos(folder) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.mp4"))


def probe_resolution(video_path) -> tuple[int, int] | None:
    """Returns (width, height), or None if the video can't be opened."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        return None
    return w, h


def scan_resolutions(folder) -> dict[tuple[int, int], list[Path]]:
    """Groups every video under `folder` by (width, height) -- the usual
    first question before cropping: which videos already match the
    model's expected size, and which don't."""
    groups: dict[tuple[int, int], list[Path]] = {}
    for video_path in find_videos(folder):
        res = probe_resolution(video_path)
        if res is None:
            res = (-1, -1)  # unreadable -- grouped together rather than dropped silently
        groups.setdefault(res, []).append(video_path)
    return groups


class CropRegionError(ValueError):
    pass


def crop_video(video_path, out_path, x: int, y: int, width: int, height: int, log=print) -> Path:
    """Crop a single video to the `width`x`height` window starting at
    (x, y), writing to out_path. Raises CropRegionError if that window
    doesn't fit inside the source frame -- never silently clamps, since a
    silently-shifted crop would put objects/keypoints at the wrong pixel
    coordinates without any visible sign something's off.
    """
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    res = probe_resolution(video_path)
    if res is None:
        raise RuntimeError(f"Could not open video for reading: {video_path}")
    src_w, src_h = res

    if x < 0 or y < 0 or x + width > src_w or y + height > src_h:
        raise CropRegionError(
            f"Crop window ({width}x{height} at ({x},{y})) doesn't fit inside "
            f"{video_path.name}'s {src_w}x{src_h} frame."
        )

    ffmpeg_exe = _get_ffmpeg_exe()
    if ffmpeg_exe is None:
        raise RuntimeError(
            "video_crop needs the `imageio_ffmpeg` package -- install it with "
            "`pip install imageio-ffmpeg`."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for reading: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ffmpeg_cmd = [
        ffmpeg_exe, "-y", "-loglevel", "error",
        # -s must match the CROPPED frame size actually being piped below,
        # not the source video's size -- ffmpeg trusts this blindly for a
        # raw byte stream, it can't infer it from the data itself.
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
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

    try:
        frame_idx = 0
        while n_frames <= 0 or frame_idx < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            cropped = frame[y:y + height, x:x + width]
            try:
                proc.stdin.write(cropped.tobytes())
            except BrokenPipeError:
                break
            frame_idx += 1
    finally:
        cap.release()
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        proc.wait()
        stderr_thread.join(timeout=5)

    if proc.returncode != 0:
        stderr = b"".join(stderr_chunks).decode(errors="replace")
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode} while writing {out_path}:\n{stderr}")

    log(f"Cropped {video_path.name} -> {out_path}")
    return out_path


def crop_folder(
    input_folder, output_folder, x: int, y: int, width: int, height: int,
    log=print, on_progress=None,
) -> list[Path]:
    """Crops every video under input_folder into the equivalent relative
    path under output_folder. A video already exactly the target size
    (at (0,0)) is copied through as-is rather than re-encoded, to avoid
    a pointless quality-losing round trip through the codec.

    on_progress(index, total), if given, is called before each video.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    videos = find_videos(input_folder)
    total = len(videos)
    written = []

    for i, video_path in enumerate(videos, start=1):
        if on_progress:
            on_progress(i, total)

        rel = video_path.relative_to(input_folder)
        out_path = output_folder / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        res = probe_resolution(video_path)
        if res == (width, height) and x == 0 and y == 0:
            shutil.copy2(video_path, out_path)
            log(f"[{i}/{total}] Already {width}x{height} -- copied {video_path.name} unchanged")
        else:
            log(f"[{i}/{total}] Cropping {video_path.name}...")
            crop_video(video_path, out_path, x, y, width, height, log=log)

        written.append(out_path)

    return written


# --- per-video crop positions (for CropSetupDialog -- the arena isn't ---
# --- always framed the same way in every video, unlike crop_folder()'s ---
# --- single position applied uniformly) ------------------------------------

def load_positions(positions_path) -> dict:
    positions_path = Path(positions_path)
    if positions_path.exists():
        with open(positions_path) as f:
            return json.load(f)
    return {}


def save_positions(positions_path, positions: dict) -> None:
    positions_path = Path(positions_path)
    positions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(positions_path, "w") as f:
        json.dump(positions, f, indent=2)


def crop_videos_with_positions(
    positions: list[tuple],  # [(video_path, x, y), ...]
    input_folder, output_folder, width: int, height: int,
    log=print, on_progress=None,
) -> list[Path]:
    """Like crop_folder(), but each video gets its own (x, y) -- for
    sessions where the camera/arena framing shifted between recordings
    rather than staying fixed for the whole folder."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    total = len(positions)
    written = []

    for i, (video_path, x, y) in enumerate(positions, start=1):
        video_path = Path(video_path)
        if on_progress:
            on_progress(i, total)

        rel = video_path.relative_to(input_folder)
        out_path = output_folder / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        res = probe_resolution(video_path)
        if res == (width, height) and x == 0 and y == 0:
            shutil.copy2(video_path, out_path)
            log(f"[{i}/{total}] Already {width}x{height} -- copied {video_path.name} unchanged")
        else:
            log(f"[{i}/{total}] Cropping {video_path.name} at ({x},{y})...")
            crop_video(video_path, out_path, x, y, width, height, log=log)

        written.append(out_path)

    return written
