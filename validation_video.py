"""
Produces a full annotated validation video for one input video -- every
frame overlaid with both object hitboxes, every tracked skeleton node
(colored by confidence), the sniff cone, the current exploration/bout/
climbing status per object, and a frame-accurate decimal timestamp.

This is the auditable record behind the Excel output: what's shown as
"counted" here comes from the same scoring.compute_frame_details() call
that produces the Excel bout numbers, so the two can't disagree.

One validation video is produced per input video by `main.py run`
(unless --skip-validation is passed), written to
OUTPUT_FOLDER/validation_videos/<video_stem>.validation.mp4.
"""

from pathlib import Path
import math
import subprocess
import threading

import cv2
import numpy as np

from geometry import hitbox_half_width_px, object_half_width_px
from object_picker import LABEL_COLORS, FOOTPRINT_OUTLINE_COLOR
from scoring import compute_frame_details

# Sniff-cone ray colors (BGR), drawn from the nose.
_CONE_CENTER_COLOR = (255, 255, 0)   # cyan -- the head vector (neck -> nose)
_CONE_EDGE_COLOR = (0, 255, 255)     # yellow -- +/- SNIFF_CONE_HALF_ANGLE_DEG boundary rays

# cv2.VideoWriter's mp4v/MJPG encoders don't reliably preserve this video's
# thin color-coded overlay lines at a normal bitrate; piping frames to
# ffmpeg directly with an explicit CRF avoids that.
_FFMPEG_CRF = 15

# Node marker colors (BGR)
_CONFIDENT_COLOR = (0, 220, 0)      # green -- confidence >= MIN_NODE_CONFIDENCE
_LOW_CONF_COLOR = (0, 0, 255)       # red -- present but below threshold
_EDGE_COLOR = (180, 180, 180)       # light gray connective lines (purely visual)

# Hitbox outline colors (BGR)
_HITBOX_DEFAULT_COLOR = (200, 200, 200)   # falls back to this if object name isn't in LABEL_COLORS
_HITBOX_COUNTED_COLOR = (0, 255, 0)       # green -- this object's bout is being counted this frame
_HITBOX_CLIMBING_COLOR = (255, 0, 255)    # magenta -- climbing exclusion triggered this frame

_TEXT_COLOR = (255, 255, 255)
_TEXT_BG = (0, 0, 0)
_LOW_CONFIDENCE_WARNING_COLOR = (0, 0, 255)


def _skeleton_edges(cfg):
    """A reasonable default chain of connective lines between tracked
    nodes -- purely visual, has no effect on scoring. Edges involving a
    node name not present in this model's skeleton are silently skipped
    at draw time."""
    return [
        (cfg.NODE_NOSE, cfg.NODE_NECK),
        (cfg.NODE_NECK, cfg.NODE_TORSO),
        (cfg.NODE_TORSO, cfg.NODE_TAIL_BASE),
        (cfg.NODE_NECK, cfg.NODE_LEFT_EAR),
        (cfg.NODE_NECK, cfg.NODE_RIGHT_EAR),
    ]


def _put_label(frame, text, org, scale=0.45, color=_TEXT_COLOR, thickness=1):
    """Draw text with a filled background box behind it so it stays
    legible over any part of the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 2, y - th - 2), (x + tw + 2, y + baseline + 2), _TEXT_BG, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _get_ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def export_validation_video(video_path, track, object_coords, cfg, out_path, node_radius=3):
    """Write a fully annotated copy of `video_path` to `out_path`,
    covering every frame in the track (not a sample or a time window --
    see frame_export.py for that). Returns out_path.
    """
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = _get_ffmpeg_exe()
    if ffmpeg_exe is None:
        raise RuntimeError(
            "validation_video export needs the `imageio_ffmpeg` package -- "
            "install it with `pip install imageio-ffmpeg`."
        )

    details = compute_frame_details(track, object_coords, cfg=cfg)
    half_width = int(round(hitbox_half_width_px(cfg)))
    footprint_half_width = int(round(object_half_width_px(cfg)))
    object_names = list(object_coords.keys())
    edges = _skeleton_edges(cfg)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for reading: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or track.fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_frames = track.n_frames
    if n_video_frames > 0 and n_video_frames != n_frames:
        n_frames = min(n_frames, n_video_frames)

    ffmpeg_cmd = [
        ffmpeg_exe, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-crf", str(_FFMPEG_CRF), "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Drain stderr on a background thread so a full OS pipe buffer can't
    # deadlock stdin writes.
    stderr_chunks = []

    def _drain_stderr():
        for chunk in iter(lambda: proc.stderr.read(4096), b""):
            stderr_chunks.append(chunk)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    node_names = track.node_names

    try:
        frame_idx = 0
        while frame_idx < n_frames:
            ok, frame = cap.read()
            if not ok:
                break

            # Z-order, back to front: skeleton edges, text panel, hitboxes
            # (+ labels), sniff-cone rays, node markers. Node markers are
            # drawn last so a tracked point is never obscured.

            # Skeleton connective lines.
            for a, b in edges:
                if a not in node_names or b not in node_names:
                    continue
                pa = track.xy(a)[frame_idx]
                pb = track.xy(b)[frame_idx]
                if np.isnan(pa).any() or np.isnan(pb).any():
                    continue
                cv2.line(frame, (int(round(pa[0])), int(round(pa[1]))), (int(round(pb[0])), int(round(pb[1]))), _EDGE_COLOR, 2)

            # Timestamp -- frame index and decimal seconds, so any frame
            # can be located precisely from either the video or the
            # Excel bout times.
            t = frame_idx / fps
            _put_label(frame, f"frame {frame_idx}  t={t:.3f}s", (6, 18), scale=0.45)

            # Per-object status text.
            y = 38
            counted_lab = details["counted_label"][frame_idx]
            raw_lab = details["raw_labels"][frame_idx]
            for obj_name in object_names:
                d = details["per_object"][obj_name]
                obj_i = d["obj_i"]
                if d["climbing"][frame_idx]:
                    status = "CLIMBING (excluded)"
                elif counted_lab == obj_i:
                    status = "EXPLORING (counted)"
                elif raw_lab == obj_i:
                    status = "exploring (filtered: too short / gap-merged away)"
                else:
                    status = None
                if status is not None:
                    _put_label(frame, f"{obj_name}: {status}", (6, y), scale=0.42)
                    y += 18

            if not details["valid"][frame_idx]:
                _put_label(
                    frame, "LOW-CONFIDENCE TRACKING THIS FRAME (excluded from scoring)",
                    (6, h - 10), scale=0.42, color=_LOW_CONFIDENCE_WARNING_COLOR,
                )

            # Hitboxes, colored by this frame's status for that object.
            for obj_name in object_names:
                obj_xy = object_coords[obj_name]
                cx, cy = int(round(obj_xy[0])), int(round(obj_xy[1]))
                d = details["per_object"][obj_name]
                obj_i = d["obj_i"]

                if d["climbing"][frame_idx]:
                    color = _HITBOX_CLIMBING_COLOR
                elif details["counted_label"][frame_idx] == obj_i:
                    color = _HITBOX_COUNTED_COLOR
                else:
                    color = LABEL_COLORS.get(obj_name, _HITBOX_DEFAULT_COLOR)

                cv2.rectangle(frame, (cx - half_width, cy - half_width), (cx + half_width, cy + half_width), color, 2)
                # Inner outline showing the object's real (unpadded) footprint.
                cv2.rectangle(
                    frame, (cx - footprint_half_width, cy - footprint_half_width),
                    (cx + footprint_half_width, cy + footprint_half_width), FOOTPRINT_OUTLINE_COLOR, 1,
                )
                _put_label(frame, obj_name, (cx - half_width, cy - half_width - 10), scale=0.35)

            # Sniff-cone rays: the head vector (cyan) and its
            # +/-SNIFF_CONE_HALF_ANGLE_DEG boundary rays (yellow), cast from
            # the same pulled-back apex scoring._head_cone_intersects_object
            # uses (SNIFF_RAY_ORIGIN_BACKSET_RATIO back from the nose tip
            # along -head_vec), not the nose tip itself.
            nose_xy = track.xy(cfg.NODE_NOSE)[frame_idx]
            neck_xy = track.xy(cfg.NODE_NECK)[frame_idx]
            if not np.isnan(nose_xy).any() and not np.isnan(neck_xy).any():
                head_vec = nose_xy - neck_xy
                head_norm = np.linalg.norm(head_vec)
                if head_norm > 1e-6:
                    unit = head_vec / head_norm
                    ray_len = half_width * 1.8
                    half_angle_rad = math.radians(cfg.SNIFF_CONE_HALF_ANGLE_DEG)
                    origin_xy = nose_xy - cfg.SNIFF_RAY_ORIGIN_BACKSET_RATIO * head_vec
                    start = (int(round(origin_xy[0])), int(round(origin_xy[1])))
                    for angle, color in (
                        (0.0, _CONE_CENTER_COLOR),
                        (half_angle_rad, _CONE_EDGE_COLOR),
                        (-half_angle_rad, _CONE_EDGE_COLOR),
                    ):
                        c, s = math.cos(angle), math.sin(angle)
                        rx = unit[0] * c - unit[1] * s
                        ry = unit[0] * s + unit[1] * c
                        end = (int(round(origin_xy[0] + rx * ray_len)), int(round(origin_xy[1] + ry * ray_len)))
                        cv2.line(frame, start, end, color, 1, cv2.LINE_AA)

            # Node markers, colored by confidence.
            for node_name in node_names:
                xy = track.xy(node_name)[frame_idx]
                if np.isnan(xy).any():
                    continue
                conf = track.conf(node_name)[frame_idx]
                confident = not np.isnan(conf) and conf >= cfg.MIN_NODE_CONFIDENCE
                color = _CONFIDENT_COLOR if confident else _LOW_CONF_COLOR
                cv2.circle(frame, (int(round(xy[0])), int(round(xy[1]))), node_radius, color, -1)

            try:
                proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                break  # ffmpeg died mid-stream -- fall through to the returncode check below
            frame_idx += 1
    finally:
        cap.release()
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        proc.wait()
        stderr_thread.join(timeout=5)

    if proc.returncode != 0:
        stderr = b"".join(stderr_chunks).decode(errors="replace")
        raise RuntimeError(
            f"ffmpeg exited with code {proc.returncode} while writing {out_path}:\n{stderr}"
        )

    return out_path
