"""
Export annotated frames from a video for a given time window, with the
tracked nose/neck points and object hitboxes drawn on top -- for visually
checking what the model predicted during a specific stretch.

Usage (via debug_score.py):
    python debug_score.py "378 novel b" --frames 32 40

Saves every other frame between 32s and 40s to
output/frames_<video>_32.0-40.0s/, annotated with the predicted nose
(green if trusted, red if below MIN_NODE_CONFIDENCE), neck (cyan), and
both object hitboxes.
"""

from pathlib import Path

import cv2

from geometry import hitbox_half_width_px
from object_picker import LABEL_COLORS

NOSE_OK_COLOR = (0, 220, 0)     # green (BGR) -- confidence >= threshold
NOSE_LOW_COLOR = (0, 0, 255)    # red -- below threshold or missing
NECK_COLOR = (255, 220, 0)      # cyan-ish


def export_annotated_frames(video_path, track, object_coords, cfg, start_s, end_s, out_dir, every_n=2):
    """Save annotated frames covering [start_s, end_s) to out_dir.

    Returns the list of saved file paths.
    """
    if end_s <= start_s:
        raise ValueError(f"end_s ({end_s}) must be greater than start_s ({start_s})")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    half_width = hitbox_half_width_px(cfg)
    fps = track.fps
    start_frame = max(0, int(round(start_s * fps)))
    end_frame = min(track.n_frames, int(round(end_s * fps)))

    nose = track.xy(cfg.NODE_NOSE)
    neck = track.xy(cfg.NODE_NECK)
    nose_conf = track.conf(cfg.NODE_NOSE)
    neck_conf = track.conf(cfg.NODE_NECK)

    cap = cv2.VideoCapture(str(video_path))
    saved = []
    try:
        for frame_idx in range(start_frame, end_frame, max(1, every_n)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            display = frame.copy()
            t = frame_idx / fps

            for obj_name, obj_xy in object_coords.items():
                ox, oy = int(round(obj_xy[0])), int(round(obj_xy[1]))
                hw = int(round(half_width))
                color = LABEL_COLORS.get(obj_name, (200, 200, 200))
                cv2.rectangle(display, (ox - hw, oy - hw), (ox + hw, oy + hw), color, 1)

            nx, ny = nose[frame_idx]
            nc = nose_conf[frame_idx]
            if not (nx != nx or ny != ny):  # not NaN
                color = NOSE_OK_COLOR if (nc == nc and nc >= cfg.MIN_NODE_CONFIDENCE) else NOSE_LOW_COLOR
                cv2.circle(display, (int(round(nx)), int(round(ny))), 3, color, -1)
                conf_str = f"{nc:.2f}" if nc == nc else "?"
                cv2.putText(display, f"nose {conf_str}", (int(nx) + 6, int(ny) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

            kx, ky = neck[frame_idx]
            if not (kx != kx or ky != ky):
                cv2.circle(display, (int(round(kx)), int(round(ky))), 3, NECK_COLOR, -1)

            cv2.putText(display, f"frame {frame_idx}  t={t:.2f}s", (8, display.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            out_path = out_dir / f"frame_{frame_idx:06d}_t{t:.2f}s.jpg"
            cv2.imwrite(str(out_path), display)
            saved.append(out_path)
    finally:
        cap.release()

    return saved
