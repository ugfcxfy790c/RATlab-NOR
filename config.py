"""
Configuration for the NOR (Novel Object Recognition) classifier app.

Edit the values below to match your setup before running main.py.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Anchored to this file's location so the RATlab folder can be moved freely.
NOR_CLASSIFIER_DIR = Path(__file__).resolve().parent
RATLAB_DIR = NOR_CLASSIFIER_DIR.parent

# Folder to scan (recursively) for .mp4 videos to process.
VIDEO_FOLDER = NOR_CLASSIFIER_DIR / "input"

# Where SLEAP prediction files (.slp) are cached, one per video.
PREDICTIONS_FOLDER = NOR_CLASSIFIER_DIR / "predictions"

# Where the final Excel workbook(s) are written.
OUTPUT_FOLDER = NOR_CLASSIFIER_DIR / "output"

# Cache of manually-clicked object coordinates, keyed by video filename.
OBJECT_COORDS_FILE = NOR_CLASSIFIER_DIR / "object_coords.json"

# Trained SLEAP model directory (contains training_config.yaml + best.ckpt).
MODEL_PATHS = [
    RATLAB_DIR / "models" / "NOR_v_2_0_1_negative_TE.single_instance.n=743",
]

# Frame size to force during inference. Leave as None unless it exactly
# matches the model's own training_config max_height/max_width.
INFERENCE_MAX_HEIGHT = None
INFERENCE_MAX_WIDTH = None

# Default crop target for video_crop.py / the GUI's "Video" menu --
# matches every current model's training_config.yaml preprocessing
# (max_height/max_width), so videos recorded at some other resolution
# can be cropped down to what the model actually expects before
# inference, rather than relying on sleap-nn's own implicit
# padding/resizing for oversized frames (which would silently distort
# the PX_PER_CM calibration everything else here depends on). Editable
# per-crop in the GUI dialog -- these are just the pre-filled defaults.
CROP_TARGET_WIDTH = 294
CROP_TARGET_HEIGHT = 292

# ---------------------------------------------------------------------------
# Skeleton node names (must match the names used when the model was trained)
# ---------------------------------------------------------------------------

NODE_NOSE = "nose"
NODE_LEFT_EAR = "left_ear"
NODE_RIGHT_EAR = "right_ear"
NODE_NECK = "neck"
NODE_TORSO = "torso"
NODE_TAIL_BASE = "tail base"

# ---------------------------------------------------------------------------
# Exploration-scoring criteria (standard NOR criteria: nose within a set
# distance of the object AND head oriented toward it; climbing/sitting on
# the object is excluded from "exploration").
# ---------------------------------------------------------------------------

# Pixels per real-world cm, for converting the *_CM settings below to px.
PX_PER_CM = 5.7

# Real object footprint size, and padding added to it to form the square
# "hitbox" used to test nose-to-object proximity.
OBJECT_SIZE_CM = 5.0
OBJECT_PADDING_CM = 2.0

# Grows the object's real-footprint square (the ray/cone-intersection
# target, and the inner outline drawn in the picker/validation videos) by
# this many px on each side, independent of OBJECT_SIZE_CM.
OBJECT_FOOTPRINT_GROW_PX = 2.0

# "Oriented toward the object": a ray from the nose in the head direction
# (neck -> nose), or within this many degrees of it, must cross the
# object's footprint. See scoring._head_cone_intersects_object().
SNIFF_CONE_HALF_ANGLE_DEG = 25.0

# Pulls the cone's apex back from the nose tip by this fraction of
# |head_vec| (neck -> nose distance), so a touch from the side of the
# snout (not just the tip) can still fall within the cone. 0 = apex at the
# nose tip exactly.
SNIFF_RAY_ORIGIN_BACKSET_RATIO = 0.25

# If the torso is closer than this to the object, the rat is treated as
# climbing/sitting on it rather than exploring it.
CLIMBING_TORSO_DISTANCE_PX = (OBJECT_SIZE_CM / 2.0) * PX_PER_CM

# Minimum SLEAP confidence required to trust a keypoint.
MIN_NODE_CONFIDENCE = 0.5

# ---------------------------------------------------------------------------
# Nose/tail-swap filter (pose_filters.filter_nose_tail_swaps): nulls out
# isolated frames where the nose prediction jumps to roughly where
# tail_base/torso actually is.
# ---------------------------------------------------------------------------

# Flags a frame if nose-neck distance exceeds this fraction of that frame's
# own neck-to-tail_base distance.
NOSE_SWAP_BODY_LENGTH_RATIO = 0.8

# Fallback cue when tail_base isn't confidently tracked: flags a frame if
# nose-neck distance exceeds this multiple of the video's median.
NOSE_SWAP_DISTANCE_RATIO = 2.5

# Only null out flagged runs up to this many consecutive frames.
NOSE_SWAP_ISOLATED_MAX_RUN_FRAMES = 1

# Bouts closer together than this (seconds) are merged into one -- for a
# *confidently tracked* gap, i.e. the tracker actually saw the animal not
# exploring, real evidence of a break. (See CONFIDENCE_AWARE_MERGE_ENABLED
# below for the separate, looser threshold used when the gap is instead
# explained by low tracking confidence.) Since low-confidence gaps no
# longer need to hide inside this one, it can be set tighter than before
# without swallowing genuine confidence dropouts as "still exploring" --
# but as with any scoring threshold, verify against your own footage
# rather than eyeballing it.
MERGE_GAP_S = 1.0

# Bouts shorter than this (seconds) are discarded as noise.
MIN_BOUT_DURATION_S = 0.3

# ---------------------------------------------------------------------------
# Confidence-aware gap merging (scoring.py's labels_to_bouts() /
# _merge_bout_runs()) -- a gap between two same-object bouts is treated
# differently depending on *why* there's a gap:
#
#   - the tracker lost the nose/neck for (most of) the gap: no real
#     evidence the animal left -- usually self-occlusion from close-up
#     sniffing, not a genuine break. Merged even past MERGE_GAP_S, up to
#     LOW_CONFIDENCE_GAP_MERGE_S.
#   - the gap was confidently tracked as "not exploring" the whole time:
#     real evidence of a break. Only merged if shorter than MERGE_GAP_S.
#
# Disable to fall back to the old behavior: MERGE_GAP_S applied flatly to
# every gap regardless of confidence.
# ---------------------------------------------------------------------------
CONFIDENCE_AWARE_MERGE_ENABLED = True

# Confidence gaps are merged even if longer than MERGE_GAP_S, up to this
# cap (seconds). Past this, an unusually long dropout is worth a person's
# judgment rather than an assumption -- see REVIEW_MAX_AUTO_MERGES below
# and gui/batch_review_dialog.py's scrub-bar markers for how these show
# up for review.
LOW_CONFIDENCE_GAP_MERGE_S = 3.0

# Fraction of a gap's frames that must be untracked/low-confidence for it
# to count as a "confidence gap" above rather than a "confident break".
# Not exposed in the Settings dialog -- an implementation nuance, not a
# scoring criterion worth hand-tuning per lab/rig the way the gap
# durations above are.
LOW_CONFIDENCE_GAP_FRACTION = 0.9

# ---------------------------------------------------------------------------
# Post-batch review flagging (review_flags.py) -- heuristics for "this
# video's tracking might not be trustworthy, take a manual look",
# surfaced in the GUI's Review screen. Independent of the scoring
# thresholds above -- these only decide what gets flagged for a human to
# double-check, not anything about the exploration-bout numbers.
# ---------------------------------------------------------------------------

# Below this percentage of frames with trustworthy (jointly confident)
# nose+neck tracking, flag the whole video for review.
REVIEW_MIN_VALID_FRAMES_PCT = 70.0

# Flag an object if the *nose's own* tracking confidence (not the neck's --
# see review_flags.py's docstring on why conflating the two over-flags
# ordinary close-up sniffing) drops below MIN_NODE_CONFIDENCE for this
# many *continuous* seconds while positioned in that object's hitbox. A
# handful of scattered single-frame blips adding up to this much across
# the whole video doesn't count -- only one sustained dropout does.
REVIEW_NOSE_DROPOUT_NEAR_OBJECT_S = 1.0

# Flag a video for review if more than this many bouts got automatically
# merged across a low-confidence gap (see CONFIDENCE_AWARE_MERGE_ENABLED
# above) -- an unusual number of these in one video is worth a person's
# spot-check even though each individual merge is, on its own, the
# scorer's best guess.
REVIEW_MAX_AUTO_MERGES = 2

# ---------------------------------------------------------------------------
# Rat ID / session parsing
# ---------------------------------------------------------------------------

# Pulls the rat ID and session out of a video filename, e.g.
# "359 novel a.mp4" -> rat ID "359", session "a".
FILENAME_PATTERN = r"^(?P<rat_id>\S+)\s+(?P<phase>.+?)\s+(?P<session>[ab])\b"
