# RATlab NOR

RATlab NOR is a tool designed to speed up your data analysis workflow and improve
accuracy. It uses deep learning models to convert novel object recognition task
videos into usable numerical data autonomously, freeing up your time to actually
do research. Just install SLEAP, train a model on your setup, load your videos
into the app, and you're good to go.

## Installation

**Option 1 — download the app.** Grab the prebuilt `RATlab NOR` app for
your OS (`.app` for macOS, `.exe` for Windows, `AppImage` for Linux) from
this repo's [Releases](../../releases) page, and unzip/extract it.
On macOS, you will need to click "Open Anyway" in Settings >
Privacy & Security, as the app is not officially verified. On the first
launch on all platforms, the app will ask you to point it to the folder
where it is installed (which will be called RATlab). You can rename or move
the folder, but make sure to keep its contents together, as the app
needs to be able to reference its `models` directory in order to run
inference.

SLEAP itself (`sleap-nn`/`sleap`) still needs to be installed and on
`PATH` on whatever machine runs the app. To install it, follow the 
instructions at <https://docs.sleap.ai/latest/installation/> to install
the SLEAP GUI for model training, then run 

```uv tool install sleap-nn --torch-backend auto```

to install the neural network backend used by RATlab NOR.

**Option 2 — build it yourself.** If there's no prebuilt release for your
OS yet, or you've changed the source and want a fresh build, see
`packaging/README.md`.

**Option 3 — run from source (for development).**

```bash
pip install -r requirements.txt
python gui/app.py
```

The app calls `sleap-nn track` directly, falling back to `sleap track`
then legacy `sleap-track` if `sleap-nn` isn't on PATH.

## The app

Queue up multiple video folders as separate jobs, confirm object
positions per job, then run the whole queue unattended with live
progress and a log — no terminal needed day to day. Each job gets its
own output folder and object-coordinate/prediction cache, so unrelated
video sets never collide.

The command-line workflow below still works standalone and drives the
same underlying modules — useful for scripting or a single one-off run.

## Command-line workflow

### 1. Install

```bash
pip install -r requirements.txt
```

The app calls `sleap-nn track` directly, falling back to `sleap track`
then legacy `sleap-track` if `sleap-nn` isn't on PATH. Verify your setup
works with `sleap-nn track --help`.

Before running inference, each video is wrapped in a minimal single-video
SLEAP Labels project (`predictions/_video_wrappers/`) with
`grayscale=True` forced explicitly, since `sleap_io`'s auto-detection can
misjudge lossy-compressed footage and silently feed a grayscale-trained
model 3-channel input. If you retrain on color video, that forced
`grayscale=True` in `sleap_inference.py` would need to change too.

Validation video export needs `imageio_ffmpeg` (normally already
installed transitively via `sleap_io`; if missing, `pip install
imageio-ffmpeg`).

### 2. Configure (`config.py`)

- `VIDEO_FOLDER` — folder to scan recursively for `.mp4` files.
- `MODEL_PATHS` — path to your trained model folder (containing
  `training_config.json` + a checkpoint). Single-instance model, so one
  path is all you need.
- Skeleton node names (`NODE_NOSE`, `NODE_NECK`, etc.) — must match the
  node names your model was trained with.
- `FILENAME_PATTERN` — regex used to pull the rat ID out of each video's
  filename so results group correctly. Defaults to matching filenames
  like `359 novel a.mp4` → rat ID `359`.
- Scoring thresholds — see `config.py`. Nose inside the object's square
  hitbox (real footprint `OBJECT_SIZE_CM` + padding `OBJECT_PADDING_CM`,
  converted to pixels via `PX_PER_CM`) AND head oriented toward it,
  excluding climbing/sitting frames. "Oriented toward it" is a ray from
  the nose in the head direction, or within `SNIFF_CONE_HALF_ANGLE_DEG`
  of it, crossing the object's footprint — see
  `scoring._head_cone_intersects_object()`. The same hitbox is what's
  drawn/dragged in the object picker, so what you see there is exactly
  what's scored.
- Grayscale: your model was trained on grayscale frames; SLEAP converts
  color mp4 frames automatically.

### 3. Set object positions

```bash
python main.py setup
```

Opens one persistent window walking through every not-yet-configured
video, with two square hitboxes overlaid on the first frame, labeled
"novel" and "original". New videos start pre-populated with whichever
positions were last confirmed, so if nothing moved you just click
**Forward**.

Controls live in a sidebar to the right of the frame:

- **Select** — highlight the next hitbox (white outline) for precise
  arrow-key nudging; clicking/dragging a hitbox also selects it
- **Swap** — swap novel/original
- **Undo** — undo the last move/swap
- **◀ Back** — go to the previous video in this session, keeping edits
- **ⓘ Info** — toggle a popup listing every control and its shortcut
- **▶ Forward** — confirm this video's positions and advance

Keyboard shortcuts (also in the info popup): **drag** a hitbox to move
it, **arrow keys** nudge it by 1px, **Tab** selects the next hitbox, **S**
swaps, **U** undoes, **R** resets this video, **Enter** confirms/advances,
**Esc** skips this video without saving it.

Positions are saved to `object_coords.json` as each video is confirmed.
Re-running `setup` only prompts for videos added since; delete
`object_coords.json` to start over and re-review everything.

### 4. Run the batch (fully unattended)

```bash
python main.py run
```

For every video with object coordinates set, this runs SLEAP tracking
(cached in `predictions/`), scores exploration bouts, writes a single
workbook to `output/NOR_results.xlsx`, and writes a full annotated
validation video to `output/validation_videos/` (see below).

Pass `--skip-validation` to skip the validation-video step (slower, since
it re-encodes every frame): `python main.py run --skip-validation`. Leave
it off for a real run meant to back a publication.

Or do both steps in one call: `python main.py all`.

## Output format

**`NOR_results.xlsx`** — one sheet per rat. Within each sheet, one block
per video/session for that rat, with two side-by-side tables (one per
object) listing every bout's start time, stop time, and duration,
followed by "Total bouts" and "Total time (s)" summary rows for each
object.

**`validation_videos/<video>.validation.mp4`** — the auditable record
behind those numbers: every frame of the source video, annotated with
both object hitboxes, every tracked skeleton node (green = confident,
red = below `MIN_NODE_CONFIDENCE`), the sniff cone (cyan = head vector,
yellow = its `±SNIFF_CONE_HALF_ANGLE_DEG` boundary rays), a frame-accurate
decimal timestamp, and a live status readout of whether each object's
bout is currently being counted, filtered out (too short / gap-merged),
or excluded as climbing. What's shown as "counted" here is computed by
the exact same code path that produces the Excel bout numbers
(`scoring.compute_frame_details`), so the video and the spreadsheet can
never disagree.

## Files

- `config.py` — all settings (paths, model, skeleton, thresholds).
- `geometry.py` — the object hitbox-size math, shared by the picker and
  the scorer.
- `object_picker.py` — the interactive novel/original positioning step.
- `sleap_inference.py` — batch wrapper around SLEAP's tracking CLI.
- `pose_utils.py` — loads a rat's track out of a SLEAP `.slp` predictions file.
- `pose_filters.py` — nulls out isolated single-frame nose/tail-base
  tracking swaps before scoring.
- `scoring.py` — hitbox + orientation exploration criteria, climbing
  exclusion, bout merging/filtering, and `compute_frame_details()` (the
  shared per-frame detail both the Excel output and the validation
  videos are built from). No SLEAP dependency.
- `excel_writer.py` — builds the output workbook.
- `validation_video.py` — writes the annotated per-video validation mp4s.
- `main.py` — ties it all together (`setup` / `run` / `all`).
- `reset.py` — clears `predictions/`/`output/` (and optionally
  `object_coords.json`) to re-run from scratch; see `python reset.py --help`.
- `debug_score.py` — fast iteration on scoring/tracking without
  re-running inference. See `python debug_score.py --help`.
- `frame_export.py` — saves annotated JPEGs for a specific time window
  (used by `debug_score.py --frames`).
- `gui/` — the desktop app (see "GUI" above); `packaging/` — scripts to
  build it into a double-clickable app per OS.

Standalone tools, each documented in its own module docstring:
`benchmark.py` (score against human-coded ground truth), `video_crop.py`
(crop footage to a model's expected resolution), `build_negative_frames.py`
(generate "no rat" training frames), `review_flags.py` (heuristics behind
the GUI's Review screen).

## Notes / assumptions

- One rat is assumed per video (single-instance tracking); if a video has
  multiple instances, `pose_utils.load_track_from_slp` takes the first
  one by default (`track_index=0`).
- `object_coords.json` is plain JSON keyed by each video's path relative
  to `VIDEO_FOLDER`, each entry `{"novel": [x, y], "original": [x, y]}` —
  you can hand-edit or script-generate it instead of using `setup`.
- The whole `RATlab` folder can be moved, renamed, or copied to another
  machine, as long as `nor_classifier/` and `models/` stay in the same
  position relative to each other.
