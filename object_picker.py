"""
Object hitbox state + persistence, shared by every "set up objects"
entry point in this codebase (GUI's ObjectSetupDialog and, via main.py,
the CLI `setup` command -- both actually run the Qt dialog defined in
gui/object_setup_dialog.py; see run_object_setup() there).

This module deliberately has no window/rendering code of its own
anymore -- it used to also contain a standalone cv2.imshow()-based
picker (setup_missing_videos() and its sidebar drawing/mouse-callback
code), which was superseded once ObjectSetupDialog was built on PySide6
and became the only picker anyone actually used. Keeping two pickers
around risked them drifting apart (indeed the auto-select-a-hitbox-on
-video-advance behavior only ever made it into the Qt one) for no
benefit, so the cv2 version was deleted rather than kept in sync.

What's still here is the state machine (_PickerState) and on-disk
coordinate format (load/save_object_coords, video_key) that both the
Qt dialog and other modules (validation_video.py, frame_export.py,
benchmark.py, debug_score.py) depend on.
"""

import json
from pathlib import Path

OBJECT_LABELS = ("novel", "original")
LABEL_COLORS = {"novel": (0, 140, 255), "original": (255, 140, 0)}  # BGR, for cv2-based rendering (validation_video.py, frame_export.py)
FOOTPRINT_OUTLINE_COLOR = (210, 210, 210)  # light gray -- the object's real (unpadded) footprint


def load_object_coords(coords_path):
    coords_path = Path(coords_path)
    if coords_path.exists():
        with open(coords_path) as f:
            return json.load(f)
    return {}


def save_object_coords(coords_path, coords):
    coords_path = Path(coords_path)
    coords_path.parent.mkdir(parents=True, exist_ok=True)
    with open(coords_path, "w") as f:
        json.dump(coords, f, indent=2)


def swap_labels(entry):
    """Swap the novel/original values in one video's coords entry
    (`{"novel": [x, y], "original": [x, y]}`), in place. Used to fix
    a video -- or, via gui/main_window.py's "Swap Novel/Original" job
    action, every video in a job at once -- where the two objects got
    mislabeled during Set Up Objects, without needing to redo pose
    inference or re-click through the picker. Mirrors _PickerState.swap()
    below, which does the same thing for the live interactive session;
    this is the equivalent for coordinates already saved to disk."""
    a, b = OBJECT_LABELS
    entry[a], entry[b] = entry.get(b), entry.get(a)
    return entry


def video_key(video_path, video_folder):
    """Stable identifier for a video, expressed relative to VIDEO_FOLDER
    rather than as an absolute path -- so object_coords.json stays valid
    even if the whole project folder is moved, renamed, or copied to
    another machine."""
    return str(Path(video_path).resolve().relative_to(Path(video_folder).resolve()))


class _PickerState:
    def __init__(self, starting_positions, half_width, object_half_width=None):
        # positions: label -> [x, y] or None if not yet placed
        self.starting_positions = {
            label: (list(pos) if pos else None) for label, pos in starting_positions.items()
        }
        self.positions = {label: (list(pos) if pos else None) for label, pos in self.starting_positions.items()}
        self.half_width = half_width
        # Real (unpadded) footprint half-width, drawn as a light inner
        # outline -- the same region scoring.py treats as "the object".
        self.object_half_width = object_half_width
        self.history = []  # list of {label: [x,y] or None, ...} snapshots
        self.dragging = None
        self.drag_offset = (0, 0)
        self.selected = None  # label currently highlighted for arrow-key nudging

    def _snapshot(self):
        return {k: (list(v) if v else None) for k, v in self.positions.items()}

    def push_history(self):
        self.history.append(self._snapshot())

    def undo(self):
        if not self.history:
            return
        self.positions = self.history.pop()
        self.dragging = None

    def reset(self):
        self.push_history()
        self.positions = {k: (list(v) if v else None) for k, v in self.starting_positions.items()}

    def swap(self):
        a, b = OBJECT_LABELS
        if self.positions.get(a) is None and self.positions.get(b) is None:
            return
        self.push_history()
        self.positions[a], self.positions[b] = self.positions.get(b), self.positions.get(a)

    def hit_test(self, x, y):
        for label, pos in self.positions.items():
            if pos is None:
                continue
            hw = self.half_width
            if abs(x - pos[0]) <= hw and abs(y - pos[1]) <= hw:
                return label
        return None

    def all_set(self):
        return all(self.positions.get(label) is not None for label in OBJECT_LABELS)

    def start_drag(self, label, x, y):
        self.push_history()
        pos = self.positions[label]
        self.dragging = label
        self.selected = label
        self.drag_offset = (x - pos[0], y - pos[1])

    def place_next_unset(self, x, y):
        for label in OBJECT_LABELS:
            if self.positions.get(label) is None:
                self.push_history()
                self.positions[label] = [x, y]
                self.selected = label
                return True
        return False

    def drag_to(self, x, y):
        if self.dragging is None:
            return
        self.positions[self.dragging] = [x - self.drag_offset[0], y - self.drag_offset[1]]

    def end_drag(self):
        self.dragging = None

    def select_next(self):
        """Cycle self.selected through placed objects (Tab hotkey)."""
        placed = [label for label in OBJECT_LABELS if self.positions.get(label) is not None]
        if not placed:
            self.selected = None
            return
        if self.selected not in placed:
            self.selected = placed[0]
            return
        idx = placed.index(self.selected)
        self.selected = placed[(idx + 1) % len(placed)]

    def move_selected(self, dx, dy):
        """Nudge the selected hitbox by (dx, dy) px -- arrow-key movement,
        for precise correction after the coarse click/drag placement."""
        if self.selected is None or self.positions.get(self.selected) is None:
            return
        self.push_history()
        pos = self.positions[self.selected]
        self.positions[self.selected] = [pos[0] + dx, pos[1] + dy]
