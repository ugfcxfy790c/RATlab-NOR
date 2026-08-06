"""
Shared logic for pointing config.py's paths at the user's real RATlab
folder when running frozen (see ratlab_locator.py's docstring for why
config.py's own __file__-derived paths break under PyInstaller).

Used by both the main GUI process (app.py, after asking the user where
RATlab is via ratlab_locator) and the batch/crop worker child processes
spawned via multiprocessing (batch_worker_process.py,
crop_worker_process.py) -- each of those is a fresh interpreter that
re-imports config.py from scratch when it starts, so each has to reapply
this patch itself; a worker can't inherit the parent GUI process's
already-patched config module across the process boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def patch_config_for_ratlab_dir(config_module, nor_classifier_dir) -> None:
    """`nor_classifier_dir` is RATLAB_DIR/nor_classifier -- true whether
    or not we're frozen, since ratlab_locator hands back the user's
    RATlab folder and app.py joins "nor_classifier" onto it the same way
    config.py itself would in an unpackaged checkout. So RATLAB_DIR here
    is just nor_classifier_dir's parent."""
    nor_classifier_dir = Path(nor_classifier_dir)
    ratlab_dir = nor_classifier_dir.parent

    config_module.RATLAB_DIR = ratlab_dir
    config_module.NOR_CLASSIFIER_DIR = nor_classifier_dir
    # Re-derive the same way config.py originally derives these from
    # RATLAB_DIR/NOR_CLASSIFIER_DIR, just rooted at the real folder
    # instead of wherever PyInstaller unpacked the bundle.
    config_module.MODEL_PATHS = [ratlab_dir / "models" / Path(p).name for p in config_module.MODEL_PATHS]
    config_module.PREDICTIONS_FOLDER = nor_classifier_dir / "predictions"
    config_module.OUTPUT_FOLDER = nor_classifier_dir / "output"
    config_module.OBJECT_COORDS_FILE = nor_classifier_dir / "object_coords.json"
