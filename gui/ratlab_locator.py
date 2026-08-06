"""
Finds the user's RATlab folder (the one containing models/ and, in an
unpackaged dev checkout, nor_classifier/) when running as a frozen
PyInstaller build (.app / .exe / AppImage).

Why this exists: config.py derives NOR_CLASSIFIER_DIR/RATLAB_DIR from its
own file location (Path(__file__).resolve().parent(.parent)). That's
correct for a normal checkout, but once PyInstaller bundles config.py
into a frozen app, __file__ points at wherever PyInstaller unpacked it
(a temp dir on macOS/Windows, a squashfs mount for an AppImage) -- not at
the user's actual RATlab folder with their models and data.

Two ways this gets resolved, tried in order:
  1. Auto-detect: guess from where the running executable actually is,
     which works whenever the packaged app was distributed sitting right
     next to models/ (e.g. everything unzipped from one downloaded
     release bundle) and nobody's moved it since -- the common case, and
     needs no prompt at all.
  2. If that guess doesn't pan out (app was moved elsewhere, run from
     Applications, etc.), fall back to whatever was chosen and remembered
     last time; if there's no remembered answer either, ask once via a
     native folder picker and remember the answer for next time.

The remembered answer is stored outside RATlab itself (in the OS's normal
per-user app-config location, via QStandardPaths), since a frozen app
doesn't have a "next to my own source" folder to write into the way the
unpackaged dev version uses nor_classifier/app_data/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QFileDialog, QMessageBox


def is_frozen() -> bool:
    """True when running inside a PyInstaller-built .app/.exe/AppImage,
    False for a normal `python gui/app.py` dev run."""
    return bool(getattr(sys, "frozen", False))


def _locator_file() -> Path:
    config_dir = QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)
    if not config_dir:
        # Extremely unlikely (QStandardPaths always resolves something on
        # every desktop OS), but fall back to the user's home dir rather
        # than crash if it ever happens.
        config_dir = str(Path.home() / ".nor_classifier")
    return Path(config_dir) / "ratlab_location.json"


def _auto_detect_ratlab_dir() -> Path | None:
    """Guess RATlab's location from wherever this packaged executable is
    actually running from, before ever bothering the user. Each
    platform's packaged layout puts the running executable a different
    number of directories below RATlab (see each build script's
    docstring for the exact layout it produces):

        macOS:   RATlab/RATlab NOR.app/Contents/MacOS/RATlab NOR
        Windows: RATlab/RATlab NOR/RATlab NOR.exe
        Linux:   RATlab/NOR_Classifier-x86_64.AppImage -- a single file,
                 and sys.executable points inside a temp squashfs mount
                 while it's running rather than at the real file, so the
                 AppImage runtime's own $APPIMAGE env var is used instead.
    """
    try:
        if sys.platform == "darwin":
            candidate = Path(sys.executable).resolve().parents[3]
        elif sys.platform.startswith("win"):
            candidate = Path(sys.executable).resolve().parents[1]
        else:
            appimage_path = os.environ.get("APPIMAGE")
            if not appimage_path:
                return None
            candidate = Path(appimage_path).resolve().parent
    except IndexError:
        return None

    if candidate.is_dir() and (candidate / "models").is_dir():
        return candidate
    return None


def _get_saved_ratlab_dir() -> Path | None:
    """Previously user-chosen RATlab folder, if any and if it still looks
    valid (still exists, still has a models/ subfolder)."""
    path_file = _locator_file()
    if not path_file.exists():
        return None
    try:
        saved = json.loads(path_file.read_text()).get("ratlab_dir")
    except (json.JSONDecodeError, OSError):
        return None
    if not saved:
        return None
    candidate = Path(saved)
    if candidate.is_dir() and (candidate / "models").is_dir():
        return candidate
    return None


def get_ratlab_dir() -> Path | None:
    """RATlab's location, trying auto-detection first (see
    _auto_detect_ratlab_dir) and falling back to a previously-saved
    choice. Returns None only if neither works -- the caller
    (app.py/frozen_config.py) should fall back to prompt_for_ratlab_dir
    in that case."""
    auto = _auto_detect_ratlab_dir()
    if auto is not None:
        # Keep the saved pointer in sync with reality even when we didn't
        # need it this time -- so if the app is later moved somewhere
        # auto-detection can't figure out, the last-known-good answer is
        # still there as a fallback instead of immediately prompting.
        set_ratlab_dir(auto)
        return auto
    return _get_saved_ratlab_dir()


def set_ratlab_dir(path: Path) -> None:
    path_file = _locator_file()
    path_file.parent.mkdir(parents=True, exist_ok=True)
    path_file.write_text(json.dumps({"ratlab_dir": str(path)}, indent=2))


def prompt_for_ratlab_dir(parent=None) -> Path | None:
    """Native folder picker for locating RATlab. Loops until the user
    either picks something with a models/ subfolder (the one thing every
    RATlab folder has) or cancels -- returns None only on cancel."""
    QMessageBox.information(
        parent, "Locate your RATlab folder",
        "This is a packaged copy of RATlab NOR, so it needs to know where "
        "your RATlab folder is (the one containing models/) -- it'll remember "
        "your choice after this.",
    )
    while True:
        folder = QFileDialog.getExistingDirectory(parent, "Choose your RATlab folder", str(Path.home()))
        if not folder:
            return None
        candidate = Path(folder)
        if (candidate / "models").is_dir():
            return candidate
        reply = QMessageBox.warning(
            parent, "Doesn't look like RATlab",
            f"{candidate} doesn't have a models/ subfolder, so this probably isn't "
            f"the right folder. Pick again?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return None
