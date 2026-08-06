"""
Entry point for the NOR classifier GUI.

Run with:
    python gui/app.py

Or double-click the packaged .app (macOS) / .exe (Windows) /
AppImage (Linux) -- see nor_classifier.spec and the build_*.sh/.bat
scripts alongside it.
"""

import multiprocessing
import sys
from pathlib import Path


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


if not _is_frozen():
    # nor_classifier's modules use flat imports ("import config", "import
    # object_picker", ...), so make that directory importable rather than
    # restructuring the existing pipeline code. Not needed when frozen --
    # nor_classifier.spec's pathex already puts nor_classifier/ and
    # nor_classifier/gui/ on sys.path inside the bundle.
    NOR_CLASSIFIER_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(NOR_CLASSIFIER_DIR))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import QApplication

import config
import object_picker
import app_settings
import ratlab_locator
from frozen_config import patch_config_for_ratlab_dir
from job_queue import JobQueue, default_app_data_dir
from main_window import MainWindow


def _resolve_frozen_paths() -> bool:
    """config.py derives all its paths (NOR_CLASSIFIER_DIR, RATLAB_DIR,
    MODEL_PATHS, ...) from its own __file__ location -- correct for a
    normal checkout, but meaningless once PyInstaller bundles config.py
    into a frozen app (__file__ then points inside the bundle, not at the
    user's actual RATlab folder). Find out where RATlab actually is (see
    ratlab_locator.get_ratlab_dir -- usually auto-detected with no prompt
    at all, when the app is sitting right next to models/ the way the
    packaging scripts' release bundles produce; a folder-picker prompt is
    only a fallback), then patch config's attributes to match -- see
    frozen_config.py. The batch/crop worker child processes reapply the
    same patch themselves once they know which nor_classifier_dir this
    resolved to (passed to them as a plain string, same as in the
    unpackaged/dev case).

    Returns False if the user canceled the picker, meaning the app
    should quit rather than run against a nonexistent/wrong RATlab.
    """
    ratlab_dir = ratlab_locator.get_ratlab_dir()
    if ratlab_dir is None:
        ratlab_dir = ratlab_locator.prompt_for_ratlab_dir()
        if ratlab_dir is None:
            return False
        ratlab_locator.set_ratlab_dir(ratlab_dir)

    patch_config_for_ratlab_dir(config, ratlab_dir / "nor_classifier")
    return True


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("RATlab NOR")
    app.setApplicationDisplayName("RATlab NOR")

    if ratlab_locator.is_frozen():
        if not _resolve_frozen_paths():
            return

    app_data_dir = default_app_data_dir(config.NOR_CLASSIFIER_DIR)
    job_queue = JobQueue(app_data_dir).load()

    window = MainWindow(
        job_queue=job_queue,
        object_picker_module=object_picker,
        default_output_base=app_settings.get_default_output_base(config.NOR_CLASSIFIER_DIR, config),
        config_module=config,
    )
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # Required for multiprocessing (used by the batch runner) to behave
    # correctly once this app is frozen into a .app/.exe/AppImage --
    # must run immediately under this guard, before anything else spawns
    # a process. Harmless as a no-op when running unfrozen.
    multiprocessing.freeze_support()
    main()
