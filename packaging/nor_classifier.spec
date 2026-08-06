# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the RATlab NOR GUI -- shared across macOS,
Windows, and Linux (the platform-specific packaging step, i.e. wrapping
this into a .app / leaving it as a folder+.exe / wrapping it into an
AppImage, happens in build_mac.sh / build_windows.bat / build_linux.sh).

Note on what's bundled vs. not: this bundles the GUI itself and its
Python dependencies (PySide6, opencv, sleap-io, etc. -- see
requirements.txt), but NOT the SLEAP training/inference framework or a
RATlab folder's models/. SLEAP is invoked as an external `sleap-track`
command via subprocess (see sleap_inference.py), so it needs to already
be installed and on PATH on whatever machine runs the packaged app --
same requirement as running it unpackaged. The RATlab folder (models/,
and where job/coordinate/output data lives) is located at runtime via
ratlab_locator.py, not bundled in.

Build with (from this packaging/ folder):
    pyinstaller --noconfirm --clean nor_classifier.spec
(the build_*.sh/.bat scripts alongside this do that plus the
platform-specific wrapping step).
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

block_cipher = None

PACKAGING_DIR = Path(SPECPATH).resolve()
NOR_CLASSIFIER_DIR = PACKAGING_DIR.parent
GUI_DIR = NOR_CLASSIFIER_DIR / "gui"

APP_NAME = "RATlab NOR"

# Every nor_classifier/*.py and nor_classifier/gui/*.py module PyInstaller's
# static analysis might not follow on its own (flat "import config" style
# imports from within other bundled modules are usually found fine, but
# listing them explicitly is cheap insurance against a module silently
# missing from the bundle and only failing at runtime on someone's machine).
HIDDEN_IMPORTS = [
    "config", "object_picker", "sleap_inference", "scoring", "excel_writer",
    "validation_video", "pose_filters", "pose_utils", "video_crop", "reset",
    "job_queue", "job_table_model", "job_config", "app_settings",
    "batch_runner", "batch_worker_process", "crop_runner", "crop_worker_process",
    "crop_setup_dialog", "add_job_dialog", "settings_dialog", "object_setup_dialog",
    "main_window", "ratlab_locator", "frozen_config",
]

# sleap_io/__init__.py uses lazy_loader.attach() to lazily import its io/,
# model/, and codecs/ subpackages on first attribute access via a dynamic
# __getattr__, rather than plain `import` statements -- PyInstaller's
# static analysis can't see that pattern, so left to its own devices it
# only bundles sleap_io's top-level __init__.py/version.py and silently
# drops everything else (surfaced at runtime as e.g. "No module named
# 'sleap_io.model'" the first time pose_utils.load_track_from_slp() -> a
# lazy-loaded sleap_io attribute -- actually gets touched). Force all of
# it in explicitly rather than trying to enumerate which specific
# submodules get touched.
HIDDEN_IMPORTS += collect_submodules("sleap_io")

# imageio's __init__.py reads its own version via
# importlib.metadata.version("imageio") at import time -- that needs the
# package's dist-info metadata on disk, which PyInstaller doesn't bundle
# by default (the contrib hook for imageio only collects its data files
# and plugin submodules, not its metadata). Without this, the app builds
# and starts fine but crashes with "No package metadata was found for
# imageio" the moment anything actually touches imageio -- video reading
# during a batch run, not at startup, which is why this can slip past a
# quick smoke test.
DATAS = copy_metadata("imageio")

# This app only uses PySide6. If the build environment is a big shared
# one (conda base env, etc.) that also happens to have another Qt
# bindings package installed for some unrelated tool (PyQt5 via Spyder or
# jupyter-qtconsole is a common one), PyInstaller hard-stops the build
# the moment its analysis touches both -- bundling two Qt bindings
# packages together isn't supported (their native libraries conflict).
# Excluding the others outright avoids that regardless of what else is
# installed alongside PySide6 in whatever environment this is built in.
EXCLUDES = ["PyQt5", "PyQt6", "PySide2"]

a = Analysis(
    [str(GUI_DIR / "app.py")],
    pathex=[str(NOR_CLASSIFIER_DIR), str(GUI_DIR)],
    binaries=[],
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

WIN_ICON_PATH = PACKAGING_DIR / "icon.ico"

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # windowed app -- no terminal window pops up
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon= only does anything on Windows (sets the .exe's file icon);
    # ignored elsewhere, so safe to pass unconditionally.
    icon=str(WIN_ICON_PATH) if WIN_ICON_PATH.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

if sys.platform == "darwin":
    # BUNDLE() only does anything on macOS -- this is what makes the app
    # show up as "RATlab NOR" in the menu bar/Dock instead of
    # "Python" (see the earlier conversation about that).
    ICON_PATH = PACKAGING_DIR / "icon.icns"
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=str(ICON_PATH) if ICON_PATH.exists() else None,
        bundle_identifier="lab.ratlab.nor",
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": "1.0.0",
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
        },
    )
