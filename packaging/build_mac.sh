#!/usr/bin/env bash
# Builds RATlab NOR.app -- run this ON A MAC (PyInstaller doesn't
# cross-compile; a .app has to be built by running PyInstaller on macOS).
#
# Usage:
#   cd nor_classifier/packaging
#   ./build_mac.sh
#
# Output:
#   dist/RATlab NOR.app        -- the app by itself
#   dist/RATlab-mac.zip            -- app + models/, one download, ready
#                                      to unzip and run (see below)
#
# If you just want the .app: move (don't just copy) it into your RATlab
# folder, next to models/ -- e.g. RATlab/RATlab NOR.app. The app
# auto-detects RATlab from its own location when it's sitting right next
# to models/ like that, so no setup prompt appears; if it's ever moved
# somewhere else, it'll ask once (see ratlab_locator.py) and remember.
#
# If you want a single shareable download (e.g. for a GitHub Release):
# RATlab-mac.zip already contains a RATlab/ folder with the app and a
# copy of models/ in the right layout for auto-detection -- someone else
# just unzips it and double-clicks, no setup needed. Upload that zip as a
# Release asset rather than committing it to the repo -- GitHub caps
# regular repo files at 100MB and this is usually well over that once
# models/ is included.
#
# First launch will likely be blocked by Gatekeeper since this isn't
# code-signed/notarized ("RATlab NOR.app is damaged and can't be
# opened" or similar) -- right-click the app -> Open, then confirm in
# the dialog that appears, instead of double-clicking. You only need to
# do this once.
#
# This script builds from a dedicated virtual environment (packaging/.venv),
# not whatever `python3` happens to resolve to on your machine -- building
# from a big shared environment (a conda base env, etc.) risks PyInstaller
# tripping over unrelated packages installed there for other tools (e.g. a
# second Qt bindings package like PyQt5 pulled in by Spyder/jupyter, which
# PyInstaller refuses to bundle alongside PySide6). The venv is created once
# and reused on later runs -- only the first build pays the setup cost.

set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating a dedicated build environment at packaging/.venv (one-time)..."
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# PyInstaller can only bundle packages that are actually importable in
# whatever Python environment runs it -- so both the app's own
# dependencies and PyInstaller itself need to be installed here, in the venv.
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade -r ../requirements.txt
python3 -m pip install --upgrade pyinstaller

# requirements.txt installs plain opencv-python, which bundles its own Qt5
# for cv2.imshow/highgui window support -- something this app never uses
# (object picking was deliberately rebuilt on PySide6 rather than cv2
# windows). That bundled Qt5 needs desktop OpenGL symbols that aren't
# reliably present once frozen, and crashes the packaged app on startup
# with "undefined symbol: glMatrixMode" the moment anything does `import
# cv2`. Swap to opencv-python-headless (same cv2 API otherwise, just
# without the GUI backend) for the build venv specifically -- leaves
# requirements.txt itself alone, since the unpackaged/legacy CLI path
# (object_picker.py's cv2-window-based picker) can still want the real thing.
python3 -m pip uninstall -y opencv-python opencv-python-headless
python3 -m pip install --upgrade opencv-python-headless

# SLEAP's own CLI ("sleap-track") must already be installed and on PATH
# on this machine for the packaged app to actually run inference -- see
# nor_classifier.spec's docstring. This script only packages the GUI.
python3 -m PyInstaller --noconfirm --clean nor_classifier.spec

deactivate

echo
echo "Built: dist/RATlab NOR.app"
echo "Move it into your RATlab folder (next to models/), then right-click -> Open the first time."

# --- release bundle: app + models/, one archive ---------------------------
RATLAB_DIR="../.."   # nor_classifier/packaging -> nor_classifier -> RATlab
if [ -d "$RATLAB_DIR/models" ]; then
  echo
  echo "Building release bundle (app + models/)..."
  RELEASE_DIR="dist/release/RATlab"
  rm -rf dist/release
  mkdir -p "$RELEASE_DIR"
  cp -R "$RATLAB_DIR/models" "$RELEASE_DIR/models"
  cp -R "dist/RATlab NOR.app" "$RELEASE_DIR/RATlab NOR.app"

  rm -f dist/RATlab-mac.zip
  if command -v ditto >/dev/null 2>&1; then
    # ditto (built into macOS) preserves .app bundle attributes/symlinks
    # correctly, unlike a naive zip -- the tool Apple itself recommends
    # for zipping .app bundles.
    ditto -c -k --keepParent "$RELEASE_DIR" "dist/RATlab-mac.zip"
  else
    (cd dist/release && zip -r -y "../RATlab-mac.zip" "RATlab")
  fi
  echo "Release bundle: dist/RATlab-mac.zip -- upload this as a GitHub Release asset."
else
  echo
  echo "No models/ found at $RATLAB_DIR/models -- skipping the release bundle (app-only build above still works)."
fi
