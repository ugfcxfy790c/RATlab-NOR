#!/usr/bin/env bash
# Builds RATlab_NOR-x86_64.AppImage -- a single self-contained
# executable file, the closest Linux equivalent to a portable .app/.exe
# (double-click and run, no install step, no root needed).
#
# Usage:
#   cd nor_classifier/packaging
#   ./build_linux.sh
#
# Output:
#   dist/RATlab_NOR-x86_64.AppImage  -- the app by itself
#   dist/RATlab-linux.tar.gz             -- app + models/, one download,
#                                            ready to extract and run
#
# If you just want the AppImage: move it into your RATlab folder, next
# to models/. The app auto-detects RATlab from its own location when
# it's sitting right next to models/ like that, so no setup prompt
# appears; if it's ever moved somewhere else, it'll ask once (see
# ratlab_locator.py) and remember.
#
# If you want a single shareable download (e.g. for a GitHub Release):
# RATlab-linux.tar.gz already contains a RATlab/ folder with the
# AppImage and a copy of models/ in the right layout for auto-detection
# -- someone else just extracts it and double-clicks (or runs it), no
# setup needed. Upload that tarball as a Release asset rather than
# committing it to the repo -- GitHub caps regular repo files at 100MB
# and this is usually well over that once models/ is included.
#
# Requires a working SLEAP install already on PATH on this machine --
# see nor_classifier.spec's docstring. This script only packages the GUI.
#
# This script builds from a dedicated virtual environment (packaging/.venv),
# not whatever `python3` happens to resolve to on your machine -- building
# from a big shared environment (a conda base env, etc.) risks PyInstaller
# tripping over unrelated packages installed there for other tools (e.g. a
# second Qt bindings package like PyQt5 pulled in by some other GUI tool,
# which PyInstaller refuses to bundle alongside PySide6). The venv is
# created once and reused on later runs -- only the first build pays the
# setup cost.

set -euo pipefail
cd "$(dirname "$0")"

APP_NAME="RATlab NOR"
DIST_DIR="dist/$APP_NAME"
APPDIR="dist/RATlab_NOR.AppDir"

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

python3 -m PyInstaller --noconfirm --clean nor_classifier.spec

deactivate

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
cp -r "$DIST_DIR"/* "$APPDIR/usr/bin/"

cat > "$APPDIR/RATlab_NOR.desktop" << EOF
[Desktop Entry]
Type=Application
Name=RATlab NOR
Exec=RATlab NOR
Icon=nor_classifier
Categories=Science;
Terminal=false
EOF

# AppImage requires an icon to build at all -- use a real one at
# packaging/icon.png if it exists, else generate a placeholder so this
# doesn't hard-fail (see make_placeholder_icon.py's docstring).
if [ -f icon.png ]; then
  cp icon.png "$APPDIR/nor_classifier.png"
else
  python3 make_placeholder_icon.py "$APPDIR/nor_classifier.png"
  echo "(using a placeholder icon -- drop a real packaging/icon.png to replace it)"
fi

ln -sf "usr/bin/$APP_NAME" "$APPDIR/AppRun"

if command -v appimagetool >/dev/null 2>&1; then
  APPIMAGETOOL=appimagetool
else
  echo "Downloading appimagetool (one-time)..."
  curl -L -o /tmp/appimagetool https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
  chmod +x /tmp/appimagetool
  APPIMAGETOOL=/tmp/appimagetool
fi

"$APPIMAGETOOL" "$APPDIR" "dist/RATlab_NOR-x86_64.AppImage"

echo
echo "Built: dist/RATlab_NOR-x86_64.AppImage"
echo "Move it into your RATlab folder (next to models/), mark it executable if needed (chmod +x), and double-click."

# --- release bundle: AppImage + models/, one archive -----------------------
RATLAB_DIR="../.."   # nor_classifier/packaging -> nor_classifier -> RATlab
if [ -d "$RATLAB_DIR/models" ]; then
  echo
  echo "Building release bundle (AppImage + models/)..."
  RELEASE_DIR="dist/release/RATlab"
  rm -rf dist/release
  mkdir -p "$RELEASE_DIR"
  cp -r "$RATLAB_DIR/models" "$RELEASE_DIR/models"
  cp "dist/RATlab_NOR-x86_64.AppImage" "$RELEASE_DIR/"
  chmod +x "$RELEASE_DIR/RATlab_NOR-x86_64.AppImage"

  rm -f dist/RATlab-linux.tar.gz
  tar -C dist/release -czf dist/RATlab-linux.tar.gz RATlab
  echo "Release bundle: dist/RATlab-linux.tar.gz -- upload this as a GitHub Release asset."
else
  echo
  echo "No models/ found at $RATLAB_DIR/models -- skipping the release bundle (app-only build above still works)."
fi
