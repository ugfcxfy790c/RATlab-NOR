@echo off
REM Builds RATlab NOR.exe -- run this ON WINDOWS (PyInstaller
REM doesn't cross-compile; a .exe has to be built by running PyInstaller
REM on Windows).
REM
REM Usage:
REM   cd nor_classifier\packaging
REM   build_windows.bat
REM
REM Output:
REM   dist\RATlab NOR\RATlab NOR.exe   -- the app by itself
REM   dist\RATlab-windows.zip                  -- app + models\, one
REM                                                download, ready to
REM                                                unzip and run
REM
REM PyInstaller's onedir mode (a whole folder, not a single file) is used
REM rather than onefile -- it starts faster and is easier to debug if
REM something's missing from the bundle.
REM
REM If you just want the .exe: move (don't just copy) the whole
REM "dist\RATlab NOR" folder into your RATlab folder, next to
REM models\. The app auto-detects RATlab from its own location when it's
REM sitting right next to models\ like that, so no setup prompt appears;
REM if it's ever moved somewhere else, it'll ask once (see
REM ratlab_locator.py) and remember.
REM
REM If you want a single shareable download (e.g. for a GitHub Release):
REM RATlab-windows.zip already contains a RATlab\ folder with the app and
REM a copy of models\ in the right layout for auto-detection -- someone
REM else just unzips it and double-clicks, no setup needed. Upload that
REM zip as a Release asset rather than committing it to the repo --
REM GitHub caps regular repo files at 100MB and this is usually well over
REM that once models\ is included.
REM
REM Requires a working SLEAP install already on PATH on this machine --
REM see nor_classifier.spec's docstring. This script only packages the GUI.
REM
REM This script builds from a dedicated virtual environment (packaging\.venv),
REM not whatever `python` happens to resolve to on your machine -- building
REM from a big shared environment (a conda base env, etc.) risks PyInstaller
REM tripping over unrelated packages installed there for other tools (e.g. a
REM second Qt bindings package like PyQt5 pulled in by Spyder/jupyter, which
REM PyInstaller refuses to bundle alongside PySide6). The venv is created once
REM and reused on later runs -- only the first build pays the setup cost.

cd /d "%~dp0"

if not exist ".venv" (
    echo Creating a dedicated build environment at packaging\.venv ^(one-time^)...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

REM PyInstaller can only bundle packages that are actually importable in
REM whatever Python environment runs it -- so both the app's own
REM dependencies and PyInstaller itself need to be installed here, in the venv.
python -m pip install --upgrade pip
python -m pip install --upgrade -r ..\requirements.txt
python -m pip install --upgrade pyinstaller

REM requirements.txt installs plain opencv-python, which bundles its own Qt5
REM for cv2.imshow/highgui window support -- something this app never uses
REM (object picking was deliberately rebuilt on PySide6 rather than cv2
REM windows). That bundled Qt5 needs desktop OpenGL symbols that aren't
REM reliably present once frozen, and crashes the packaged app on startup
REM the moment anything does `import cv2`. Swap to opencv-python-headless
REM (same cv2 API otherwise, just without the GUI backend) for the build
REM venv specifically -- leaves requirements.txt itself alone, since the
REM unpackaged/legacy CLI path (object_picker.py's cv2-window-based picker)
REM can still want the real thing.
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install --upgrade opencv-python-headless

python -m PyInstaller --noconfirm --clean nor_classifier.spec

call .venv\Scripts\deactivate.bat

echo.
echo Built: dist\RATlab NOR\RATlab NOR.exe
echo Move the whole "dist\RATlab NOR" folder into your RATlab folder (next to models\).

REM --- release bundle: app + models\, one archive ---------------------------
set "RATLAB_DIR=..\.."
if exist "%RATLAB_DIR%\models" (
    echo.
    echo Building release bundle ^(app + models\^)...
    set "RELEASE_DIR=dist\release\RATlab"
    if exist dist\release rmdir /s /q dist\release
    mkdir "%RELEASE_DIR%"
    xcopy /E /I /Q "%RATLAB_DIR%\models" "%RELEASE_DIR%\models" >nul
    xcopy /E /I /Q "dist\RATlab NOR" "%RELEASE_DIR%\RATlab NOR" >nul

    if exist dist\RATlab-windows.zip del /f /q dist\RATlab-windows.zip
    powershell -NoProfile -Command "Compress-Archive -Path '%RELEASE_DIR%' -DestinationPath 'dist\RATlab-windows.zip' -Force"
    echo Release bundle: dist\RATlab-windows.zip -- upload this as a GitHub Release asset.
) else (
    echo.
    echo No models\ found at %RATLAB_DIR%\models -- skipping the release bundle ^(app-only build above still works^).
)
