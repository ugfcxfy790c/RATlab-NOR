# Packaging RATlab NOR

Turns the GUI into a double-clickable app: `.app` on macOS, a `.exe` folder
on Windows, an `.AppImage` on Linux. One shared PyInstaller spec
(`nor_classifier.spec`), one build script per OS.

**Important: PyInstaller doesn't cross-compile.** A `.app` has to be built
by running its script on an actual Mac, a `.exe` on an actual Windows
machine. You can't build all three from one computer. Each script below
must be run on that OS.

Every build also needs SLEAP's own `sleap-track` command already
installed and on `PATH` on that machine -- the packaged app shells out to
it for inference the same way the unpackaged version does. Packaging
doesn't bundle SLEAP itself, only the GUI and its own pipeline code.

## macOS

```
cd nor_classifier/packaging
./build_mac.sh
```

Produces `dist/RATlab NOR.app` and `dist/RATlab-mac.zip` (see
"Sharing a build" below). Since the app isn't code-signed/notarized, the
first launch needs a right-click -> Open (not a double-click) to get past
Gatekeeper; after that first time, double-clicking works normally.

## Windows

```
cd nor_classifier\packaging
build_windows.bat
```

Produces `dist\RATlab NOR\RATlab NOR.exe` (a whole folder,
PyInstaller's onedir mode, not a single file) and `dist\RATlab-windows.zip`.

## Linux

```
cd nor_classifier/packaging
./build_linux.sh
```

Produces `dist/RATlab_NOR-<arch>.AppImage` (a single file, no install
step) and `dist/RATlab-linux-<arch>.tar.gz`, where `<arch>` is whatever
`uname -m` reports on the machine you ran the script on (`x86_64` or
`aarch64`) -- PyInstaller doesn't cross-compile, so it always builds for
the architecture it's actually running on. Run the script again on the
other architecture if you need both; there's no single build that
covers them.

This script downloads `appimagetool` automatically the first time if it's
not already on your `PATH`. It also generates a placeholder icon if
`packaging/icon.png` doesn't exist yet -- drop a real one there anytime
and future builds will use it instead.

**Don't have Linux hardware?** `.github/workflows/build-linux.yml` runs
this same script on real GitHub Actions runners for both `x86_64` (most
desktops/laptops) and `aarch64` (Raspberry Pi, Ampere/Graviton-style ARM
cloud VMs, ARM dev laptops) -- push a tag like `v1.2.3` (or trigger it
manually from the Actions tab) and download the resulting
`RATlab_NOR-x86_64.AppImage` / `RATlab_NOR-aarch64.AppImage` from the
run's artifacts, or have them attach automatically to a GitHub Release
for that tag. Building both matters beyond convenience: unlike
Windows-on-ARM (x64 emulation) or Apple Silicon (Rosetta), stock Linux
ARM64 has no built-in x86_64 compatibility layer, so an x86_64-only
AppImage simply won't run on an ARM Linux machine at all.

## Sharing a build (one download, no setup)

Each script also builds a second output -- `RATlab-mac.zip` /
`RATlab-windows.zip` / `RATlab-linux.tar.gz` -- containing a `RATlab/`
folder with the packaged app *and* a copy of your `models/` folder
together, in the same layout the app auto-detects on its own (see below).
Someone else downloads that one file, extracts it, and double-clicks --
no folder-picking step needed, since the app finds `models/` sitting
right next to it.

This is what you'd attach to a **GitHub Release**, not commit into the
repo -- regular repo files are capped at 100MB on GitHub, and these
bundles are usually well over that once `models/` is included (currently
~200MB across the checkpoints in this project). Release assets don't have
that limit.

If `models/` isn't found at build time (e.g. `RATLAB_DIR/models` doesn't
exist on the machine you're building on), this step is skipped and you
still get the plain app output above.

## How the app finds your RATlab folder

Two ways, tried in order:

1. **Auto-detect.** If the packaged app is sitting right next to `models/`
   -- which is exactly the layout the release bundles above produce, and
   also what you get if you just move the plain app output into your
   RATlab folder yourself -- it figures this out from its own location
   and needs no prompt at all.
2. **Ask once.** If auto-detect doesn't pan out (app was moved somewhere
   else, run from Applications, etc.), it falls back to whatever was
   chosen last time; if there's no remembered answer either, it asks via
   a normal folder picker and remembers the answer for next time.

That remembered answer lives in the OS's normal per-app config location,
separate from RATlab itself. If the app ever seems confused about paths
(e.g. after moving RATlab without moving the app), deleting that file
makes it ask again.

## Updating the app after code changes

Re-run the same build script -- there's no separate "update" step, a
build always produces a fresh copy from the current source. The
`dist/` and `build/` folders under `packaging/` are safe to delete
between builds if a stale one ever causes confusion (not tracked in
version control).
