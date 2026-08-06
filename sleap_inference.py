"""
Wrapper around SLEAP's inference CLI for unattended batch pose tracking.

Calls `sleap-nn track` directly rather than going through the `sleap`
package's wrapper command, falling back to `sleap track` / legacy
`sleap-track` only if `sleap-nn` isn't on PATH. Every video is wrapped in
a minimal single-video `.slp` Labels project before tracking (see
`_ensure_video_wrapper_slp`) and passed with `--video_index 0`, matching
what the SLEAP GUI does internally and forcing `grayscale=True` explicitly
rather than relying on (unreliable, on lossy-compressed footage)
auto-detection.

Shells out to the command line so it works with whatever SLEAP install is
on PATH; only imports `sleap_io` (lightweight, no GPU) to build the
video-wrapper .slp files.
"""

from pathlib import Path
import io
import os
import re
import shutil
import subprocess
import sys
import time


class SleapTrackNotFound(RuntimeError):
    pass


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class _ProgressStreamer:
    """Shared line-splitting/filtering logic for both
    _stream_subprocess_pty and _stream_subprocess_pipe below -- only the
    part that gets characters onto the wire differs between them.

    The tracking command's output is two very different kinds of thing
    mixed into one stream: a tqdm-style progress bar that rewrites a
    single line via carriage returns ('\\r'), and everything else (config
    dumps, device/model info, warnings) as ordinary '\\n'-terminated
    lines. The progress bar is the one thing actually worth watching live
    during a long tracking run -- the rest is mostly startup noise that
    would flood the log around it. So: '\\r' updates go to `progress`
    (throttled to at most one call per min_interval_s -- a tqdm bar can
    emit dozens a second) rather than `log`, specifically so a caller
    that wants to redraw a single line in place (as the GUI does -- see
    main_window.py's _on_progress_line) can tell "this is the same bar
    updating" apart from "this is a new, separate message". '\\n' lines
    go to neither live -- they're buffered silently (self.plain_lines) so
    a caller can decide afterwards whether they're worth surfacing (via
    `log`) -- see dump_plain_lines() and run_inference()'s use of it.
    Not every "the process technically exited 0" run is actually fine
    (e.g. sleap-nn reporting success without having actually written its
    output file to a flaky filesystem), so that decision is left to the
    caller rather than being tied to exit code alone in here.
    """

    def __init__(self, log, progress, min_interval_s: float):
        self.log = log
        self.progress = progress
        self.min_interval_s = min_interval_s
        self.buf = ""
        self.last_emit = 0.0
        self.last_logged = None
        self.plain_lines: list[str] = []
        # True right after seeing a '\r' whose meaning isn't known yet --
        # see feed()'s docstring-in-place below for why this needs one
        # character of lookahead.
        self._pending_cr = False

    @staticmethod
    def _clean(raw: str) -> str:
        # Forcing a pty (see _stream_subprocess_pty) makes some tools emit
        # ANSI color/cursor codes they'd otherwise skip when writing to a
        # plain pipe -- strip those so the log shows plain text.
        return _ANSI_ESCAPE_RE.sub("", raw).strip()

    def feed(self, ch: str):
        # A pty's own line-ending translation (ONLCR) rewrites every
        # plain '\n' a child writes into '\r\n' before we ever see it --
        # so an ordinary print()'d line and a real tqdm carriage-return
        # redraw both start by handing us a '\r'; the only way to tell
        # them apart is whether a '\n' immediately follows. Hence the
        # one-character lookahead via _pending_cr: a '\r' immediately
        # followed by '\n' is an ordinary line ending (buffered, not
        # shown live); a '\r' followed by anything else is a genuine
        # progress-bar redraw (shown live, throttled).
        if self._pending_cr:
            self._pending_cr = False
            if ch == "\n":
                self._flush_plain()
                return
            self._flush_progress()
            # fall through -- `ch` still needs to be processed below,
            # it's the start of whatever comes after the redraw

        if ch == "\r":
            self._pending_cr = True
        elif ch == "\n":
            self._flush_plain()
        else:
            self.buf += ch

    def _flush_plain(self):
        line = self._clean(self.buf)
        self.buf = ""
        if line:
            self.plain_lines.append(line)

    def _flush_progress(self):
        line = self._clean(self.buf)
        self.buf = ""
        if line and line != self.last_logged:
            now = time.monotonic()
            if now - self.last_emit >= self.min_interval_s:
                self.progress(f"    {line}")
                self.last_emit = now
                self.last_logged = line
            # else: an intermediate progress-bar redraw, dropped by
            # design -- the trailing flush in finish() still catches it
            # if the process happens to exit right after.

    def finish(self):
        # A trailing '\r' with nothing after it (no more input, EOF) is
        # ambiguous -- but in practice that's always the final progress
        # state right before the process exits, never a real line ending
        # (those get their '\n' before EOF), so treat it as progress.
        if self._pending_cr:
            self._flush_progress()
            self._pending_cr = False

        # Flush whatever's left unterminated -- e.g. the final "100%"
        # state if it wasn't itself followed by a '\r'/'\n' before exit.
        tail = self._clean(self.buf)
        if tail and tail != self.last_logged:
            self.progress(f"    {tail}")

    def dump_plain_lines(self, reason: str):
        """Surfaces the buffered '\\n' output (see class docstring) via
        `log`, prefixed with `reason` -- call this when the caller
        decides, by whatever criteria, that this run's output is worth
        digging into. A no-op if nothing was buffered."""
        if not self.plain_lines:
            return
        self.log(f"    -- full output ({reason}) --")
        for line in self.plain_lines:
            self.log(f"    {line}")


def _stream_subprocess_pty(cmd, env, min_interval_s, log, progress) -> tuple[int, "_ProgressStreamer"]:
    """POSIX (mac/Linux) implementation: runs cmd with its stdout/stderr
    attached to a pseudo-terminal instead of a plain pipe.

    This matters beyond just buffering: tqdm (and most other CLI progress
    bars) explicitly check isatty() and behave very differently
    depending on the answer -- connected to a plain pipe (as a normal
    subprocess.PIPE would be), many of them redraw far less often, or
    only print a final summary, specifically to avoid spamming a log
    file. A pty makes the child see what looks like a real interactive
    terminal, so it behaves the same way it would if you'd run it
    directly in a terminal yourself.

    Also sets a window size on the pty (openpty() alone leaves it at
    0x0) -- without this, tqdm can't determine a terminal width and
    renders nothing at all rather than falling back to a sane default,
    which looks identical to "the progress bar is broken" from here even
    though isatty() is already True.
    """
    import pty
    import errno
    import fcntl
    import struct
    import termios

    master_fd, slave_fd = pty.openpty()
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
    proc = subprocess.Popen(
        cmd, stdout=slave_fd, stderr=slave_fd, stdin=subprocess.DEVNULL,
        close_fds=True, env=env,
    )
    os.close(slave_fd)  # only the child needs the slave end

    streamer = _ProgressStreamer(log, progress, min_interval_s)
    # newline="" disables universal-newlines translation -- the default
    # (newline=None) silently rewrites every '\r' to '\n' before this code
    # ever sees it (true of both os.fdopen and subprocess's own
    # text=True/PIPE, which is why the plain-pipe path below doesn't use
    # that either), which would make every tqdm update look like a plain
    # '\n' line (buffered, not shown live) instead of a progress-bar redraw.
    with os.fdopen(master_fd, "r", buffering=1, newline="", errors="replace") as master:
        while True:
            try:
                ch = master.read(1)
            except OSError as e:
                if e.errno == errno.EIO:
                    break  # child closed its end -- normal EOF for a pty
                raise
            if ch == "":
                if proc.poll() is not None:
                    break
                continue
            streamer.feed(ch)

    proc.wait()
    streamer.finish()
    return proc.returncode, streamer


def _stream_subprocess_pipe(cmd, env, min_interval_s, log, progress) -> tuple[int, "_ProgressStreamer"]:
    """Windows fallback: a real pty needs platform APIs this codebase
    doesn't otherwise depend on (pywinpty/conpty), so this uses a plain
    pipe instead. PYTHONUNBUFFERED (see _stream_subprocess) still fixes
    the buffering half of the problem; a tool that specifically disables
    its progress bar when not attached to a real terminal may still
    print less on Windows than on mac/Linux -- see
    _stream_subprocess_pty's docstring for why that's pty-specific.
    """
    # Deliberately *not* using Popen(text=True) here: it applies the same
    # universal-newlines translation as os.fdopen's default (silently
    # rewriting every '\r' to '\n' before this code ever sees it,
    # breaking the progress-bar/plain-line distinction _ProgressStreamer
    # relies on), and Popen has no way to override that translation
    # itself -- so the pipe is opened in raw binary mode and wrapped by
    # hand with newline="" instead.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, env=env,
    )
    stdout_text = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace", newline="")
    streamer = _ProgressStreamer(log, progress, min_interval_s)
    try:
        while True:
            ch = stdout_text.read(1)
            if ch == "":
                if proc.poll() is not None:
                    break
                continue
            streamer.feed(ch)
    finally:
        stdout_text.close()

    proc.wait()
    streamer.finish()
    return proc.returncode, streamer


def _stream_subprocess(cmd, log, progress, min_interval_s: float = 0.5) -> tuple[int, _ProgressStreamer]:
    """Runs cmd, forwarding its live tqdm-style tracking-progress output
    through `progress` as it runs (throttled -- see _ProgressStreamer),
    rather than staying silent for however long each video's tracking
    takes (the majority of a job's runtime) and only surfacing everything
    after the fact. `progress` is called repeatedly for what is
    conceptually the *same* single line being redrawn -- as opposed to
    `log`, which is for discrete one-off messages -- so a caller that
    wants to show that redraw in place (the GUI does; see
    main_window.py's _on_progress_line) can tell the two apart.

    Ordinary '\\n'-terminated output (config dumps, device info,
    warnings -- not the progress bar) is intentionally kept out of the
    live log entirely. Returns (exit code, the _ProgressStreamer used) --
    the caller decides whether that buffered output is worth surfacing
    via the streamer's dump_plain_lines(), since a nonzero exit code
    isn't the only situation where it's the only clue to what went wrong
    (see run_inference's "reported success but wrote nothing" case).
    """
    # sleap-nn/sleap-track is itself a Python program. CPython only
    # line-buffers stdout when it's attached to a real terminal --
    # writing to a plain pipe makes it fall back to full block buffering
    # (~8KB) regardless of how the writer flushes. PYTHONUNBUFFERED forces
    # CPython's stdout/stderr to be unbuffered for the child process
    # (same effect as `python -u`) -- kept even with the pty path below,
    # since a pty fixes tqdm's own tty-detection behavior but this is a
    # separate, lower-level buffering layer underneath it.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if sys.platform.startswith("win"):
        return _stream_subprocess_pipe(cmd, env, min_interval_s, log, progress)
    return _stream_subprocess_pty(cmd, env, min_interval_s, log, progress)


def _uv_tool_bin_dirs() -> list[Path]:
    """Extra directories to check for a SLEAP command beyond PATH.
    `uv tool install sleap-nn` (the install method this app's own "SLEAP
    not found" warning recommends) puts its shim in a bin directory that
    normally reaches PATH via shell startup files (.zshrc, etc.) -- which
    a double-clicked GUI app never sources, since it isn't launched
    through a login shell. Mirrors uv's own resolution order:
    https://docs.astral.sh/uv/reference/storage/"""
    dirs = []
    for env_var in ("UV_TOOL_BIN_DIR", "XDG_BIN_HOME"):
        value = os.environ.get(env_var)
        if value:
            dirs.append(Path(value))
    dirs.append(Path.home() / ".local" / "bin")  # uv's default on every OS it supports
    return dirs


def _which_with_uv_fallback(name: str) -> str | None:
    """Like shutil.which(), but also checks _uv_tool_bin_dirs() if PATH
    alone doesn't find it. Returns an absolute path (not just the bare
    command name) specifically so the result is directly runnable via
    subprocess regardless of whether the eventual subprocess call
    inherits the same augmented PATH this lookup used."""
    found = shutil.which(name)
    if found:
        return found
    exe_name = f"{name}.exe" if sys.platform.startswith("win") else name
    for directory in _uv_tool_bin_dirs():
        candidate = directory / exe_name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def predictions_path_for(video_path, predictions_folder):
    video_path = Path(video_path)
    predictions_folder = Path(predictions_folder)
    return predictions_folder / f"{video_path.stem}.predictions.slp"


def _find_track_base_cmd():
    """Base command list for tracking, in order of preference: `sleap-nn
    track`, `sleap-nn-track`, `sleap track`, legacy `sleap-track`. Returns
    None if nothing usable is found on PATH or in the uv tool-install
    fallback locations (see _which_with_uv_fallback). The first element
    is always an absolute path when found via the fallback, so the
    resulting command is directly runnable even in a process whose PATH
    doesn't happen to include it.
    """
    for exe_name, suffix in (
        ("sleap-nn", ["track"]),
        ("sleap-nn-track", []),
        ("sleap", ["track"]),
        ("sleap-track", []),
    ):
        resolved = _which_with_uv_fallback(exe_name)
        if resolved is not None:
            return [resolved] + suffix
    return None


def _is_legacy_sleap_track(base_cmd) -> bool:
    """True if base_cmd resolved to the legacy single-command
    `sleap-track` CLI (no Labels-project/--video_index/--max_height/
    --max_width support) -- checked by basename rather than exact string
    equality, since base_cmd[0] may be an absolute path rather than the
    bare command name (see _find_track_base_cmd)."""
    return len(base_cmd) == 1 and Path(base_cmd[0]).name == "sleap-track"


def check_sleap_track_available():
    """Find a usable tracking command. Returns the base command list to
    build track invocations from."""
    base_cmd = _find_track_base_cmd()
    if base_cmd is None:
        raise SleapTrackNotFound(
            "None of `sleap-nn`, `sleap`, or `sleap-track` were found on "
            "PATH. Make sure you're running this in the same Python/conda "
            "environment where sleap-nn is installed (the one you trained "
            "the model in) -- e.g. `conda activate <your-sleap-env>` before "
            "running main.py."
        )
    return base_cmd


def sleap_install_instructions() -> str:
    """Copy-pasteable commands for getting a usable `sleap-nn` onto PATH,
    for the GUI's "SLEAP not found" warning (see main_window.py). Mirrors
    sleap-nn's own recommended install path (uv, not pip/conda) --
    https://nn.sleap.ai/dev/installation/ is the source of truth this is
    kept in sync with; re-check there if these commands stop working."""
    if sys.platform == "darwin":
        install_uv = "curl -LsSf https://astral.sh/uv/install.sh | sh"
        install_sleap = "uv tool install sleap-nn"
    elif sys.platform.startswith("win"):
        install_uv = 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
        install_sleap = "uv tool install sleap-nn --torch-backend auto"
    else:
        install_uv = "curl -LsSf https://astral.sh/uv/install.sh | sh"
        install_sleap = "uv tool install sleap-nn --torch-backend auto"

    return (
        "1. Install uv (skip if you already have it):\n"
        f"   {install_uv}\n\n"
        "2. Install sleap-nn (open a new terminal first if you just installed uv):\n"
        f"   {install_sleap}\n\n"
        "3. Verify it worked:\n"
        "   sleap-nn system\n\n"
        "Then restart this app so it picks up the new PATH."
    )


def _video_wrapper_slp_path(video_path, predictions_folder):
    video_path = Path(video_path)
    predictions_folder = Path(predictions_folder)
    return predictions_folder / "_video_wrappers" / f"{video_path.stem}.video_wrapper.slp"


def _ensure_video_wrapper_slp(video_path, predictions_folder):
    """Wrap a raw video file in a minimal single-video SLEAP Labels
    project (.slp) and return its path, (re)building it if the video is
    newer than the cached wrapper. Forces `grayscale=True` explicitly,
    since auto-detection is unreliable on lossy-compressed footage and
    this model was trained on grayscale input.
    """
    import sleap_io as sio

    video_path = Path(video_path)
    wrapper_path = _video_wrapper_slp_path(video_path, predictions_folder)
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)

    if wrapper_path.exists() and wrapper_path.stat().st_mtime >= video_path.stat().st_mtime:
        return wrapper_path

    video = sio.Video.from_filename(str(video_path), grayscale=True)
    labels = sio.Labels(videos=[video])
    sio.save_slp(labels, str(wrapper_path))
    return wrapper_path


def _build_track_cmd(
    base_cmd, data_path, model_paths, out_path,
    max_height=None, max_width=None, video_index=None,
):
    if _is_legacy_sleap_track(base_cmd):
        # Legacy CLI: video is a positional arg, no Labels-project/--video_index support.
        cmd = list(base_cmd) + [str(data_path)]
    else:
        cmd = list(base_cmd) + ["-i", str(data_path)]
        if video_index is not None:
            cmd += ["--video_index", str(video_index)]

    for model_path in model_paths:
        cmd += ["-m", str(model_path)]
    cmd += ["-o", str(out_path)]

    # Legacy sleap-track doesn't support these flags.
    if not _is_legacy_sleap_track(base_cmd):
        if max_height is not None:
            cmd += ["--max_height", str(max_height)]
        if max_width is not None:
            cmd += ["--max_width", str(max_width)]

    return cmd


def run_inference(
    video_path,
    model_paths,
    predictions_folder,
    base_cmd=None,
    overwrite=False,
    max_height=None,
    max_width=None,
    log=print,
    progress=None,
):
    """Run tracking on a single video. Returns the path to the .slp
    predictions file. Skips re-running if the output already exists,
    unless overwrite=True.

    `log` receives discrete one-off messages (the command being run;
    the full tool output if it fails -- see _ProgressStreamer). `progress`
    receives the tracking command's own live progress-bar output as it
    runs (see _stream_subprocess) -- repeated calls for what's
    conceptually the same redrawing line, as opposed to `log`'s discrete
    messages, so a caller that wants to show that in place (the GUI
    does) can tell the two apart. Defaults to `log` if not given, e.g.
    for the CLI's plain print()-based usage, where "in place" doesn't
    apply the same way.
    """
    if progress is None:
        progress = log
    video_path = Path(video_path)
    predictions_folder = Path(predictions_folder)
    predictions_folder.mkdir(parents=True, exist_ok=True)

    out_path = predictions_path_for(video_path, predictions_folder)
    if out_path.exists() and not overwrite:
        return out_path

    if overwrite and out_path.exists():
        out_path.unlink()

    if base_cmd is None:
        base_cmd = check_sleap_track_available()

    # Write to a temp path and rename only on success, so an interrupted
    # run doesn't leave a half-written file that looks "done" on resume.
    # Keep ".slp" as the actual suffix -- sleap_io infers format from it.
    tmp_path = out_path.with_name(out_path.stem + ".tmp.slp")
    if tmp_path.exists():
        tmp_path.unlink()

    if _is_legacy_sleap_track(base_cmd):
        # Legacy fallback CLI doesn't understand Labels-project wrapping.
        data_path = video_path
        video_index = None
    else:
        data_path = _ensure_video_wrapper_slp(video_path, predictions_folder)
        video_index = 0

    cmd = _build_track_cmd(
        base_cmd, data_path, model_paths, tmp_path,
        max_height=max_height, max_width=max_width, video_index=video_index,
    )

    log(f"  $ {' '.join(cmd)}")
    try:
        returncode, streamer = _stream_subprocess(cmd, log, progress)
    except BaseException:
        # Clean up any partial temp file, including on KeyboardInterrupt.
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    if returncode != 0:
        if tmp_path.exists():
            tmp_path.unlink()
        streamer.dump_plain_lines("run failed")
        raise RuntimeError(
            f"Tracking exited with code {returncode} for {video_path}. "
            f"See its output above for details."
        )

    if not tmp_path.exists():
        # The tool itself reported success, but there's still no file --
        # e.g. it silently declined to write to a cloud-synced folder
        # (Dropbox/iCloud/OneDrive) whose real-time sync can lock or
        # intercept writes out from under another process. That's
        # normally invisible: the '\n' output most likely to explain it
        # is exactly what's suppressed from the live log on a "successful"
        # run (see _ProgressStreamer) -- surface it now since this is the
        # one case where exit-code-0 output still matters.
        streamer.dump_plain_lines("no prediction file was produced")
        raise RuntimeError(
            f"Tracking reported success (exit code 0) for {video_path}, but "
            f"the expected output file was not created:\n  {tmp_path}\n"
            f"Check the output above for what actually happened -- if it's "
            f"empty, common causes are a bad -m model path (MODEL_PATHS in "
            f"config.py), an -o path it doesn't like, a video it couldn't "
            f"read, or -- if the input/output folders are inside "
            f"Dropbox/iCloud/OneDrive -- that cloud sync interfering with "
            f"the write. Try a job whose input and output folders are "
            f"fully local (not cloud-synced) to check."
        )

    tmp_path.replace(out_path)
    return out_path


def run_batch(
    video_paths,
    model_paths,
    predictions_folder,
    overwrite=False,
    on_progress=None,
    max_height=None,
    max_width=None,
):
    """Run tracking on a list of videos sequentially (unattended).
    on_progress(video_path, index, total) is called before each video, if given.
    """
    base_cmd = check_sleap_track_available()
    print(f"Using tracking command: {' '.join(base_cmd)}")

    out_paths = {}
    total = len(video_paths)
    for i, video_path in enumerate(video_paths, start=1):
        if on_progress:
            on_progress(video_path, i, total)
        out_paths[str(video_path)] = run_inference(
            video_path,
            model_paths,
            predictions_folder,
            base_cmd=base_cmd,
            overwrite=overwrite,
            max_height=max_height,
            max_width=max_width,
        )
    return out_paths
