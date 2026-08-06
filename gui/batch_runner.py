"""
GUI-side handle for a batch run. Delegates the actual pipeline work to a
separate OS process (batch_worker_process.run_batch_worker) rather than
a thread inside this process -- see that module's docstring for why
(short version: OpenCV video I/O + an active Qt/Cocoa event loop in the
same process, even on different threads, is a real macOS segfault risk;
a separate process has no native state in common with the GUI at all).

Exposes the same signal interface the old QThread-based version did
(log/job_started/job_progress/job_finished/all_finished/finished), so
main_window.py's wiring didn't need to change -- only how this class is
constructed and started.
"""

from __future__ import annotations

import multiprocessing as mp

from PySide6.QtCore import QObject, QTimer, Signal

from batch_worker_process import run_batch_worker


class BatchRunner(QObject):
    log = Signal(str)
    progress = Signal(str)  # same redrawing line updating -- see main_window.py's _on_progress_line
    job_started = Signal(str)
    job_progress = Signal(str, int, int)
    job_finished = Signal(str, str, str)
    all_finished = Signal()
    finished = Signal()  # fires once the worker process has fully exited

    def __init__(self, jobs, app_data_dir, nor_classifier_dir, parent=None):
        super().__init__(parent)
        self._job_dicts = [j.to_dict() for j in jobs]
        self._app_data_dir = str(app_data_dir)
        self._nor_classifier_dir = str(nor_classifier_dir)
        self._stop_event = mp.Event()
        self._queue = mp.Queue()
        self._process: mp.Process | None = None
        self._saw_all_finished = False

        self._timer = QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._poll)

    def start(self):
        self._process = mp.Process(
            target=run_batch_worker,
            args=(self._nor_classifier_dir, self._job_dicts, self._app_data_dir, self._queue, self._stop_event),
            daemon=True,
        )
        self._process.start()
        self._timer.start()

    def request_stop(self):
        self._stop_event.set()

    def wait(self, timeout_ms: int = 5000):
        """Mirrors QThread.wait() -- used by MainWindow.closeEvent."""
        if self._process is not None:
            self._process.join(timeout=timeout_ms / 1000)

    # -- internal --

    def _poll(self):
        while True:
            try:
                msg = self._queue.get_nowait()
            except Exception:
                break
            self._dispatch(msg)

        if self._process is not None and not self._process.is_alive():
            if not self._saw_all_finished:
                self.log.emit("Batch worker process exited unexpectedly.")
            self._timer.stop()
            self._process.join(timeout=2)
            self.finished.emit()

    def _dispatch(self, msg):
        kind = msg[0]
        if kind == "log":
            self.log.emit(msg[1])
        elif kind == "progress":
            self.progress.emit(msg[1])
        elif kind == "job_started":
            self.job_started.emit(msg[1])
        elif kind == "job_progress":
            self.job_progress.emit(msg[1], msg[2], msg[3])
        elif kind == "job_finished":
            self.job_finished.emit(msg[1], msg[2], msg[3])
        elif kind == "all_finished":
            self._saw_all_finished = True
            self.all_finished.emit()
