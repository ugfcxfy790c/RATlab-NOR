"""
Qt table model wrapping a JobQueue, so the main window's table view stays
in sync with the underlying job list without duplicating state.
"""

from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex

from job_queue import JobStatus

COLUMNS = ["Group", "Status", "Progress", "Input Folder", "Output Folder"]

_STATUS_LABELS = {
    JobStatus.NEEDS_SETUP: "Needs setup",
    JobStatus.READY: "Ready",
    JobStatus.RUNNING: "Running…",
    JobStatus.DONE: "Done",
    JobStatus.FAILED: "Failed",
    JobStatus.CANCELED: "Canceled",
}


class JobTableModel(QAbstractTableModel):
    def __init__(self, job_queue, parent=None):
        super().__init__(parent)
        self.job_queue = job_queue

    # -- required QAbstractTableModel overrides --

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.job_queue.jobs)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole or orientation != Qt.Horizontal:
            return None
        return COLUMNS[section]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        job = self.job_queue.jobs[index.row()]
        col = index.column()

        if role == Qt.DisplayRole:
            if col == 0:
                return job.group_name
            if col == 1:
                label = _STATUS_LABELS.get(job.status, job.status.value)
                return f"{label}  ({job.error_message})" if job.status == JobStatus.FAILED and job.error_message else label
            if col == 2:
                if job.videos_total:
                    return f"{job.videos_done}/{job.videos_total}"
                return "—"
            if col == 3:
                return job.input_folder
            if col == 4:
                return str(job.output_folder)

        if role == Qt.ForegroundRole:
            from PySide6.QtGui import QColor
            if col == 1:
                if job.status == JobStatus.FAILED:
                    return QColor("#c0392b")
                if job.status == JobStatus.DONE:
                    return QColor("#2e7d32")
                if job.status == JobStatus.RUNNING:
                    return QColor("#1565c0")
                if job.status == JobStatus.NEEDS_SETUP:
                    return QColor("#8a6d00")

        return None

    # -- helpers for the main window --

    def job_at(self, row) -> "Job":
        return self.job_queue.jobs[row]

    def refresh(self):
        """Call after the underlying job_queue.jobs list or any job's
        fields change, to repaint the view."""
        self.beginResetModel()
        self.endResetModel()
