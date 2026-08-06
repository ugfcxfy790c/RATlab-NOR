"""
Job queue data model for the NOR classifier GUI.

A "Job" is one group's worth of work: a folder of videos to score, an
output destination, and per-job overrides of the pipeline settings that
currently live in config.py. Jobs are queued in the GUI (object-coordinate
setup happens up front, per job, before the batch is started), then run
one at a time -- e.g. overnight -- by the batch runner (see task #5).

This module owns:
  - the Job dataclass + JobStatus enum
  - persistence of the queue to app_data/queue.json (so the GUI can be
    closed and reopened without losing a half-built queue)
  - per-job paths for object coordinates and SLEAP prediction caching,
    namespaced by job id so two jobs can't collide even if their video
    folders contain same-named files
  - the "is this job ready to run" check (all videos have object coords)

No GUI or pipeline code lives here -- this is pure data/state so it can be
unit tested and reused by both the GUI and the batch runner.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# App data location
# ---------------------------------------------------------------------------

def default_app_data_dir(nor_classifier_dir: Path) -> Path:
    """Where the GUI stores its own state: the queue file, per-job object
    coordinates, and per-job prediction caches. Lives alongside the
    existing nor_classifier install so it moves/copies with it, rather
    than in a user Library folder the person will never look in."""
    return Path(nor_classifier_dir) / "app_data"


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    NEEDS_SETUP = "needs_setup"   # one or more videos still need object coords clicked
    READY = "ready"                # fully set up, waiting in the queue
    RUNNING = "running"            # batch runner is actively processing this job
    DONE = "done"                  # finished with no errors
    FAILED = "failed"              # hit an error; batch runner blocked it and moved on
    CANCELED = "canceled"          # user pulled it before/while it ran


# Terminal statuses a job can be safely re-queued from.
RERUNNABLE_STATUSES = {JobStatus.FAILED, JobStatus.DONE, JobStatus.CANCELED}


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
    return slug or "group"


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

@dataclass
class Job:
    group_name: str
    input_folder: str
    output_base_folder: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    skip_validation: bool = False
    # Per-job overrides of config.py values. Empty dict = use current
    # config.py defaults at run time. Keys are config attribute names,
    # e.g. {"SNIFF_CONE_HALF_ANGLE_DEG": 30.0}.
    config_overrides: dict = field(default_factory=dict)
    status: JobStatus = JobStatus.NEEDS_SETUP
    error_message: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    # Video-level progress, filled in by the batch runner as it works
    # through the job so the GUI can show "3/12 videos" while running.
    videos_total: int = 0
    videos_done: int = 0

    @property
    def slug(self) -> str:
        """Filesystem-safe version of the group name, used for the per-job
        output folder and as a namespace for cached app_data."""
        return _slugify(self.group_name)

    @property
    def output_folder(self) -> Path:
        """Where this job's bundled Excel + validation videos land:
        <output_base_folder>/<group name>/"""
        return Path(self.output_base_folder) / self.group_name

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        d = dict(d)
        d["status"] = JobStatus(d.get("status", "needs_setup"))
        return cls(**d)


# ---------------------------------------------------------------------------
# Per-job paths (namespaced by job.id so jobs never collide)
# ---------------------------------------------------------------------------

def job_coords_path(job: Job, app_data_dir: Path) -> Path:
    """Object-coordinate cache for this job alone. Separate per job (rather
    than one shared object_coords.json) because two jobs can point at
    folders with same-named video files, and video_key() is only unique
    relative to a single video folder."""
    return Path(app_data_dir) / "coords" / f"{job.id}_{job.slug}.json"


def job_predictions_folder(job: Job, app_data_dir: Path) -> Path:
    """SLEAP .slp prediction cache for this job alone -- same collision
    reasoning as job_coords_path."""
    return Path(app_data_dir) / "predictions" / f"{job.id}_{job.slug}"


# ---------------------------------------------------------------------------
# Readiness check
# ---------------------------------------------------------------------------

def find_videos(folder) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(folder.rglob("*.mp4"))


def refresh_job_readiness(job: Job, app_data_dir: Path, object_picker_module) -> Job:
    """Recompute whether `job` has object coordinates set for every video
    in its input folder. Mutates and returns job.status; does not touch
    RUNNING/DONE/FAILED/CANCELED jobs (those are runner-owned states)."""
    if job.status not in (JobStatus.NEEDS_SETUP, JobStatus.READY):
        return job

    videos = find_videos(job.input_folder)
    job.videos_total = len(videos)

    if not videos:
        job.status = JobStatus.NEEDS_SETUP
        return job

    coords_path = job_coords_path(job, app_data_dir)
    coords = object_picker_module.load_object_coords(coords_path)
    missing = [
        v for v in videos
        if object_picker_module.video_key(v, job.input_folder) not in coords
    ]
    job.status = JobStatus.NEEDS_SETUP if missing else JobStatus.READY
    return job


# ---------------------------------------------------------------------------
# Queue persistence
# ---------------------------------------------------------------------------

class JobQueue:
    """Ordered list of jobs, persisted to app_data/queue.json. Add-order is
    run order -- no reordering/prioritization (by design, keeps this and
    the GUI simple)."""

    def __init__(self, app_data_dir: Path):
        self.app_data_dir = Path(app_data_dir)
        self.path = self.app_data_dir / "queue.json"
        self.jobs: list[Job] = []

    # -- persistence --

    def load(self) -> "JobQueue":
        if self.path.exists():
            with open(self.path) as f:
                raw = json.load(f)
            self.jobs = [Job.from_dict(j) for j in raw.get("jobs", [])]
        else:
            self.jobs = []
        self._recover_stale_running_jobs()
        return self

    def _recover_stale_running_jobs(self) -> None:
        """A job can be left with status=RUNNING in queue.json if the app
        (or the batch worker process) was killed or crashed mid-run
        instead of finishing normally -- nothing is actually processing
        it by the time we're loading this file fresh on startup, so it
        would otherwise show as permanently "Running" with no way to
        stop or re-run it. Treat it as failed instead."""
        changed = False
        for job in self.jobs:
            if job.status == JobStatus.RUNNING:
                job.status = JobStatus.FAILED
                job.error_message = "Interrupted -- the app closed or crashed while this job was running."
                changed = True
        if changed:
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"jobs": [j.to_dict() for j in self.jobs]}, f, indent=2)
        tmp.replace(self.path)

    # -- CRUD --

    def add(self, job: Job) -> Job:
        self.jobs.append(job)
        self.save()
        return job

    def remove(self, job_id: str) -> None:
        self.jobs = [j for j in self.jobs if j.id != job_id]
        self.save()

    def get(self, job_id: str) -> Optional[Job]:
        return next((j for j in self.jobs if j.id == job_id), None)

    def update(self, job: Job) -> None:
        for i, j in enumerate(self.jobs):
            if j.id == job.id:
                self.jobs[i] = job
                break
        self.save()

    # -- queries the batch runner needs --

    def runnable_jobs(self) -> list[Job]:
        """Jobs in add-order that are ready to be picked up by an
        overnight run. Jobs still needing setup are left behind -- per the
        design decision that setup happens interactively, up front, not
        mid-run."""
        return [j for j in self.jobs if j.status == JobStatus.READY]

    def needs_setup_jobs(self) -> list[Job]:
        return [j for j in self.jobs if j.status == JobStatus.NEEDS_SETUP]
