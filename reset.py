"""
Clears cached/generated data so you can re-run the pipeline from scratch --
handy for testing.

Two file structures exist now:

  1. The legacy single-folder CLI workflow (`main.py setup`/`run`) --
     config.py's PREDICTIONS_FOLDER/OUTPUT_FOLDER/OBJECT_COORDS_FILE.
  2. The GUI's per-job job queue (nor_classifier/app_data/) -- each job
     in queue.json has its own coordinate cache, prediction cache, and
     output folder (job_queue.py's job_coords_path()/
     job_predictions_folder()/Job.output_folder).

Usage:
    python reset.py                     # clear legacy predictions/ and output/
    python reset.py --output-only       # clear ONLY output/ -- keeps cached
                                         # .slp predictions, so a re-run
                                         # reuses them instead of re-tracking
                                         # (e.g. after changing a scoring
                                         # constant like MERGE_GAP_S, which
                                         # doesn't require new inference)
    python reset.py --coords            # clear ONLY legacy object_coords.json
    python reset.py --all               # clear legacy predictions/, output/, AND coords
    python reset.py --yes               # skip the confirmation prompt

    python reset.py --job "Cohort 1"    # target one GUI job by group name
                                         # instead of the legacy folders --
                                         # same --output-only/--coords/--all/--yes
                                         # flags apply
    python reset.py --list-jobs         # list GUI job group names and exit

By default this does NOT touch object coordinates, since re-clicking
object positions is the slow part -- pass --coords if you specifically
want to wipe just that, or --all to also start predictions/output over
from nothing. Same logic applies to --job.
"""

import argparse
import shutil
import sys
from pathlib import Path

import config

# job_queue.py is pure Python (no PySide6 import), so it's safe/cheap to
# use from this CLI-only script -- just needs the gui/ folder on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent / "gui"))
from job_queue import JobQueue, default_app_data_dir, job_coords_path, job_predictions_folder  # noqa: E402


def _rmtree_contents(folder):
    folder = Path(folder)
    if not folder.exists():
        return 0
    count = 0
    for child in folder.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        count += 1
    return count


# --- legacy single-folder mode ---------------------------------------------

def describe_targets(cfg, clear_predictions=False, clear_output=False, clear_coords=False):
    """Human-readable list of what perform_reset() would touch, for
    confirmation prompts/dialogs -- doesn't delete anything."""
    lines = []
    if clear_predictions:
        lines.append(str(cfg.PREDICTIONS_FOLDER))
    if clear_output:
        lines.append(str(cfg.OUTPUT_FOLDER))
    if clear_coords:
        lines.append(f"{cfg.OBJECT_COORDS_FILE}  (all clicked novel/original positions)")
    return lines


def perform_reset(cfg, clear_predictions=False, clear_output=False, clear_coords=False, log=print):
    """Does the actual deleting for the legacy single-folder workflow.
    `log` is called with progress strings -- pass a GUI log function
    instead of the default print() to route output somewhere other than
    stdout. Predictions and output are cleared independently -- e.g.
    clear_output alone leaves cached .slp predictions in place, so a
    re-run picks them up instead of re-tracking (useful after changing a
    scoring-only constant like MERGE_GAP_S)."""
    total = 0
    targets = []
    if clear_predictions:
        targets.append(cfg.PREDICTIONS_FOLDER)
    if clear_output:
        targets.append(cfg.OUTPUT_FOLDER)
    for target in targets:
        n = _rmtree_contents(target)
        total += n
        log(f"Cleared {n} item(s) from {target}")

    if clear_coords and cfg.OBJECT_COORDS_FILE.exists():
        cfg.OBJECT_COORDS_FILE.unlink()
        log(f"Deleted {cfg.OBJECT_COORDS_FILE}")

    return total


# --- per-job (GUI queue) mode -----------------------------------------------

def describe_job_targets(job, app_data_dir, clear_predictions=False, clear_output=False, clear_coords=False):
    """Same idea as describe_targets(), but for one GUI job -- its own
    prediction cache, output folder, and coordinate cache under
    app_data_dir, rather than the legacy global folders."""
    lines = []
    if clear_predictions:
        lines.append(str(job_predictions_folder(job, app_data_dir)))
    if clear_output:
        lines.append(str(job.output_folder))
    if clear_coords:
        lines.append(
            f"{job_coords_path(job, app_data_dir)}  "
            f"(all clicked novel/original positions for '{job.group_name}')"
        )
    return lines


def perform_job_reset(job, app_data_dir, clear_predictions=False, clear_output=False, clear_coords=False, log=print):
    """Clears one GUI job's own cache/output -- does not touch any other
    job, and does not touch the legacy global folders at all. Predictions
    and output are cleared independently -- see perform_reset()'s
    docstring for why that matters."""
    total = 0
    targets = []
    if clear_predictions:
        targets.append(job_predictions_folder(job, app_data_dir))
    if clear_output:
        targets.append(job.output_folder)
    for target in targets:
        n = _rmtree_contents(target)
        total += n
        log(f"Cleared {n} item(s) from {target}")

    if clear_coords:
        coords_path = job_coords_path(job, app_data_dir)
        if coords_path.exists():
            coords_path.unlink()
            log(f"Deleted {coords_path}")

    return total


def _find_job_by_group_name(app_data_dir, group_name):
    queue = JobQueue(app_data_dir).load()
    matches = [j for j in queue.jobs if j.group_name == group_name]
    if not matches:
        available = ", ".join(repr(j.group_name) for j in queue.jobs) or "(no jobs queued)"
        raise SystemExit(f"No job named {group_name!r} found in {queue.path}.\nAvailable: {available}")
    if len(matches) > 1:
        raise SystemExit(f"Multiple jobs are named {group_name!r} -- this shouldn't happen (job names should be unique).")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-only", action="store_true", help="clear ONLY output/ -- keeps cached predictions, so a re-run reuses them instead of re-tracking")
    parser.add_argument("--coords", action="store_true", help="clear ONLY object coordinates (leaves predictions/output alone)")
    parser.add_argument("--all", action="store_true", help="clear predictions/output AND object coordinates")
    parser.add_argument("--yes", "-y", action="store_true", help="skip confirmation prompt")
    parser.add_argument("--job", metavar="GROUP_NAME", help="target this GUI job (by group name) instead of the legacy global folders")
    parser.add_argument("--list-jobs", action="store_true", help="list GUI job group names and exit")
    args = parser.parse_args()

    app_data_dir = default_app_data_dir(config.NOR_CLASSIFIER_DIR)

    if args.list_jobs:
        queue = JobQueue(app_data_dir).load()
        if not queue.jobs:
            print(f"No jobs queued ({queue.path}).")
        else:
            for j in queue.jobs:
                print(f"  {j.group_name!r}  ({j.status.value})")
        return

    exclusive_flags = [f for f in (args.output_only, args.coords, args.all) if f]
    if len(exclusive_flags) > 1:
        parser.error("--output-only, --coords, and --all are mutually exclusive -- pick one")

    if args.output_only:
        clear_predictions, clear_output, clear_coords = False, True, False
    elif args.coords:
        clear_predictions, clear_output, clear_coords = False, False, True
    elif args.all:
        clear_predictions, clear_output, clear_coords = True, True, True
    else:
        clear_predictions, clear_output, clear_coords = True, True, False

    if args.job:
        job = _find_job_by_group_name(app_data_dir, args.job)
        targets = describe_job_targets(job, app_data_dir, clear_predictions, clear_output, clear_coords)
        print(f"This will delete, for job '{job.group_name}':")
    else:
        targets = describe_targets(config, clear_predictions, clear_output, clear_coords)
        print("This will delete:")

    for line in targets:
        print(f"  - {line}")
    if not clear_coords:
        print("  (leaving object coordinates alone -- pass --coords or --all to clear those too)")

    if not args.yes:
        answer = input("\nProceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Cancelled.")
            sys.exit(1)

    if args.job:
        perform_job_reset(job, app_data_dir, clear_predictions, clear_output, clear_coords)
    else:
        perform_reset(config, clear_predictions, clear_output, clear_coords)
        print(f"\nDone. Run `python main.py {'setup' if clear_coords else 'run'}` to start fresh.")


if __name__ == "__main__":
    main()
