"""Best-effort first-error evidence; never replace the original exception."""
import os
from pathlib import Path
import sys
import time
import traceback
from .checkpoint_integrity import atomic_json


def record_failure(output, phase, error, *, rank=None, step=None):
    rank = rank if rank is not None else os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))
    job = os.environ.get("SLURM_JOB_ID", "local")
    path = Path(output) / f"failure-job{job}-rank{rank}-pid{os.getpid()}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            atomic_json({"job": job, "rank": rank, "pid": os.getpid(), "step": step, "phase": phase,
                         "time_ns": time.time_ns(), "exception": type(error).__name__, "message": str(error),
                         "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))}, path)
    except Exception as diagnostic_error:
        print(f"[TTN failure evidence] could not write {path}: {diagnostic_error}", file=sys.stderr, flush=True)
