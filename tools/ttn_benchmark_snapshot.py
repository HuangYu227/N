"""Pin a published training checkpoint AND all resume shards, without touching a job.

Use the same shared filesystem for source/output. Published checkpoint files and
UUID-named shards are immutable; hard links preserve them across last.pt replacement.
Evaluation-only snapshots lack optimizer/RNG shards and cannot benchmark exact resume.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import torch
import uuid


def benchmark_environment(snapshot):
    """Restore the saved training problem, never silently change data/scope."""
    from tools.ttn_slurm_chain import PROFILE
    snapshot = Path(snapshot)
    args = json.loads((snapshot / "run_config.json").read_text(encoding="utf-8"))["arguments"]
    payload = torch.load(snapshot / "last.pt", map_location="cpu", weights_only=False, mmap=True)
    # These are the benchmark's independent variable, not the saved training problem.
    execution = {"ttn_core_backend", "ttn_psi_backend"}
    profile = {env: str(args[name]) for name, env in PROFILE.items()
               if name not in execution and args.get(name) is not None}
    profile["TRAIN_SCOPE"] = payload.get("train_scope", "ttn")
    return profile


def snapshot_training_bundle(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    lock = source / ".checkpoint-save.lock"
    if output.exists(): raise FileExistsError(output)
    # Hold the same lock as publication/retention while pinning immutable files.
    from worldttn.checkpoint_integrity import acquire_checkpoint_lock, release_checkpoint_lock
    token = "benchmark-snapshot-" + uuid.uuid4().hex
    acquire_checkpoint_lock(lock, token)
    try:
        return _snapshot_training_bundle(source, output)
    finally:
        try: release_checkpoint_lock(lock, token)
        except OSError as error: print(f"[TTN snapshot] could not release save lock: {error}", flush=True)


def _snapshot_training_bundle(source, output):
    from worldttn.checkpoint_integrity import audit_checkpoint
    output.mkdir(parents=True, exist_ok=False)
    os.link(source / "last.pt", output / "last.pt")
    payload = torch.load(output / "last.pt", map_location="cpu", weights_only=False, mmap=True)
    meta = payload.get("distributed")
    if not meta: raise ValueError("benchmark requires a full distributed resume bundle")
    folder = meta["resume_dir"]
    if Path(folder).name != folder: raise ValueError("invalid resume directory")
    (output / folder).mkdir()
    for rank in range(meta["world_size"]):
        name = f"rank-{rank:05d}.pt"
        os.link(source / folder / name, output / folder / name)
    if meta.get("manifest"):
        for name in ("model.pt", "manifest.json"):
            os.link(source / folder / name, output / folder / name)
    audit = audit_checkpoint(output / "last.pt")
    run = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    record = None
    for line in (source / "train.jsonl").read_text(encoding="utf-8").splitlines():
        try: row = json.loads(line)
        except json.JSONDecodeError: continue
        if row.get("step") == payload["step"]: record = row; break
    if record is None or record["stage"] != payload["stage"] or run["base"]["sha256"] != payload["base_sha256"]:
        raise ValueError("pinned weights and completed training records disagree")
    from worldttn.evaluation import file_sha256
    (output / "run_config.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    (output / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest = {"source": str(source), "purpose": "independent exact-resume benchmark", "step": payload["step"],
                "scope": payload.get("train_scope"), "checkpoint_sha256": file_sha256(output / "last.pt"),
                "resume_dir": folder, "world_size": meta["world_size"], "integrity": audit["status"]}
    (output / "benchmark_snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run")
    parser.add_argument("--output")
    parser.add_argument("--print-env", help="read-only: emit shell-quoted saved training profile for the benchmark launcher")
    args = parser.parse_args()
    if args.print_env:
        if args.training_run or args.output: parser.error("--print-env is separate from snapshot creation")
        for key, value in benchmark_environment(args.print_env).items(): print(f"export {key}={shlex.quote(value)}")
        return
    if not args.training_run or not args.output: parser.error("snapshot creation requires --training-run and --output")
    print(snapshot_training_bundle(args.training_run, args.output), flush=True)


if __name__ == "__main__": main()
