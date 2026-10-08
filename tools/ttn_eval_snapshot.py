"""Freeze a completed checkpoint from an active training run for paired evaluation.

The trainer publishes last.pt with os.replace. A hard link pins the completed
inode even when the next checkpoint is published. No optimizer/RNG shards are
needed for evaluation; this directory is not an exact-training-resume bundle.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import uuid


def expected_evaluation_children(run):
    training = run.get("training", {})
    config = training.get("meta_ttt", training.get("ttn", {}))
    if config.get("memory_update") == "proximal": return {"long", "short", "align"}
    return {"long", "short", "align"} | ({"no-local", "no-persistent"}
        if config.get("local_update", False) or config.get("persistent_meta", False) else set())


def snapshot_training_run(source, *, retain_model=True):
    source = Path(source).resolve()
    run = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    output = source.parent / f"eval-snapshot-{uuid.uuid4().hex[:12]}"
    output.mkdir(exist_ok=False)
    # Keep the snapshot on the source filesystem: linking is atomic and needs
    # neither a multi-GB copy nor another full checkpoint in CPU RAM.
    try:
        return _write_snapshot(source, output, run, retain_model)
    except Exception:
        for name in ("last.pt", "run_config.json", "train.jsonl", "snapshot.json"):
            try: (output / name).unlink(missing_ok=True)
            except OSError: pass
        try: output.rmdir()
        except OSError: pass
        raise


def _write_snapshot(source, output, run, retain_model):
    import torch
    os.link(source / "last.pt", output / "last.pt")
    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False, mmap=True)
    step = int(checkpoint["step"])
    record = None
    with (source / "train.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if candidate.get("step") == step:
                record = candidate
                break
    if record is None:
        raise ValueError(f"No completed training record for pinned checkpoint step {step}")
    if record["stage"] != checkpoint["stage"] or run["base"]["sha256"] != checkpoint["base_sha256"]:
        raise ValueError("Pinned checkpoint and training records have different identities")
    (output / "run_config.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    (output / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    stat = (output / "last.pt").stat()
    (output / "snapshot.json").write_text(json.dumps({"format": "TTN-evaluation-snapshot-v2",
        "source": str(source), "step": step, "purpose": "evaluation only", "retain_model": retain_model,
        "model_identity": [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]}, indent=2) + "\n", encoding="utf-8")
    return output


def release_evaluation_model(snapshot, result):
    """Release only an opted-in snapshot alias after all evaluation children finish."""
    from worldttn.checkpoint_integrity import atomic_json
    snapshot = Path(snapshot)
    if snapshot.is_symlink(): raise ValueError("evaluation snapshot cannot be a symbolic link")
    snapshot = snapshot.resolve()
    marker = snapshot / "snapshot.json"
    if not marker.is_file(): return {"status": "unmanaged"}
    meta = json.loads(marker.read_text(encoding="utf-8"))
    if (meta.get("format") != "TTN-evaluation-snapshot-v2" or meta.get("retain_model") is not False
            or (snapshot / ".keep").exists()):
        return {"status": "retained"}
    run = json.loads((snapshot / "run_config.json").read_text(encoding="utf-8"))
    if (result.get("status") != "completed" or set(result.get("results", {})) != expected_evaluation_children(run)
            or result.get("identity", {}).get("step") != meta["step"]
            or Path(result.get("training_run", "")).resolve() != snapshot):
        raise ValueError("evaluation must complete against this exact snapshot before releasing its model")
    source = Path(meta["source"]).resolve()
    if snapshot.parent != source.parent or not snapshot.name.startswith("eval-snapshot-"):
        raise ValueError("unmanaged evaluation snapshot location")
    model = snapshot / "last.pt"
    if model.is_symlink(): raise ValueError("evaluation model alias cannot be a symbolic link")
    if model.exists():
        stat = model.stat()
        if [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns] != meta["model_identity"]:
            raise ValueError("evaluation model alias changed; refusing release")
        model.unlink()
    meta["model_released"] = True
    atomic_json(meta, marker)
    return {"status": "released", "step": meta["step"], "metrics_and_logs_retained": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True)
    args = parser.parse_args()
    try:
        output = snapshot_training_run(args.training_run)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Evaluation snapshot failed: {error}", file=sys.stderr)
        return 1
    print(output, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
