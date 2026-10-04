"""Pin a published training checkpoint AND all resume shards, without touching a job.

Use the same shared filesystem for source/output. Published checkpoint files and
UUID-named shards are immutable; hard links preserve them across last.pt replacement.
Evaluation-only snapshots lack optimizer/RNG shards and cannot benchmark exact resume.
"""
import argparse
import json
import os
from pathlib import Path
import torch


def snapshot_training_bundle(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
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
                "resume_dir": folder, "world_size": meta["world_size"]}
    (output / "benchmark_snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(snapshot_training_bundle(args.training_run, args.output), flush=True)


if __name__ == "__main__": main()
