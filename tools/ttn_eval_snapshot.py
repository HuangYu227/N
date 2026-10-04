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


def snapshot_training_run(source):
    import torch

    source = Path(source).resolve()
    run = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    output = source.parent / f"eval-snapshot-{uuid.uuid4().hex[:12]}"
    output.mkdir(exist_ok=False)
    # Keep the snapshot on the source filesystem: linking is atomic and needs
    # neither a multi-GB copy nor another full checkpoint in CPU RAM.
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
    (output / "snapshot.json").write_text(json.dumps({"source": str(source), "step": step,
                                                     "purpose": "evaluation only"}, indent=2) + "\n", encoding="utf-8")
    return output


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
