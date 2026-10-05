"""Read-only exact comparison of independent uninterrupted/resumed bundles."""
import argparse
import json
from pathlib import Path
import torch
from worldttn.checkpoint_integrity import audit_checkpoint


def identical(a, b, name):
    assert type(a) is type(b), f"{name}: type differs"
    if isinstance(a, dict):
        assert a.keys() == b.keys(), f"{name}: keys differ"
        for key in a: identical(a[key], b[key], f"{name}.{key}")
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), f"{name}: lengths differ"
        for i, (x, y) in enumerate(zip(a, b)): identical(x, y, f"{name}.{i}")
    elif isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=lambda message: f"{name}: {message}")
    elif hasattr(a, "shape") and hasattr(a, "dtype"):
        assert a.shape == b.shape and a.dtype == b.dtype and (a == b).all(), f"{name}: array differs"
    else:
        assert a == b, f"{name}: value differs"


def compare(continuous, resumed):
    paths = [Path(continuous), Path(resumed)]
    audits = [audit_checkpoint(path) for path in paths]
    a, b = [torch.load(path, map_location="cpu", weights_only=False, mmap=True) for path in paths]
    for field in ("step", "stage", "config", "base_sha256", "train_scope", "weight_scope", "optimizer_policy", "adapter"):
        identical(a[field], b[field], field)
    for field in ("mode", "world_size", "training_config"):
        identical(a["distributed"][field], b["distributed"][field], field)
    folders = [path.parent / payload["distributed"]["resume_dir"] for path, payload in zip(paths, (a, b))]
    for rank in range(a["distributed"]["world_size"]):
        left, right = [torch.load(folder / f"rank-{rank:05d}.pt", map_location="cpu", weights_only=False, mmap=True)
                       for folder in folders]
        for field in ("optimizer", "rng", "data", "optimizer_parameter_names"):
            identical(left[field], right[field], f"rank-{rank}.{field}")
    return {"status": "exact_match", "step": a["step"], "world_size": a["distributed"]["world_size"],
            "paths": [str(path.resolve()) for path in paths], "integrity": [r["status"] for r in audits],
            "scope": "model, Adam, RNG and data cursor; independent checkpoint IDs/timestamps excluded"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("continuous")
    parser.add_argument("resumed")
    args = parser.parse_args()
    print(json.dumps(compare(args.continuous, args.resumed), indent=2), flush=True)


if __name__ == "__main__": main()
