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


def _comparison(a, b, name):
    try:
        identical(a, b, name)
        return {"equal": True}
    except AssertionError as error:
        return {"equal": False, "first_difference": str(error)}


def _run_records(path):
    folder = Path(path).parent
    config = folder / "run_config.json"
    run = json.loads(config.read_text(encoding="utf-8")) if config.exists() else {}
    records = []
    log = folder / "train.jsonl"
    if log.exists():
        with log.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                records.append({key: row.get(key) for key in ("step", "loss", "outer_grad_norm")})
    return {"initialization": run.get("initialization"), "precision": run.get("precision"),
            "launch": run.get("launch"), "implementation": run.get("implementation"), "steps": records}


def diagnose(continuous, resumed):
    """Read existing files without repeating SHA reads; this is never an acceptance result."""
    paths = [Path(continuous), Path(resumed)]
    a, b = [torch.load(path, map_location="cpu", weights_only=False, mmap=True) for path in paths]
    fields = ("step", "stage", "config", "base_sha256", "train_scope", "weight_scope", "optimizer_policy")
    report = {"status": "diagnostic_only", "integrity": "not rechecked; use the default comparison for acceptance",
              "metadata": {field: _comparison(a[field], b[field], field) for field in fields},
              "distributed": {field: _comparison(a["distributed"][field], b["distributed"][field], field)
                              for field in ("mode", "world_size", "training_config")},
              "model": _comparison(a["adapter"], b["adapter"], "adapter"), "ranks": []}
    for name in a["adapter"]:
        if name not in b["adapter"]: continue
        x, y = a["adapter"][name], b["adapter"][name]
        if x.shape == y.shape and x.dtype == y.dtype and not torch.equal(x, y):
            delta = y.double() - x.double()
            norm = x.double().norm().item()
            report["model_scale_example"] = {"name": name, "max_abs_difference": delta.abs().max().item(),
                                              "relative_l2": delta.norm().item() / norm if norm else None}
            break
    folders = [path.parent / payload["distributed"]["resume_dir"] for path, payload in zip(paths, (a, b))]
    if a["distributed"]["world_size"] != b["distributed"]["world_size"]:
        return report
    for rank in range(a["distributed"]["world_size"]):
        left, right = [torch.load(folder / f"rank-{rank:05d}.pt", map_location="cpu", weights_only=False, mmap=True)
                       for folder in folders]
        report["ranks"].append({"rank": rank,
            "rng": {field: _comparison(left["rng"].get(field), right["rng"].get(field), f"rank-{rank}.rng.{field}")
                    for field in sorted(left["rng"].keys() | right["rng"].keys())},
            "data": {**_comparison(left["data"], right["data"], f"rank-{rank}.data"),
                     "continuous": left["data"], "resumed": right["data"]},
            "optimizer": _comparison(left["optimizer"], right["optimizer"], f"rank-{rank}.optimizer"),
            "optimizer_param_groups": _comparison(left["optimizer"]["param_groups"], right["optimizer"]["param_groups"],
                                                   f"rank-{rank}.optimizer.param_groups") if left["optimizer"] is not None and right["optimizer"] is not None else None,
            "optimizer_parameter_names": _comparison(left["optimizer_parameter_names"], right["optimizer_parameter_names"],
                                                      f"rank-{rank}.optimizer_parameter_names")})
    report["runs"] = [_run_records(path) for path in paths]
    initial = (report["runs"][1].get("initialization") or {}).get("checkpoint")
    if initial and Path(initial).exists(): report["source_run"] = _run_records(initial)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("continuous")
    parser.add_argument("resumed")
    parser.add_argument("--diagnose", action="store_true", help="report RNG/cursor/Adam and run logs after a failed comparison; no acceptance or repeated SHA audit")
    args = parser.parse_args()
    operation = diagnose if args.diagnose else compare
    print(json.dumps(operation(args.continuous, args.resumed), indent=2), flush=True)


if __name__ == "__main__": main()
