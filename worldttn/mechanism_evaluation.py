"""Collect same-checkpoint interventions. These are diagnostic, not retrained models."""
import argparse
import json
from pathlib import Path

from .evaluation import paired_summary


VARIANTS = {
    "full": ("full", "generated", ["sana", "ttn"]),
    "no-ttt": ("no-ttt", "generated", ["ttn"]),
    "identity": ("identity", "generated", ["ttn"]),
    "gt-history": ("full", "gt", ["sana", "ttn"]),
}
META_VARIANTS = {**VARIANTS, "no-local": ("no-local", "generated", ["ttn"]),
                 "no-persistent": ("no-persistent", "generated", ["ttn"])}
IDENTITY = ("stage", "step", "checkpoint_sha256", "fixed_cases_sha256", "frames", "noise_frames",
            "steps", "cfg_scale", "cached_blocks", "flow_shift", "ttn_camera_attention",
            "cross_attn_backend", "training_latent_frames", "provenance")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def collect_mechanisms(output):
    output = Path(output)
    summaries, statuses = {}, {}
    identity = None
    full = output / "full/summary.json"
    meta = json.loads(full.read_text())["protocol"].get("meta_ttt", {}) if full.is_file() else {}
    plan_path = output / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.is_file() else {}
    names = plan.get("variants", list(META_VARIANTS if any(meta.values()) else VARIANTS))
    if names not in (list(VARIANTS), list(META_VARIANTS)):
        raise ValueError("unexpected mechanism variants")
    variants = {name: META_VARIANTS[name] for name in names}
    for name, (ablation, history, methods) in variants.items():
        folder = output / name
        status = output / f"{name}-status.json"
        statuses[name] = json.loads(status.read_text()) if status.is_file() else {"status": "pending"}
        path = folder / "summary.json"
        if not path.is_file(): continue
        summary = json.loads(path.read_text())
        protocol = summary["protocol"]
        if (protocol["ttn_ablation"], protocol["history_source"], protocol["eval_methods"]) != (ablation, history, methods):
            raise ValueError(f"{name}: wrong runtime intervention/history/methods")
        if not protocol["state_diagnostics"] or protocol["stage"] != "C":
            raise ValueError("mechanism comparisons require C checkpoints and state diagnostics")
        current = {key: protocol[key] for key in IDENTITY}
        current["meta_ttt"] = protocol.get("meta_ttt", {"local_update": False, "persistent_meta": False})
        if identity is not None and current != identity:
            raise ValueError("mechanism runs used different checkpoints/cases/sampling/source")
        identity = current
        summaries[name] = summary
        statuses[name] = {"status": "completed", "path": str(path)}
    result = {"status": "running", "identity": identity, "variants": statuses,
              "interpretation": "same-weight runtime interventions; single-case results are not causal proof or a benchmark",
              "prefix_protocol": "13-frame metrics are the causal prefix of each 61-frame rollout",
              "results": {}, "contrasts": {}}
    if any(s["status"] == "failed" for s in statuses.values()): result["status"] = "failed"
    if "full" not in summaries: return result
    def rows(name, method):
        selected = [r for r in summaries[name]["episodes"] if r["method"] == method]
        indexed = {(r["case_id"], r["seed"]): r for r in selected}
        if not selected or len(indexed) != len(selected): raise ValueError("missing/duplicate mechanism case")
        return indexed
    base = rows("full", "sana")
    baseline_ttn = rows("full", "ttn")
    signatures = ("input_sha256", "initial_noise_sha256", "base_sha256")
    if plan_path.is_file():
        original = json.loads((Path(plan["source_evaluation"]) / "long/summary.json").read_text())
        for field in ("checkpoint_sha256", "fixed_cases_sha256", "stage", "step", "frames", "steps",
                      "cfg_scale", "cached_blocks", "flow_shift", "ttn_camera_attention"):
            if original["protocol"][field] != summaries["full"]["protocol"][field]:
                raise ValueError(f"Full C differs from the source milestone protocol: {field}")
        prior = {(r["case_id"], r["seed"]): r for r in original["episodes"] if r["method"] == "ttn"}
        if prior.keys() != baseline_ttn.keys(): raise ValueError("Full C differs from source milestone cases")
        reproduction = []
        for key, now in baseline_ttn.items():
            old = prior[key]
            if any(now[s] != old[s] for s in signatures): raise ValueError("Full C source milestone input/noise differs")
            a, b = old["metrics"]["per_frame_latent_mse"], now["metrics"]["per_frame_latent_mse"]
            if len(a) != len(b): raise ValueError("Full C source milestone metric horizons differ")
            reproduction.append({"case_id": key[0], "seed": key[1],
                "mean_mse_delta": now["metrics"]["mean_future_latent_mse"] - old["metrics"]["mean_future_latent_mse"],
                "max_abs_per_frame_mse_delta": max(abs(x-y) for x, y in zip(a[1:], b[1:])),
                "within_tolerance": all(abs(x-y) <= 1e-5 + 1e-4 * abs(x) for x, y in zip(a[1:], b[1:]))})
        result["full_c_reproduction"] = {"metric_tolerance": "abs 1e-5 + rel 1e-4; diagnostic, not bitwise latent equality",
                                        "cases": reproduction}
    for name in summaries:
        current = rows(name, "ttn")
        reference = rows(name, "sana") if name == "gt-history" else base
        if current.keys() != base.keys() or reference.keys() != base.keys() or baseline_ttn.keys() != base.keys():
            raise ValueError("mechanism case/seed sets differ")
        for key in base:
            if any(base[key][s] != r[s] for r in (current[key], reference[key], baseline_ttn[key]) for s in signatures):
                raise ValueError("mechanism paired inputs/noise/base identities differ")
        comparisons = {}
        for horizon, field in (("long", "metrics"), ("short", "prefix_13_metrics")):
            records = [dict(r, metrics=r[field]) for method in (reference, current) for r in method.values()]
            comparisons[horizon] = paired_summary(records)
        result["results"][name] = {"history_source": variants[name][1], **comparisons}
    def contrast(a, b, method="ttn"):
        return {h: {metric: result["results"][a][h]["metrics"][metric][method] - values[method]
                    if values[method] is not None and result["results"][a][h]["metrics"][metric][method] is not None else None
                    for metric, values in result["results"][b][h]["metrics"].items()}
                for h in ("short", "long")}
    if {"no-ttt", "identity"} <= summaries.keys():
        result["contrasts"]["controller_no_ttt_minus_identity"] = contrast("no-ttt", "identity")
    if "no-ttt" in summaries:
        result["contrasts"]["online_ttt_minus_no_ttt"] = contrast("full", "no-ttt")
    for name, key in (("no-local", "local_minus_no_local"), ("no-persistent", "persistent_minus_no_persistent")):
        if name in summaries: result["contrasts"][key] = contrast("full", name)
    if "gt-history" in summaries:
        result["contrasts"]["ttn_gt_history_minus_generated_history"] = contrast("gt-history", "full")
        result["contrasts"]["sana_gt_history_minus_generated_history"] = contrast("gt-history", "full", "sana")
    if len(summaries) == len(variants): result["status"] = "completed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--variant", choices=META_VARIANTS)
    parser.add_argument("--status", choices=("running", "completed", "failed"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if bool(args.variant) != bool(args.status): parser.error("--variant and --status must be specified together")
    output.mkdir(parents=True, exist_ok=True)
    if args.variant:
        # Do not pre-create a variant's directory: evaluate requires a new path.
        atomic_json(output / f"{args.variant}-status.json", {"status": args.status})
    try:
        result = collect_mechanisms(output)
    except Exception as error:
        atomic_json(output / "summary.json", {"status": "failed", "error": str(error)})
        raise
    atomic_json(output / "summary.json", result)
    print(f"[TTN mechanisms] status={result['status']} completed={len(result['results'])}/{len(result['variants'])} output={output}", flush=True)
    if result["status"] == "completed":
        if "full_c_reproduction" in result:
            print("[TTN Full C reproduction] " + json.dumps(result["full_c_reproduction"]), flush=True)
        for name, comparison in result["results"].items():
            short = comparison["short"]["metrics"]["mean_future_latent_mse"]["ttn"]
            long = comparison["long"]["metrics"]
            print(f"{name}: short={short:.6f} long={long['mean_future_latent_mse']['ttn']:.6f} "
                  f"tail={long['after_training_horizon_mean_latent_mse']['ttn']:.6f} "
                  f"final={long['final_chunk_latent_mse']['ttn']:.6f}", flush=True)


if __name__ == "__main__": main()
