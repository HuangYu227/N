"""Submit four single-GPU interventions using a completed milestone's pinned snapshot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): result.update(block)
    return result.hexdigest()


def prepare(evaluation, output=None, fixed_cases=None):
    evaluation = Path(evaluation).resolve()
    completed = json.loads((evaluation / "summary.json").read_text())
    if completed["status"] != "completed": raise ValueError("source milestone evaluation must be completed")
    protocol = json.loads((evaluation / "long/summary.json").read_text())["protocol"]
    if protocol["stage"] != "C" or protocol["frames"] != 61 or protocol["ttn_camera_attention"] != "sana":
        raise ValueError("requires a C / original-SANA-camera / 61-frame milestone")
    if protocol.get("camera_ablation", False): raise ValueError("source evaluation must not override the trained camera")
    snapshot = Path(protocol["training_run"])
    metadata = json.loads((snapshot / "snapshot.json").read_text())
    if metadata["step"] != protocol["step"] or digest(snapshot / "last.pt") != protocol["checkpoint_sha256"]:
        raise ValueError("pinned checkpoint no longer matches the completed milestone")
    candidates = [Path(fixed_cases)] if fixed_cases else [Path(metadata["source"]) / "fixed-cases.pt",
        *[parent / "fixed-cases.pt" for parent in list(evaluation.parents)[:4]]]
    cases = next((p for p in candidates if p.is_file() and digest(p) == protocol["fixed_cases_sha256"]), None)
    if cases is None: raise ValueError("matching fixed cases not found; pass --fixed-cases explicitly")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(output).resolve() if output else evaluation.parent / f"mechanisms-step-{protocol['step']:06d}-{stamp}"
    if output.exists(): raise ValueError("use a new mechanism output directory")
    return {"output": str(output), "source_evaluation": str(evaluation), "snapshot": str(snapshot),
            "fixed_cases": str(cases.resolve()), "checkpoint_sha256": protocol["checkpoint_sha256"],
            "fixed_cases_sha256": protocol["fixed_cases_sha256"], "step": protocol["step"],
            "steps": protocol["steps"], "cfg_scale": protocol["cfg_scale"], "cached_blocks": protocol["cached_blocks"],
            "eval_cases": completed["results"]["long"]["metrics"]["mean_future_latent_mse"]["paired_count"],
            "seed": json.loads((evaluation / "long/manifest.json").read_text())["cases"][0]["seed"],
            "compile": protocol.get("compile", {}),
            "cross_attn_backend": protocol.get("cross_attn_backend", "math")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, help="completed joint/evaluations/step-000100 directory")
    parser.add_argument("--fixed-cases")
    parser.add_argument("--output")
    parser.add_argument("--partition", default="short")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = prepare(args.evaluation, args.output, args.fixed_cases)
    project = Path(__file__).resolve().parents[1]
    root = Path(os.environ.get("ROOT", "/data/group/zhaolab/home/z2zhang/huangyu"))
    python = root / "envs/worldttn/bin/python"
    if Path(sys.executable).resolve() != python.resolve():
        raise ValueError(f"run this submitter with the existing prefix interpreter: {python}")
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("SLURM_", "SBATCH_")) or key in (
            "RANK", "LOCAL_RANK", "WORLD_SIZE", "NODE_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
            "CUDA_VISIBLE_DEVICES", "COMMAND", "ADAPTER", "CAMERA_ATTENTION", "CAMERA_ABLATION", "NOISE_FRAMES",
            "TTN_ABLATION", "HISTORY_SOURCE", "STATE_DIAGNOSTICS", "EVAL_METHODS", "CONFIG", "SANA_CONFIG", "BASE_WEIGHTS"):
            env.pop(key, None)
    env.update(ROOT=str(root), PYTHON=str(python), PROJECT_ROOT=str(project), TRAINING_RUN=plan["snapshot"],
        MECHANISM_OUTPUT=plan["output"], FIXED_CASES=plan["fixed_cases"], STEPS=str(plan["steps"]),
        CFG_SCALE=str(plan["cfg_scale"]), CACHED_BLOCKS=str(plan["cached_blocks"]),
        EVAL_CASES=str(plan["eval_cases"]), SEED=str(plan["seed"]), CROSS_ATTN_BACKEND=plan["cross_attn_backend"])
    for name in ("GDN_DISABLE_COMPILE", "GDN_DISABLE_COMPLEX_COMPILE"):
        if plan["compile"].get(name) is not None: env[name] = plan["compile"][name]
    command = ["sbatch", "--parsable", "--export=ALL", f"--partition={args.partition}", f"--chdir={project}",
               f"--output={plan['output']}/slurm-%A_%a.out", str(project / "tools/ttn_slurm_mechanism.sbatch")]
    if args.dry_run:
        print(json.dumps({"plan": plan, "command": command}, indent=2))
        return
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    response = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    job = response.stdout.strip().split(";")[0]
    if not job.isdecimal(): raise RuntimeError(f"unexpected sbatch response: {response.stdout!r}")
    (output / "job.json").write_text(json.dumps({"job": job, "array": "0-3%1"}) + "\n")
    print(f"MECHANISM_JOB={job}\nMECHANISM_OUTPUT={output}", flush=True)


if __name__ == "__main__": main()
