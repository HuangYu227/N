"""Continue an existing full-C/DiT run in sequential, hour-limited Slurm jobs.

Uses the existing training launcher and exact-resume protocol unchanged. Only
one successor is submitted at a time, after a successful training segment.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


PROFILE = {
    "config": "CONFIG", "sana_config": "SANA_CONFIG", "base_weights": "BASE_WEIGHTS",
    "dataset_root": "DATASET_ROOT", "data_dir": "DATA_DIR", "vae_cache_dir": "VAE_CACHE_DIR",
    "batch_file": "BATCH_FILE", "seed": "SEED", "tbptt": "TBPTT",
    "backbone_lr": "BACKBONE_LR", "text_encoder_device": "TEXT_ENCODER_DEVICE",
    "activation_offload": "ACTIVATION_OFFLOAD", "cross_attn_backend": "CROSS_ATTN_BACKEND",
}


def last_step(run):
    """Ignore a partly written final JSON line while another job is finishing."""
    path = Path(run) / "train.jsonl"
    step = None
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = int(record["step"])
    return step


def require_checkpoint(run, expected):
    if last_step(run) != expected or not (Path(run) / "last.pt").is_file():
        raise ValueError(f"{run}: expected a completed step-{expected} run and last.pt")


def parent_dependency(job):
    """Completed jobs may have left slurmctld; use accounting before omitting dependency."""
    if not job.isdecimal():
        raise ValueError("--after-job must be a numeric Slurm job ID")
    result = subprocess.run(["squeue", "--noheader", "--jobs", job, "--format=%T"],
                            capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return job
    result = subprocess.run(["sacct", "--noheader", "--allocations", "--jobs", job,
                             "--format=State,ExitCode", "--parsable2"],
                            capture_output=True, text=True, check=True)
    rows = [line.split("|")[:2] for line in result.stdout.splitlines() if line.strip()]
    if rows != [["COMPLETED", "0:0"]]:
        raise ValueError(f"Job {job} is not known to have succeeded: {result.stdout.strip()}")
    return None


def environment(plan, start):
    env = os.environ.copy()
    # A new allocation chooses its own rendezvous; submitting from a Slurm
    # session must not carry that session's ranks, GPU mask or launch options.
    for name in list(env):
        if name.startswith(("SLURM_", "SBATCH_")) or name in (
            *PROFILE.values(), "OUTPUT", "ADAPTER", "RESUME", "MASTER_ADDR", "MASTER_PORT",
            "RANK", "LOCAL_RANK", "WORLD_SIZE", "NODE_RANK", "LOCAL_WORLD_SIZE", "CUDA_VISIBLE_DEVICES",
            "STAGES", "FRAMES", "STEPS", "LATENT_HEIGHT", "LATENT_WIDTH", "DIAGNOSTIC_UNMASK_ALL_VALID",
        ):
            env.pop(name, None)
    env.update(plan["environment"])
    env.update(OUTPUT=plan["output"], MAX_STEPS=str(min(start + plan["segment_steps"], plan["target_step"])),
               # The trainer always saves at MAX_STEPS. Save once per segment
               # instead of also saving at unrelated absolute-step multiples.
               SAVE_EVERY=str(plan["target_step"] + 1),
               ADAPTER=str(Path(plan["source"] if start == plan["initial_step"] else plan["output"]) / "last.pt"))
    return env


def submit(plan, manifest, start, dependency=None):
    stop = min(start + plan["segment_steps"], plan["target_step"])
    command = ["sbatch", "--parsable", "--export=ALL", "--kill-on-invalid-dep=yes",
               "--job-name=ttn-formal", f"--partition={plan['partition']}",
               "--nodes=4", "--ntasks=4", "--ntasks-per-node=1", "--gres=gpu:1",
               "--cpus-per-task=8", "--mem=256G", "--time=01:00:00",
               f"--chdir={plan['project']}", f"--output={plan['output']}/slurm-%j.out"]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command += [str(Path(plan["project"]) / "tools/ttn_slurm_chain.sbatch"),
                str(manifest), str(start)]
    result = subprocess.run(command, env=environment(plan, start), capture_output=True, text=True, check=True)
    job = result.stdout.strip().split(";")[0]
    if not job.isdecimal():
        raise RuntimeError(f"Unexpected sbatch response: {result.stdout!r}")
    record = {"job": job, "dependency": dependency, "start_step": start, "stop_step": stop}
    with (Path(plan["output"]) / "jobs.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    print(f"[TTN chain] job={job} steps={start + 1}..{stop} target={plan['target_step']}", flush=True)
    return job


def start_chain(args):
    project = Path(__file__).resolve().parents[1]
    source = Path(args.from_run).resolve()
    run = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    original = run["arguments"]
    if (run["parallel"], run["world_size"], original["stage"], original["train_scope"]) != ("fsdp2", 4, "C", "dit"):
        raise ValueError("Expected an existing 4-rank FSDP2 full Stage C/DiT training run")
    initial = int(original["max_steps"])
    if args.target_step <= initial or args.segment_steps < 1:
        raise ValueError("Target must exceed the source run's final target; segment steps must be positive")
    dependency = parent_dependency(args.after_job) if args.after_job else None
    if not dependency:
        require_checkpoint(source, initial)
    root = Path(os.environ.get("ROOT", project.parent)).resolve()
    python = root / "envs/worldttn/bin/python"
    if not python.is_file():
        raise ValueError(f"Existing prefix Python not found: {python}; no environment will be created")
    profile = {key: str(original[name]) for name, key in PROFILE.items() if original.get(name) is not None}
    profile.update(ROOT=str(root), PROJECT_ROOT=str(project), PYTHON=str(python), COMMAND="train",
                   PARALLEL="fsdp2", STAGE="C", TRAIN_SCOPE="dit", RESUME="1", MEMORY_TRACE="1",
                   CUDA_TRACE="0", CUDA_LAUNCH_BLOCKING="0", GDN_DISABLE_COMPILE="1",
                   GDN_DISABLE_COMPLEX_COMPILE="0", DISTRIBUTED_TIMEOUT="1800")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = project / "output/worldttn" / f"formal-C-{stamp}-{uuid.uuid4().hex[:6]}"
    output.mkdir(parents=True, exist_ok=False)
    plan = {"source": str(source), "project": str(project), "output": str(output), "environment": profile,
            "initial_step": initial, "target_step": args.target_step,
            "segment_steps": args.segment_steps, "partition": args.partition}
    manifest = output / "chain.json"
    manifest.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"[TTN chain] output={output}", flush=True)
    submit(plan, manifest, initial, dependency)


def worker(manifest, start):
    plan = json.loads(manifest.read_text(encoding="utf-8"))
    if start < plan["initial_step"] or start >= plan["target_step"]:
        raise ValueError("Invalid segment start")
    source = Path(plan["source"] if start == plan["initial_step"] else plan["output"])
    require_checkpoint(source, start)
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdecimal() or os.environ.get("SLURM_NTASKS") != "4":
        raise ValueError("Chain worker requires a four-task sbatch allocation")
    env = environment(plan, start)
    # For training inside THIS allocation retain Slurm's allocation metadata.
    env.update({key: value for key, value in os.environ.items() if key.startswith("SLURM_")})
    subprocess.run(["bash", str(Path(plan["project"]) / "tools/ttn_slurm_train.sbatch")],
                   env=env, cwd=plan["project"], check=True)
    stop = int(env["MAX_STEPS"])
    require_checkpoint(Path(plan["output"]), stop)
    if stop < plan["target_step"]:
        submit(plan, manifest, stop, dependency=job)
    else:
        print(f"[TTN chain] completed target step {stop}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    start = modes.add_parser("start", help="Submit the first segment; successors are submitted on success")
    start.add_argument("--from-run", required=True)
    start.add_argument("--after-job", help="Wait for this existing source job to finish successfully")
    start.add_argument("--target-step", type=int, default=500, help="Cumulative optimizer step, not per-job count")
    start.add_argument("--segment-steps", type=int, default=10)
    start.add_argument("--partition", default="short")
    work = modes.add_parser("worker", help=argparse.SUPPRESS)
    work.add_argument("manifest", type=Path)
    work.add_argument("start_step", type=int)
    args = parser.parse_args()
    try:
        start_chain(args) if args.mode == "start" else worker(args.manifest, args.start_step)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"[TTN chain] stopped: {error}", file=sys.stderr, flush=True)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
