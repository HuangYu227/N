"""Train joint DiT (optional TTN-new frozen control) in hour-limited Slurm jobs.

Uses the existing training launcher and exact-resume protocol unchanged. Only
one successor is submitted at a time, after a successful training segment.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid


PROFILE = {
    "config": "CONFIG", "sana_config": "SANA_CONFIG", "base_weights": "BASE_WEIGHTS",
    "dataset_root": "DATASET_ROOT", "data_dir": "DATA_DIR", "vae_cache_dir": "VAE_CACHE_DIR",
    "batch_file": "BATCH_FILE", "seed": "SEED", "tbptt": "TBPTT",
    "backbone_lr": "BACKBONE_LR", "text_encoder_device": "TEXT_ENCODER_DEVICE",
    "activation_offload": "ACTIVATION_OFFLOAD", "cross_attn_backend": "CROSS_ATTN_BACKEND",
    "ttn_core_backend": "TTN_CORE_BACKEND", "ttn_psi_backend": "TTN_PSI_BACKEND",
    "optimizer_policy": "OPTIMIZER_POLICY",
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


def training_world_size(value):
    if type(value) is not int or value < 2:
        raise ValueError("FSDP2 training requires an integer world size >= 2")
    return value


def verify_checkpoint_world(report, world):
    if report.get("mode") != "fsdp2" or report.get("world_size") != world:
        raise ValueError(f"checkpoint backend/world differs from planned fsdp2/{world}")


def require_checkpoint(run, expected, world=None):
    from worldttn.checkpoint_integrity import audit_checkpoint
    if last_step(run) != expected or not (Path(run) / "last.pt").is_file():
        raise ValueError(f"{run}: expected a completed step-{expected} run and last.pt")
    report = audit_checkpoint(Path(run) / "last.pt", expected)
    verify_training_record(run, report)
    if world is not None: verify_checkpoint_world(report, world)
    return report


def verify_training_record(run, checkpoint):
    record = None
    with (Path(run) / "train.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            try: row = json.loads(line)
            except json.JSONDecodeError: continue
            if row.get("step") == checkpoint["step"]: record = row
    if record is None:
        raise ValueError(f"{run}: no completed training record for checkpoint step {checkpoint['step']}")
    for key in ("stage", "train_scope"):
        if checkpoint.get(key) is not None and record.get(key) != checkpoint[key]:
            raise ValueError(f"{run}: training record {key} differs from checkpoint step {checkpoint['step']}")


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
            *PROFILE.values(), "OUTPUT", "ADAPTER", "RESUME", "UNFREEZE", "TRAIN_SCOPE", "MASTER_ADDR", "MASTER_PORT",
            "RANK", "LOCAL_RANK", "WORLD_SIZE", "NODE_RANK", "LOCAL_WORLD_SIZE", "CUDA_VISIBLE_DEVICES", "TTN_RENDEZVOUS_FILE",
            "STAGES", "FRAMES", "STEPS", "LATENT_HEIGHT", "LATENT_WIDTH", "DIAGNOSTIC_UNMASK_ALL_VALID",
            "CAMERA_ATTENTION", "CAMERA_ABLATION", "TRAINING_RUN", "FIXED_CASES", "EVAL_CASES",
            "TTN_PROFILE", "TTN_LAYOUT_AUDIT", "TTN_COMPARE_REFERENCE", "BENCHMARK_SAVE_CHECKPOINT",
            "TTN_ENTRY_MODULE", "META_TEST_OUTPUT",
        ):
            env.pop(name, None)
    env.update(plan["environment"])
    env.update(OUTPUT=plan["output"], MAX_STEPS=str(segment_stop(plan, start)),
               # The trainer always saves at MAX_STEPS. Save once per segment
               # instead of also saving at unrelated absolute-step multiples.
               SAVE_EVERY=str(plan["target_step"] + 1),
               ADAPTER=str(Path(plan["source"] if start == plan["initial_step"] else plan["output"]) / "last.pt"))
    if plan.get("fresh"):
        env["RESUME"] = "0" if start == 0 else "1"
        if start == 0: env.pop("ADAPTER", None)
    if plan.get("unfreeze") and start == plan["initial_step"]:
        env.update(UNFREEZE="1", RESUME="0")
    return env


def segment_stop(plan, start):
    stop = min(start + plan["segment_steps"], plan["target_step"])
    evaluation = plan.get("evaluation")
    if evaluation:
        every = evaluation["every"]
        stop = min(stop, (start // every + 1) * every)
        if plan.get("unfreeze") and start == plan["initial_step"]: stop = min(stop, start + 1)
    return stop


def evaluation_settings(args, output):
    if getattr(args, "evaluation", None): return args.evaluation
    every = getattr(args, "eval_every", 0)
    if every < 0: raise ValueError("eval-every must be nonnegative; zero explicitly disables periodic evaluation")
    if not every: return None
    if getattr(args, "eval_steps", 20) < 1 or getattr(args, "eval_cases", 1) < 1:
        raise ValueError("positive eval-steps and eval-cases are required")
    key_steps = sorted(set(getattr(args, "keep_model_steps", (25, 50, 100, 250))) | {args.target_step})
    if any(step < 1 for step in key_steps): raise ValueError("key model steps must be positive")
    frames = getattr(args, "eval_frames", 61)
    if frames < 61 or frames % 3 != 1: raise ValueError("eval-frames must be 1+3n and >=61")
    return {"every": every, "seed": getattr(args, "eval_seed", 3407), "steps": getattr(args, "eval_steps", 20),
            "frames": frames, "cfg_scale": getattr(args, "eval_cfg_scale", 4.5),
            "keep_model_steps": key_steps,
            "cases": getattr(args, "eval_cases", 1), "fixed_cases": str(Path(output) / "fixed-cases.pt"),
            "output": str(Path(output) / "evaluations")}


def submit_evaluation(plan, step, parent_job):
    from tools.ttn_eval_snapshot import snapshot_training_run
    evaluation = plan["evaluation"]
    if not parent_job.isdecimal(): raise ValueError("evaluation requires a numeric parent Slurm job")
    # Old chains without a retention policy preserve their previous behavior.
    retain = "keep_model_steps" not in evaluation or step in evaluation["keep_model_steps"]
    snapshot = snapshot_training_run(plan["output"], retain_model=retain)
    output = Path(evaluation["output"]) / f"step-{step:06d}"
    output.parent.mkdir(parents=True, exist_ok=True)
    env = environment(plan, step)
    for key in ("ADAPTER", "RESUME", "UNFREEZE", "MAX_STEPS", "SAVE_EVERY", "TRAIN_SCOPE", "STAGE"):
        env.pop(key, None)
    env.update(COMMAND="stage-evaluate", TRAINING_RUN=str(snapshot), OUTPUT=str(output),
               FIXED_CASES=evaluation["fixed_cases"], EVAL_CASES=str(evaluation["cases"]),
               SEED=str(evaluation["seed"]), STEPS=str(evaluation["steps"]), CACHED_BLOCKS="2",
               CFG_SCALE=str(evaluation.get("cfg_scale", 4.5)), FRAMES=str(evaluation.get("frames", 61)))
    env.update(TTN_CORE_BACKEND="reference", TTN_PSI_BACKEND="reference")
    command = ["sbatch", "--parsable", "--export=ALL", "--kill-on-invalid-dep=yes", "--job-name=ttn-stage-eval",
               f"--partition={plan['partition']}", "--nodes=1", "--ntasks=1", "--ntasks-per-node=1",
               "--gres=gpu:1", "--cpus-per-task=8", "--mem=128G", f"--time={plan.get('time_limit', '01:00:00')}",
               f"--chdir={plan['project']}", f"--output={output.parent}/step-{step:06d}-%j.out",
               f"--dependency=afterok:{parent_job}", str(Path(plan["project"]) / "tools/ttn_slurm_eval.sbatch")]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    job = result.stdout.strip().split(";")[0]
    if not job.isdecimal(): raise RuntimeError(f"Unexpected evaluation sbatch response: {result.stdout!r}")
    with (Path(plan["output"]) / "eval_jobs.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"step": step, "job": job, "snapshot": str(snapshot), "output": str(output),
                                 "stdout": str(output.parent / f"step-{step:06d}-{job}.out"),
                                 "stderr": str(output.parent / f"step-{step:06d}-{job}.out"), "workdir": plan["project"],
                                 "train_scope": plan["environment"]["TRAIN_SCOPE"]}) + "\n")
    print(f"[TTN evaluation queued] step={step} job={job} snapshot={snapshot} output={output}", flush=True)


def submit(plan, manifest, start, dependency=None):
    world = training_world_size(plan.get("world_size", 4))
    stop = segment_stop(plan, start)
    command = ["sbatch", "--parsable", "--export=ALL", "--kill-on-invalid-dep=yes",
               "--job-name=ttn-formal", f"--partition={plan['partition']}",
               f"--nodes={world}", f"--ntasks={world}", "--ntasks-per-node=1", "--gres=gpu:1",
               "--cpus-per-task=8", f"--mem={plan.get('memory', '256G')}", f"--time={plan.get('time_limit', '01:00:00')}",
               f"--chdir={plan['project']}", f"--output={plan['output']}/slurm-%j.out"]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command += [str(Path(plan["project"]) / "tools/ttn_slurm_chain.sbatch"),
                str(manifest), str(start)]
    result = subprocess.run(command, env=environment(plan, start), capture_output=True, text=True, check=True)
    job = result.stdout.strip().split(";")[0]
    if not job.isdecimal():
        raise RuntimeError(f"Unexpected sbatch response: {result.stdout!r}")
    record = {"job": job, "dependency": dependency, "start_step": start, "stop_step": stop, "world_size": world,
              "stdout": str(Path(plan["output"]) / f"slurm-{job}.out"),
              "stderr": str(Path(plan["output"]) / f"slurm-{job}.out"), "workdir": plan["project"]}
    with (Path(plan["output"]) / "jobs.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    print(f"[TTN chain] job={job} steps={start + 1}..{stop} target={plan['target_step']} world_size={world}", flush=True)
    return job


def start_chain(args):
    project = Path(__file__).resolve().parents[1]
    source = Path(args.from_run).resolve()
    run = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    original = run["arguments"]
    world = training_world_size(run.get("world_size"))
    requested_world = getattr(args, "world_size", None)
    if requested_world is not None and training_world_size(requested_world) != world:
        raise ValueError("exact resume requires the source world size; start a fresh run to change it")
    unfreeze = getattr(args, "unfreeze", False)
    expected_scopes = ("ttn-visual", "ttn-new") if unfreeze else ("dit",)
    if ((run["parallel"], original["stage"]) != ("fsdp2", "C")
            or original["train_scope"] not in expected_scopes):
        raise ValueError("Expected an existing FSDP2 full Stage C/DiT training run")
    if unfreeze and run["training"].get("train_scope") not in ("ttn-visual", "ttn-new"):
        raise ValueError("unfreeze source must be a visual warmup run")
    dependency = parent_dependency(args.after_job) if args.after_job else None
    if dependency:
        # This explicit dependency waits for the source's declared endpoint;
        # worker must still validate the actually published checkpoint.
        initial = int(original["max_steps"])
        checkpoint_report = {"status": "awaiting_parent", "step": initial}
    else:
        from worldttn.checkpoint_integrity import audit_checkpoint
        checkpoint_report = audit_checkpoint(source / "last.pt")
        verify_training_record(source, checkpoint_report)
        verify_checkpoint_world(checkpoint_report, world)
        initial = int(checkpoint_report["step"])
        observed = last_step(source)
        if observed is None or observed < initial:
            raise ValueError("published checkpoint has no matching completed training history")
        checkpoint_report["last_logged_step"] = observed
        checkpoint_report["unsaved_steps"] = [initial + 1, observed] if observed > initial else None
        if observed > initial:
            print(f"[TTN recovery] saved={initial} logged={observed}; unsaved steps {initial + 1}..{observed}; "
                  "resume in a new output, original logs preserved", flush=True)
    if args.target_step <= initial or args.segment_steps < 1:
        raise ValueError("Target must exceed the verified checkpoint step; segment steps must be positive")
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
    output = (Path(args.output).resolve() if getattr(args, "output", None) else
              project / "output/worldttn" / f"formal-C-{stamp}-{uuid.uuid4().hex[:6]}")
    output.mkdir(parents=True, exist_ok=False)
    plan = {"source": str(source), "project": str(project), "output": str(output), "environment": profile,
            "initial_step": initial, "target_step": args.target_step, "source_checkpoint": checkpoint_report,
            "segment_steps": args.segment_steps, "partition": args.partition, "world_size": world,
            "time_limit": getattr(args, "time_limit", "01:00:00"), "memory": getattr(args, "memory", "256G")}
    if unfreeze: plan["unfreeze"] = True
    evaluation = evaluation_settings(args, output)
    if evaluation: plan["evaluation"] = evaluation
    manifest = output / "chain.json"
    manifest.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"[TTN chain] output={output}", flush=True)
    submit(plan, manifest, initial, dependency)


def fresh_chain(args):
    """Start from SANA, warm up visual TTN, then deliberately rebuild joint training."""
    world = training_world_size(getattr(args, "world_size", 4))
    project = Path(__file__).resolve().parents[1]
    root = Path(os.environ.get("ROOT", project.parent)).resolve()
    python = root / "envs/worldttn/bin/python"
    dataset = Path(args.dataset_root or root / "datasets/sana-wm-example").resolve()
    config = Path(args.config or project / "configs/worldttn/reference_sana_camera.json").resolve()
    reference = json.loads(config.read_text(encoding="utf-8"))
    if reference["ttn"].get("stage") != "C" or reference["ttn"].get("camera_attention") != "sana":
        raise ValueError("fresh adaptation requires the Stage C/original SANA camera config")
    if not python.is_file() or not dataset.is_dir():
        raise ValueError("Existing prefix Python and dataset are required; no environment/data will be created")
    if args.warmup_steps < 0 or args.segment_steps < 1 or args.target_step <= args.warmup_steps:
        raise ValueError("nonnegative warmup, positive segment and target-step > warmup-steps are required")
    if args.warmup_only and args.warmup_steps == 0:
        raise ValueError("warmup-only requires positive warmup-steps")
    if not math.isfinite(args.backbone_lr) or args.backbone_lr <= 0:
        raise ValueError("backbone learning rate must be finite and positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (Path(args.output).resolve() if args.output else
              project / "output/worldttn" / f"ucpe-C-{stamp}-{uuid.uuid4().hex[:6]}")
    output.mkdir(parents=True, exist_ok=False)
    warmup = output / ("adapt" if args.warmup_steps else "joint")
    warmup.mkdir()
    profile = {"ROOT": str(root), "PROJECT_ROOT": str(project), "PYTHON": str(python),
               "CONFIG": str(config), "DATASET_ROOT": str(dataset), "COMMAND": "train",
               "PARALLEL": "fsdp2", "STAGE": "C", "TRAIN_SCOPE": "ttn-new" if args.warmup_steps else "dit", "RESUME": "0",
               "OPTIMIZER_POLICY": "origin",
               "SEED": str(args.seed), "TBPTT": str(args.tbptt), "BACKBONE_LR": str(args.backbone_lr),
               "ACTIVATION_OFFLOAD": "cpu", "TEXT_ENCODER_DEVICE": "cpu", "CROSS_ATTN_BACKEND": "math",
               "MEMORY_TRACE": "1", "CUDA_TRACE": "0", "CUDA_LAUNCH_BLOCKING": "0",
               "GDN_DISABLE_COMPILE": "1", "GDN_DISABLE_COMPLEX_COMPILE": "0", "DISTRIBUTED_TIMEOUT": "1800"}
    if args.base_weights: profile["BASE_WEIGHTS"] = args.base_weights
    plan = {"source": "", "fresh": True, "project": str(project), "output": str(warmup),
            "environment": profile, "initial_step": 0, "target_step": args.warmup_steps or args.target_step,
            "segment_steps": args.segment_steps, "partition": args.partition, "world_size": world,
            "time_limit": getattr(args, "time_limit", "01:00:00"), "memory": getattr(args, "memory", "256G")}
    if args.warmup_steps and not args.warmup_only:
        plan.update(joint_output=str(output / "joint"), joint_target_step=args.target_step)
    evaluation = evaluation_settings(args, output)
    if evaluation: plan["evaluation"] = evaluation
    manifest = warmup / "chain.json"
    manifest.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"[TTN chain] output={output} frozen_steps={args.warmup_steps} "
          f"joint={'held' if args.warmup_only else str(args.warmup_steps + 1) + '..' + str(args.target_step)}", flush=True)
    submit(plan, manifest, 0)


def _worker(plan, manifest, start, status):
    if start < plan["initial_step"] or start >= plan["target_step"]:
        raise ValueError("Invalid segment start")
    world = training_world_size(plan.get("world_size", 4))
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdecimal() or os.environ.get("SLURM_NTASKS") != str(world):
        raise ValueError(f"Chain worker requires a {world}-task sbatch allocation")
    source = Path(plan["source"] if start == plan["initial_step"] else plan["output"])
    if not (plan.get("fresh") and start == 0):
        if start == plan["initial_step"] and plan.get("source_checkpoint", {}).get("unsaved_steps"):
            from worldttn.checkpoint_integrity import audit_checkpoint
            report = audit_checkpoint(source / "last.pt", start)
            verify_training_record(source, report)
            verify_checkpoint_world(report, world)
        else:
            require_checkpoint(source, start, world)
    env = environment(plan, start)
    # For training inside THIS allocation retain Slurm's allocation metadata.
    env.update({key: value for key, value in os.environ.items() if key.startswith("SLURM_")})
    status["phase"] = "training"
    subprocess.run(["bash", str(Path(plan["project"]) / "tools/ttn_slurm_train.sbatch")],
                   env=env, cwd=plan["project"], check=True)
    stop = int(env["MAX_STEPS"])
    status["phase"] = "checkpoint-validation"
    require_checkpoint(Path(plan["output"]), stop, world)
    evaluation = plan.get("evaluation")
    status["phase"] = "successor-submission"
    if evaluation and (stop % evaluation["every"] == 0 or stop == plan["target_step"] or
                       (plan.get("unfreeze") and start == plan["initial_step"])):
        submit_evaluation(plan, stop, job)
    if stop < plan["target_step"]:
        submit(plan, manifest, stop, dependency=job)
    elif plan.get("joint_target_step"):
        start_chain(SimpleNamespace(from_run=plan["output"], after_job=job, unfreeze=True,
                                   output=plan["joint_output"], target_step=plan["joint_target_step"],
                                   segment_steps=plan["segment_steps"], partition=plan["partition"], evaluation=evaluation,
                                   world_size=world,
                                   time_limit=plan.get("time_limit", "01:00:00"), memory=plan.get("memory", "256G")))
    else:
        print(f"[TTN chain] completed target step {stop}", flush=True)


def worker(manifest, start):
    from worldttn.checkpoint_integrity import atomic_json
    from worldttn.failure import record_failure
    plan = json.loads(manifest.read_text(encoding="utf-8"))
    job = os.environ.get("SLURM_JOB_ID", "local")
    status = {"job": job, "start_step": start, "status": "running", "phase": "source-validation",
              "stdout": str(Path(plan["output"]) / f"slurm-{job}.out")}
    target = Path(plan["output"]) / f"chain-status-{job}.json"
    atomic_json(status, target)
    try:
        _worker(plan, manifest, start, status)
        status.update(status="completed", phase="completed")
    except Exception as error:
        record_failure(plan["output"], status["phase"], error, rank="chain", step=start)
        status.update(status="failed", error=str(error))
        raise
    finally:
        try: atomic_json(status, target)
        except OSError as error: print(f"[TTN chain] could not write status: {error}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    start = modes.add_parser("start", help="Submit the first segment; successors are submitted on success")
    start.add_argument("--from-run", required=True)
    start.add_argument("--world-size", type=int, help="Inherit source world size; an override must match exact resume")
    start.add_argument("--after-job", help="Wait for this existing source job to finish successfully")
    start.add_argument("--target-step", type=int, default=500, help="Cumulative optimizer step, not per-job count")
    start.add_argument("--segment-steps", type=int, default=10)
    start.add_argument("--partition", default="short")
    start.add_argument("--unfreeze", action="store_true", help="Source is C/sana-camera visual warmup; reset optimizer once")
    start.add_argument("--output", help="New joint output directory")
    fresh = modes.add_parser("fresh", help="Fresh C/SANA-camera joint DiT; optional corrected TTN-new frozen control")
    fresh.add_argument("--world-size", type=int, default=4, help="One GPU per node/task; FSDP2 requires >=2 (default: 4)")
    fresh.add_argument("--warmup-steps", type=int, default=0, help="frozen TTN-new control; zero starts joint DiT immediately")
    fresh.add_argument("--target-step", type=int, default=500, help="Total adaptation + joint steps")
    fresh.add_argument("--segment-steps", type=int, default=10)
    fresh.add_argument("--partition", default="short")
    fresh.add_argument("--warmup-only", action="store_true", help="Stop at warmup for quality review before unfreezing")
    fresh.add_argument("--dataset-root")
    fresh.add_argument("--base-weights")
    fresh.add_argument("--config")
    fresh.add_argument("--output")
    fresh.add_argument("--seed", type=int, default=3407)
    fresh.add_argument("--tbptt", type=int, choices=(1, 2, 4), default=2)
    fresh.add_argument("--backbone-lr", type=float, default=1e-6)
    for command in (start, fresh):
        command.add_argument("--time-limit", default="01:00:00", help="Slurm wall time per training/evaluation allocation")
        command.add_argument("--memory", default="256G", help="Slurm host memory per training node")
        command.add_argument("--eval-every", type=int, default=25, help="immutable fixed-case evaluation cadence; zero disables")
        command.add_argument("--eval-seed", type=int, default=3407)
        command.add_argument("--eval-steps", type=int, default=20)
        command.add_argument("--eval-cases", type=int, default=1)
        command.add_argument("--eval-frames", type=int, default=61)
        command.add_argument("--eval-cfg-scale", type=float, default=4.5)
        command.add_argument("--keep-model-steps", type=int, nargs="*", default=[25, 50, 100, 250],
                             help="pin evaluation models only at these steps and target; all metric logs remain")
    work = modes.add_parser("worker", help=argparse.SUPPRESS)
    work.add_argument("manifest", type=Path)
    work.add_argument("start_step", type=int)
    args = parser.parse_args()
    try:
        if args.mode == "start": start_chain(args)
        elif args.mode == "fresh": fresh_chain(args)
        else: worker(args.manifest, args.start_step)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"[TTN chain] stopped: {error}", file=sys.stderr, flush=True)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
