"""Run the fixed step175 four-way probe from an isolated, minimal source overlay."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


GROUPS = ("baseline", "low-zero", "low-reference", "full-reference")


def bootstrap(project):
    overlay = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    import worldttn, tools, diffusion.scheduler
    for package, relative in ((worldttn, "worldttn"), (tools, "tools"), (diffusion.scheduler, "diffusion/scheduler")):
        package.__path__.insert(0, str(overlay / relative))
    return overlay


def sana_config(args):
    run = json.loads((args.output/"step175-snapshot/run_config.json").read_text())
    source = Path(run["arguments"].get("sana_config") or
                  json.loads((args.project/"configs/worldttn/frame_fullgrad.json").read_text())["sana_config"])
    return source if source.is_absolute() else args.project/source


def smoke(args):
    bootstrap(args.project)
    import torch
    from types import SimpleNamespace
    from worldttn.evaluation import load_evaluation_run, diagnostic_noise, tensor_sha256
    from worldttn.cli import rollout, seed_everything, to_device
    from worldttn.sana import build_sana, configure_cross_attention
    from worldttn.checkpoint import load_checkpoint
    from worldttn.performance import configure_execution, ExecutionOptions
    from worldttn.spectral_readout import SpectralReadout, install_spectral_readout

    evaluation = SimpleNamespace(training_run=args.output/"step175-snapshot", adapter=None, stage=None,
        config=args.project/"configs/worldttn/frame_fullgrad.json", sana_config=sana_config(args), dataset_root=None,
        data_dir=None, vae_cache_dir=None)
    run, config, ttn, adapter, digest, row = load_evaluation_run(evaluation)
    assert row["step"] == 175 and digest == json.loads((args.output/"experiment.json").read_text())["sha256"]
    del row; gc.collect()
    batch = torch.load(args.conditioning/"input-bundle.pt", map_location="cpu", weights_only=True)
    full_shape = (1, config.vae.vae_latent_dim, batch["camera_conditions"].shape[1], *batch["initial_latent"].shape[-2:])
    noise = diagnostic_noise(full_shape, "cuda", 3407)[:, :, :16].contiguous()
    batch["camera_conditions"] = batch["camera_conditions"][:, :16]
    batch["chunk_plucker"] = batch["chunk_plucker"][:, :, :16]
    batch = to_device(batch, "cuda")
    seed_everything(3407)
    model = build_sana(config, ttn, run["base"]["source"], "cuda", install_adapter=True, dtype=torch.float32)
    assert model.base_load_report["sha256"] == run["base"]["sha256"]
    configure_execution(model, ExecutionOptions())
    load_checkpoint(adapter, model)
    configure_cross_attention(model, "math")
    model.eval().requires_grad_(False)
    results, proof = {}, {}
    with torch.no_grad():
        for name in ("original", "baseline", "low-reference"):
            seed_everything(3407)
            if name != "original":
                controller = SpectralReadout(name, diagnostics=args.output/f"smoke-{name}.jsonl")
                install_spectral_readout(model, controller)
            generated, runtime, _ = rollout(model, config, batch, 4, 4.5, 2, initial_noise=noise,
                on_chunk=lambda r: print(f"[Spectral smoke] {name} chunk={r['chunk']}", flush=True))
            assert runtime.commit_count == 6 and torch.equal(generated[:, :, :1], batch["initial_latent"])
            results[name] = generated.cpu()
            if name != "original": proof[name] = controller.verify()
        torch.testing.assert_close(results["original"], results["baseline"], rtol=0, atol=0)
        assert proof["baseline"]["reference_sha256"] == proof["low-reference"]["reference_sha256"]
        first = [json.loads(line) for line in (args.output/"smoke-low-reference.jsonl").read_text().splitlines()]
        first = [row for row in first if row["call"] == 0 and row["frame_ids"] == [[1, 2, 3]]*2]
        assert {row["anchor"] for row in first if row["active"] and row["private_candidate"]
                and row["output_effect"]["delta_norm"] > 0} == {3, 7, 11, 15, 19}
    proof.update(status="passed", step=175, checkpoint_sha256=digest, commits=6,
                 baseline_exact=True, first_future_frames_verified=True,
                 original_latents_sha256=tensor_sha256(results["original"]))
    (args.output/"smoke.json").write_text(json.dumps(proof, indent=2)+"\n")
    print("[Spectral smoke] passed; original and diagnostic baseline exactly equal", flush=True)


def gpu_ready(gpu):
    output = subprocess.check_output(["nvidia-smi", f"--id={gpu}",
        "--query-gpu=uuid,memory.used,memory.free", "--format=csv,noheader,nounits"], text=True).strip()
    uuid, used, free = (value.strip() for value in output.split(","))
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader"], text=True)
    available_kib = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
    ready = int(used) < 1024 and int(free) >= 38000 and uuid not in processes and available_kib >= 80*1024**2
    return ready, {"gpu": gpu, "uuid": uuid, "memory_used_mib": int(used), "memory_free_mib": int(free),
                   "host_available_gib": round(available_kib/1024**2, 2)}


def run(args):
    state_path = args.output/"experiment.json"
    state = json.loads(state_path.read_text())
    case = json.loads(args.case.read_text())
    if (state["step"] != 175 or case["latent_frames"] != 121 or case["seed"] != 3407
            or case["camera"]["trajectory"] != "static"):
        raise ValueError("this probe requires the agreed step175 static 121-latent case/seed")
    previous_pid = state.get("pid")
    if state.get("status") in ("waiting_for_gpu", "running") and previous_pid:
        try: os.kill(previous_pid, 0)
        except ProcessLookupError: pass
        else: raise RuntimeError("this experiment already has a live launcher")
    overlay = Path(__file__).resolve().parents[1]
    manifest = json.loads((overlay/"files.json").read_text())
    def verify_sources():
        for directory, files in ((overlay, manifest["overlay"]), (args.project, manifest["base"])):
            for relative, digest in files.items():
                if hashlib.sha256((directory/relative).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"source file changed: {directory/relative}")
    verify_sources()
    state.update(mode="launch-only", pid=os.getpid(), status="waiting_for_gpu", exit_code=None,
        error=None, wrapper_exit_code=None, waiting_announced=False, phase=None, child_pid=None, started_utc=None,
        command=sys.argv, code_manifest=manifest, log=str(args.output/"launch.log"),
        tmux_session="wm", tmux_window=os.environ["TTN_TMUX_WINDOW"], tmux_pane=os.environ["TMUX_PANE"],
        groups=list(GROUPS), protocol={"gain": .25, "sigma_xy_nyquist": .125, "noise_min": .8,
        "cfg": 4.5, "solver_steps": 20, "latent_frames": 121, "seed": 3407})
    def publish(**changes):
        state.update(changes, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2)+"\n")
        os.replace(temporary, state_path)
    child = None
    try:
        while True:
            ready, resources = gpu_ready(args.gpu)
            publish(resources=resources)
            if ready: break
            if not state.get("waiting_announced"):
                print("[Spectral queue] waiting for an unoccupied GPU and 80 GiB available host RAM: "
                      +json.dumps(resources), flush=True)
                publish(waiting_announced=True)
            time.sleep(30)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", GDN_DISABLE_COMPILE="1",
                   PYTHONUNBUFFERED="1")
        common = [sys.executable, "-u", str(Path(__file__).resolve()), "--project", str(args.project),
                  "--output", str(args.output), "--conditioning", str(args.conditioning), "--case", str(args.case)]
        for phase in ("smoke", *GROUPS):
            verify_sources()
            # One isolated process per phase makes RAM peaks meaningful and releases all CUDA state.
            command = [*common, "--phase", phase]
            child = subprocess.Popen(command, cwd=args.project, env=env)
            publish(status="running", phase=phase, child_pid=child.pid, child_command=command,
                    started_utc=state.get("started_utc") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            code = child.wait()
            if code: raise RuntimeError(f"phase {phase} exited {code}; see preserved log")
        rows = []
        for group in GROUPS:
            summary = json.loads((args.output/group/"summary.json").read_text())
            episode = next(r for r in summary["episodes"] if r["method"] == "ttn")
            assert summary["protocol"]["step"] == 175 and summary["protocol"]["checkpoint_sha256"] == state["sha256"]
            rows.append({"group": group, **episode["latent_diagnostics"], "timing": episode["timing"],
                         "reference_sha256": episode["spectral_readout"]["reference_sha256"],
                         "video": str(args.output/group/"videos/comparison.mp4")})
        assert len({r["reference_sha256"] for r in rows}) == 1
        (args.output/"results.json").write_text(json.dumps({"status": "completed", "step": 175,
            "checkpoint_sha256": state["sha256"], "note": "One static case/seed; latent drift is not future-GT accuracy",
            "groups": rows}, indent=2)+"\n")
        publish(status="completed", exit_code=0, child_pid=None)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        publish(status="failed", exit_code=1, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditioning", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--phase", choices=("run", "smoke", *GROUPS), default="run")
    args = parser.parse_args()
    if args.phase == "run": return run(args)
    if args.phase == "smoke": return smoke(args)
    bootstrap(args.project)
    from tools.ttn_custom_inference import main as generate
    reuse = ["--reuse-conditioning", str(args.conditioning)] if args.phase == "baseline" else [
        "--reuse-baseline", str(args.output/"baseline")]
    generate(["--training-run", str(args.output/"step175-snapshot"), "--case", str(args.case),
        "--output", str(args.output/args.phase), "--config", str(args.project/"configs/worldttn/frame_fullgrad.json"),
        "--sana-config", str(sana_config(args)),
        "--steps", "20", "--cfg-scale", "4.5", "--cached-blocks", "2", "--cross-attn-backend", "math",
        "--spectral-readout", args.phase, *reuse])


if __name__ == "__main__":
    main()
