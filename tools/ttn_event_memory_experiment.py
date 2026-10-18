"""Pinned forest-road inference ablations; never trains or reads future GT into history.

Prepare once on CPU, then run one process per allowed GPU. The checkpoint,
conditioning, original CPU FP32 noise, and source receipt are shared by all groups.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GROUPS = ("baseline", "initial", "recent", "content", "shuffled")
FOREST_CASE = "sekai_game/00100100001_0004650_0006450_s000000"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_receipt():
    """Hash executable/config source, including new files not yet staged in Git."""
    files = sorted(path for folder in ("worldttn", "diffusion", "sana", "tools", "configs")
                   for path in (ROOT / folder).rglob("*")
                   if path.is_file() and path.suffix in (".py", ".json", ".yaml", ".yml", ".toml"))
    hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    result = {"tree_sha256": digest, "files": hashes}
    try:
        result["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        result["git_status"] = subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        result["git_head"] = None
    return result


def configuration(run):
    from worldttn.evaluation import evaluation_config
    source = Path(run["arguments"].get("sana_config") or "configs/worldttn/sana_teacher_single_gpu.yaml")
    return evaluation_config(run, source if source.is_absolute() else ROOT / source)


def options_for(group, manifest):
    from worldttn.event_memory import EventMemoryOptions
    options = dict(manifest["event_memory"])
    options["anchors"] = tuple(options["anchors"])
    options.update(mode="off" if group == "baseline" else "state-replay",
                   selector="content" if group == "baseline" else group)
    return EventMemoryOptions(**options)


def prepare(args):
    """CPU-only: pin an existing complete protocol without regenerating inputs."""
    import torch
    from worldttn.core import BASE_REVISION
    from worldttn.evaluation import tensor_sha256
    original = args.reference_root.resolve()
    previous = json.loads((original / "experiment.json").read_text())
    cases = previous["cases"]
    if len(cases) != 1 or cases[0]["case_id"] != FOREST_CASE:
        raise ValueError("requires the previously verified single forest-road 60s protocol")
    if (previous["step"], previous["frames"], previous["raw_frames"], previous["fps"],
            previous["solver_steps"], previous["cfg_scale"], previous["cached_blocks"]) != (750, 121, 961, 16, 20, 4.5, 2):
        raise ValueError("reference is not the pinned step750 / 60s / 20-step / CFG4.5 / cache2 protocol")
    if previous.get("status") != "completed" or previous.get("ttn_ablation") != "full":
        raise ValueError("reference must be a completed full-StateTTT run")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    try:
        row = cases[0]
        input_path = original / row["name"] / "input.pt"
        checkpoint = Path(previous["checkpoint"]).resolve()
        checkpoint_sha = sha256(checkpoint)
        if checkpoint_sha != previous["checkpoint_sha256"]:
            raise ValueError("checkpoint bytes differ from original forest protocol")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        if (payload.get("step") != 750 or payload.get("base_revision") != BASE_REVISION
                or payload.get("weight_scope") != "dit"
                or payload.get("distributed", {}).get("checkpoint_id") != previous["checkpoint_id"]):
            raise ValueError("checkpoint metadata differs from original forest protocol")
        checkpoint_identity = {key: payload.get(key) for key in
                               ("step", "base_revision", "base_sha256", "weight_scope", "stage")}
        checkpoint_identity["checkpoint_id"] = previous["checkpoint_id"]
        del payload
        packet = torch.load(input_path, map_location="cpu", weights_only=False)
        if packet["reference"].shape != (1, 128, 121, 22, 40):
            raise ValueError("unexpected GT latent shape")
        if packet["noise"].dtype != torch.float32 or packet["noise"].device.type != "cpu":
            raise ValueError("must reuse original CPU FP32 noise")
        hashes = {key: tensor_sha256(packet["batch"][key]) for key in row["input_sha256"]}
        if hashes != row["input_sha256"] or tensor_sha256(packet["noise"]) != row["initial_noise_sha256"]:
            raise ValueError("saved conditioning/noise differs from original protocol")
        if not torch.equal(packet["batch"]["initial_latent"], packet["reference"][:, :, :1]):
            raise ValueError("observed latent differs from GT frame0")
        if "clean_latents" in packet["batch"]:
            raise ValueError("inference batch may contain only initial_latent, not future GT")
        run_config = original / "run_config.json"
        run = json.loads(run_config.read_text())
        if run["base"]["sha256"] != checkpoint_identity["base_sha256"]:
            raise ValueError("checkpoint and stored SANA base disagree")
        manifest = {
            "status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(),
            "reference_root": str(original), "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha, "checkpoint_identity": checkpoint_identity,
            "input_path": str(input_path), "input_file_sha256": sha256(input_path),
            "run_config": str(run_config), "run_config_sha256": sha256(run_config),
            "case": row, "frames": 121, "raw_frames": 961, "fps": 16,
            "solver_steps": 20, "cfg_scale": 4.5, "cached_blocks": 2,
            "precision": "FP32 weights; BF16 autocast; reference core and StateTTT; math text attention",
            "source": source_receipt(), "groups": list(GROUPS),
            "event_memory": {
                "anchors": args.anchors, "capacity": args.capacity, "topk": args.topk,
                "temperature": args.temperature, "gain": args.gain,
                "max_relative_delta": args.max_relative_delta, "retention": args.retention,
                "start_chunk": args.start_chunk, "max_read_sigma": args.max_read_sigma,
            },
            "scope": "one fixed training-seen forest case/seed; diagnostic, not independent statistical evidence",
            "history": "frame0 observed, future clean history self-generated; future GT stays on CPU for post-rollout metrics",
            "memory_representation": "complete post-RoPE per-event G/B/D statistics; no resampling or RoPE rewrites",
            "controls": "identical retention and archive coverage; selectors differ only in read selection; shuffled preserves K/V correspondence",
            "metrics_scope": "latent GT MSE; decoded GT PSNR/SSIM optional, reconstruction metrics are not perceptual quality or proof of memory",
        }
        for group in GROUPS:
            options_for(group, manifest)  # fail before launching any expensive worker
        write_json(root / "experiment.json", manifest)
        print(f"[Event memory prepared] {root}", flush=True)
    except BaseException as error:
        write_json(root / "prepare-error.json", {"error": repr(error)})
        raise


def check_sources(manifest):
    if source_receipt()["tree_sha256"] != manifest["source"]["tree_sha256"]:
        raise ValueError("source/config changed after protocol preparation; prepare a new experiment")


def worker(args):
    import torch
    from worldttn.anchor import offline_state_dict
    from worldttn.checkpoint import apply_checkpoint_weights
    from worldttn.cli import rollout, seed_everything, timed_cuda, to_device
    from worldttn.core import TTNConfig
    from worldttn.evaluation import latent_metrics, tensor_sha256
    from worldttn.performance import configure_execution, ExecutionOptions
    from worldttn.sana import build_sana, configure_cross_attention
    root = args.output.resolve()
    manifest = json.loads((root / "experiment.json").read_text())
    check_sources(manifest)
    folder = root / args.group
    folder.mkdir(exist_ok=False)
    write_json(folder / "status.json", {"status": "starting", "pid": os.getpid(),
               "group": args.group, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
    try:
        if torch.cuda.device_count() != 1:
            raise ValueError("worker must see exactly one GPU through CUDA_VISIBLE_DEVICES")
        torch.cuda.set_device(0)
        properties = torch.cuda.get_device_properties(0)
        cap = args.allocator_gib * 2**30
        if cap >= properties.total_memory:
            raise ValueError("allocator cap must be below device capacity")
        torch.cuda.set_per_process_memory_fraction(cap / properties.total_memory)
        if torch.cuda.mem_get_info()[0] < args.minimum_free_gib * 2**30:
            raise RuntimeError("selected GPU no longer has enough free memory")
        if sha256(manifest["input_path"]) != manifest["input_file_sha256"]:
            raise ValueError("pinned input file changed")
        if sha256(manifest["run_config"]) != manifest["run_config_sha256"]:
            raise ValueError("pinned run_config changed")
        if sha256(manifest["checkpoint"]) != manifest["checkpoint_sha256"]:
            raise ValueError("pinned checkpoint changed")
        run = json.loads(Path(manifest["run_config"]).read_text())
        config = configuration(run)
        payload = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
        identity = manifest["checkpoint_identity"]
        if any(payload.get(key) != identity[key] for key in
               ("step", "base_revision", "base_sha256", "weight_scope", "stage")):
            raise ValueError("checkpoint metadata changed")
        if payload.get("distributed", {}).get("checkpoint_id") != identity["checkpoint_id"]:
            raise ValueError("checkpoint ID changed")
        ttn = TTNConfig(**payload["config"])
        if ttn.state_ttt_rank != 64 or ttn.stage != "C" or not ttn.local_update or not ttn.persistent_meta:
            raise ValueError("requires the enhanced StateTTT stageC checkpoint")
        seed_everything(3407)  # same construction RNG as the original forest runner
        model = build_sana(config, ttn, run["base"]["source"], "cpu", dtype=torch.float32)
        if model.base_load_report["sha256"] != identity["base_sha256"]:
            raise ValueError("loaded a different SANA base")
        expected = offline_state_dict(model, "dit")
        if expected.keys() != payload["adapter"].keys() or any(
                expected[key].shape != payload["adapter"][key].shape for key in expected):
            raise ValueError("trained model keys/shapes differ")
        apply_checkpoint_weights(model, payload)
        del expected, payload
        gc.collect()
        configure_execution(model, ExecutionOptions())
        configure_cross_attention(model, "math")
        model.eval().requires_grad_(False).to("cuda")
        packet = torch.load(manifest["input_path"], map_location="cpu", weights_only=False)
        if "clean_latents" in packet["batch"]:
            raise ValueError("future GT must never be passed to rollout")
        reference = packet["reference"]  # explicitly kept on CPU
        row = manifest["case"]
        seed_everything(row["seed"])
        batch = to_device(packet["batch"], "cuda")
        noise = packet["noise"].to("cuda")
        hashes = {key: tensor_sha256(batch[key]) for key in row["input_sha256"]}
        if hashes != row["input_sha256"] or tensor_sha256(noise) != row["initial_noise_sha256"]:
            raise ValueError("conditioning or noise changed during device transfer")
        options = options_for(args.group, manifest)
        write_json(folder / "options.json", asdict(options))

        def progress(record):
            with (folder / "chunks.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            print("[Event memory chunk] " + json.dumps({key: record[key] for key in
                  ("chunk", "start", "end", "seconds")}), flush=True)

        def state_record(chunk, runtime):
            record = {"chunk": chunk, "commit_count": runtime.commit_count,
                      "event_memory": runtime.last_stats.get("event_memory"),
                      "committed_frame_ids": [sorted(ids) for ids in runtime.committed_frame_ids]}
            with (folder / "states.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

        with torch.no_grad():
            (latents, runtime, chunks), timing = timed_cuda(lambda: rollout(
                model, config, batch, manifest["solver_steps"], manifest["cfg_scale"],
                manifest["cached_blocks"], initial_noise=noise, on_chunk=progress,
                on_state=state_record, memory_options=options))
        if len(chunks) != 40 or runtime.commit_count != 41:
            raise AssertionError("expected40 predicted chunks /41 commits")
        if not torch.equal(latents[:, :, :1].cpu(), reference[:, :, :1]):
            raise AssertionError("observed latent changed")
        if {key: tensor_sha256(batch[key]) for key in hashes} != hashes:
            raise AssertionError("rollout mutated conditioning")
        if tensor_sha256(noise) != row["initial_noise_sha256"]:
            raise AssertionError("rollout mutated shared noise")
        generated = latents.cpu()
        result = {"status": "latents-completed", "group": args.group, "case_id": row["case_id"],
                  "step": 750, "seed": row["seed"], "timing": timing,
                  "metrics": latent_metrics(generated, reference, []),
                  "input_sha256": hashes, "initial_noise_sha256": row["initial_noise_sha256"],
                  "checkpoint_sha256": manifest["checkpoint_sha256"],
                  "source_tree_sha256": manifest["source"]["tree_sha256"],
                  "memory_options": asdict(options), "commits": runtime.commit_count,
                  "gpu": {"name": properties.name, "total_bytes": properties.total_memory,
                          "allocator_cap_bytes": cap},
                  "limitations": "one training-seen case/seed; GT metrics do not isolate memory/camera-following/perceptual quality"}
        torch.save({"latents": generated, "case_id": row["case_id"], "group": args.group}, folder / "latents.pt")
        write_json(folder / "result.json", result)
        del batch, noise, latents, runtime, chunks, packet, model
        gc.collect()
        torch.cuda.empty_cache()
        if not args.no_decode:
            decode(config, generated, reference, folder, result, args.decoded_gt_metrics)
        result["status"] = "completed"
        write_json(folder / "result.json", result)
        write_json(folder / "status.json", {"status": "completed", "pid": os.getpid(), "group": args.group})
    except BaseException as error:
        write_json(folder / "status.json", {"status": "failed", "pid": os.getpid(),
                   "group": args.group, "error": repr(error)})
        raise


def decoded_metrics(generated, target, stride=16):
    """RGB PSNR and valid11x11 Gaussian SSIM on selected decoded GT frames.

    This compares VAE reconstructions before MP4 encoding, excluding observed0.
    SSIM uses sigma1.5/C1=.01²/C2=.03² on RGB in[0,1], with no rescaling.
    """
    import torch
    import torch.nn.functional as functional
    if generated.shape != target.shape:
        raise ValueError("decoded GT/generation dimensions differ")
    coordinate = torch.arange(11, dtype=torch.float32) - 5
    kernel = torch.exp(-coordinate.square() / (2 * 1.5**2))
    kernel /= kernel.sum()
    window = (kernel[:, None] * kernel[None, :]).expand(3, 1, 11, 11).contiguous()
    records = []
    for frame in range(stride, len(generated), stride):
        a = torch.from_numpy(generated[frame].copy()).permute(2, 0, 1)[None].float() / 255
        b = torch.from_numpy(target[frame].copy()).permute(2, 0, 1)[None].float() / 255
        mse = (a - b).square().mean().item()
        mu_a, mu_b = (functional.conv2d(x, window, groups=3) for x in (a, b))
        var_a = functional.conv2d(a.square(), window, groups=3) - mu_a.square()
        var_b = functional.conv2d(b.square(), window, groups=3) - mu_b.square()
        cov = functional.conv2d(a * b, window, groups=3) - mu_a * mu_b
        ssim = (((2 * mu_a * mu_b + .01**2) * (2 * cov + .03**2)) /
                ((mu_a.square() + mu_b.square() + .01**2) * (var_a + var_b + .03**2))).mean().item()
        records.append({"raw_frame": frame, "seconds": frame / 16, "rgb_mse": mse,
                        "psnr_db": -10 * math.log10(max(mse, 1e-12)), "ssim": ssim})
    return {"frame_stride": stride, "raw_frame0_excluded": True,
            "target": "VAE-decoded GT latents, before MP4 encoding; not original RGB",
            "psnr_ceiling_db_for_identical_pixels": 120,
            "mean_psnr_db": sum(row["psnr_db"] for row in records) / len(records),
            "mean_ssim": sum(row["ssim"] for row in records) / len(records), "frames": records}


def decode(config, generated, reference, folder, result, with_gt_metrics):
    import imageio.v2 as imageio
    import torch
    from diffusion.model.builder import get_vae, vae_decode
    from sana.tools import resolve_hf_path
    from tools.ttn_decode_comparison import pixels
    dtype = getattr(torch, config.vae.weight_dtype)
    vae = get_vae(config.vae.vae_type, resolve_hf_path(config.vae.vae_pretrained),
                  device="cuda", dtype=dtype, config=config.vae)
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    if hasattr(vae, "use_framewise_encoding"):
        vae.use_framewise_encoding = vae.use_framewise_decoding = True
        vae.tile_sample_stride_num_frames = config.vae.tile_sample_stride_num_frames
        vae.tile_sample_min_num_frames = config.vae.tile_sample_min_num_frames
    with torch.inference_mode():
        video = pixels(vae_decode(config.vae.vae_type, vae, generated.to("cuda", dtype)))
    if video.shape != (961, 704, 1280, 3):
        raise ValueError(f"unexpected decoded video dimensions: {video.shape}")
    output = folder / "video.mp4"
    imageio.mimwrite(output, video, fps=16, codec="libx264", macro_block_size=1,
                    output_params=["-threads", "2"])
    result["video"] = str(output)
    if with_gt_metrics:
        torch.cuda.empty_cache()
        with torch.inference_mode():
            gt = pixels(vae_decode(config.vae.vae_type, vae, reference.to("cuda", dtype)))
        result["decoded_gt_metrics"] = decoded_metrics(video, gt)
        del gt
    del video, vae
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Event memory video] {output}", flush=True)


def launch(args):
    """Launch two independent fixed-GPU workers; main training GPUs are never used."""
    root = args.output.resolve()
    manifest = json.loads((root / "experiment.json").read_text())
    check_sources(manifest)
    if len(args.groups) != len(set(args.groups)):
        raise ValueError("duplicate groups")
    if any((root / group).exists() for group in args.groups):
        raise ValueError("group output already exists; inspect it before retrying")
    free = subprocess.check_output(["nvidia-smi", "-i", "4,5", "--query-gpu=memory.free",
                                    "--format=csv,noheader,nounits"], text=True).splitlines()
    if len(free) != 2 or min(map(int, free)) < args.minimum_free_gib * 1024:
        raise RuntimeError("both GPU4/5 must be sufficiently free immediately before launch")
    state = {"status": "running", "pid": os.getpid(), "groups": args.groups,
             "gpu_queues": {"4": args.groups[::2], "5": args.groups[1::2]}, "workers": []}
    write_json(root / "launcher.json", state)
    queues = {4: list(args.groups[::2]), 5: list(args.groups[1::2])}
    active = {}
    codes = {}
    # Each GPU has at most one worker. No retry, broad process kills, or main-run edits.
    while any(queues.values()) or active:
        for gpu in (4, 5):
            if gpu not in active and queues[gpu]:
                group = queues[gpu].pop(0)
                log = (root / f"{group}.log").open("x", encoding="utf-8")
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                           "--output", str(root), "--group", group,
                           "--allocator-gib", str(args.allocator_gib),
                           "--minimum-free-gib", str(args.minimum_free_gib)]
                if args.no_decode:
                    command.append("--no-decode")
                if args.decoded_gt_metrics:
                    command.append("--decoded-gt-metrics")
                process = subprocess.Popen(command, cwd=ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"),
                    stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (group, process, log)
                state["workers"].append({"group": group, "gpu": gpu, "pid": process.pid, "command": command})
                write_json(root / "launcher.json", state)
                print(f"[Event memory launched] {group} GPU={gpu} PID={process.pid}", flush=True)
        if not active:
            break
        # Wait one second through process.wait, so Python does not spin while GPUs work.
        for gpu, (group, process, log) in list(active.items()):
            try:
                code = process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                continue
            log.close()
            codes[group] = code
            del active[gpu]
            if code:
                queues = {4: [], 5: []}  # allow the other already-running worker to finish
            write_json(root / "worker-exit-codes.json", codes)
    state.update(status="failed" if any(codes.values()) else "workers-completed", exit_codes=codes)
    write_json(root / "launcher.json", state)
    if any(codes.values()):
        raise RuntimeError(f"worker failed: {codes}")
    try:
        compare(args)
    except BaseException as error:
        state.update(status="failed", phase="comparison", error=repr(error))
        write_json(root / "launcher.json", state)
        raise
    state.update(status="completed", phase="comparison")
    write_json(root / "launcher.json", state)


def compare(args):
    import torch
    root = args.output.resolve()
    manifest = json.loads((root / "experiment.json").read_text())
    completed = {group: json.loads((root / group / "result.json").read_text())
                 for group in GROUPS if (root / group / "result.json").exists()}
    if "baseline" not in completed:
        raise ValueError("fresh baseline is required; do not silently reuse prior videos")
    baseline = torch.load(root / "baseline/latents.pt", map_location="cpu", weights_only=True)["latents"]
    original = torch.load(Path(manifest["reference_root"]) / manifest["case"]["name"] / "ttn-latents.pt",
                          map_location="cpu", weights_only=True)["latents"]
    historical = {"exact": torch.equal(baseline, original),
                  "max_abs_difference": (baseline - original).abs().max().item(),
                  "allclose_atol1e-5_rtol1e-4": torch.allclose(baseline, original, atol=1e-5, rtol=1e-4)}
    prefix_frames = min(manifest["frames"], 1 + 3 * (manifest["event_memory"]["start_chunk"] - 1))
    comparisons = {}
    for group, result in completed.items():
        if result["status"] != "completed":
            raise ValueError(f"{group} has incomplete results")
        if result["source_tree_sha256"] != manifest["source"]["tree_sha256"]:
            raise ValueError("group sources differ")
        if any(result[key] != completed["baseline"][key] for key in
               ("input_sha256", "initial_noise_sha256", "checkpoint_sha256", "case_id", "seed")):
            raise ValueError("group inputs/checkpoint differ")
        latents = torch.load(root / group / "latents.pt", map_location="cpu", weights_only=True)["latents"]
        prefix = {"latent_frames": prefix_frames,
                  "exact": torch.equal(latents[:, :, :prefix_frames], baseline[:, :, :prefix_frames]),
                  "max_abs_difference": (latents[:, :, :prefix_frames] - baseline[:, :, :prefix_frames]).abs().max().item()}
        if not prefix["exact"]:
            raise AssertionError(f"{group}: first{prefix_frames} latent frames changed before memory starts")
        names = ("mean_future_latent_mse", "final_chunk_latent_mse", "error_slope_per_latent_frame")
        comparisons[group] = {"prefix_check": prefix,
            "metrics": result["metrics"],
            "minus_baseline": {name: result["metrics"][name] - completed["baseline"]["metrics"][name]
                               for name in names}, "timing": result["timing"]}
    write_json(root / "summary.json", {"status": "completed", "groups": list(completed),
               "historical_baseline_check": historical, "comparison": comparisons,
               "scope": manifest["scope"], "note": "lower GT MSE is not proof of better perceptual quality or causal memory benefit"})
    if not args.no_decode:
        for group, result in completed.items():
            if group != "baseline" and result.get("video") and completed["baseline"].get("video"):
                comparison_video(root / "baseline/video.mp4", root / group / "video.mp4",
                                 root / f"baseline-vs-{group}.mp4", group)
    print(f"[Event memory summary] {root / 'summary.json'}", flush=True)


def comparison_video(baseline, candidate, output, group):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    with imageio.get_reader(baseline) as left, imageio.get_reader(candidate) as right:
        with imageio.get_writer(output, fps=16, codec="libx264", macro_block_size=1,
                                output_params=["-threads", "2"]) as writer:
            count = 0
            for index, (a, b) in enumerate(zip(left, right)):
                if a.shape != b.shape:
                    raise ValueError("comparison videos have different dimensions")
                height, width, _ = a.shape
                canvas = Image.new("RGB", (width * 2, height + 40))
                canvas.paste(Image.fromarray(a), (0, 40))
                canvas.paste(Image.fromarray(b), (width, 40))
                draw = ImageDraw.Draw(canvas)
                draw.text((10, 8), f"StateTTT step750 | memory OFF | {index / 16:.2f}s", font=font, fill="white")
                draw.text((width + 10, 8), f"StateTTT step750 | {group} event replay", font=font, fill="white")
                writer.append_data(np.asarray(canvas))
                count += 1
            if count != 961:
                raise ValueError(f"expected961 comparison frames, got{count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "launch", "compare"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=["baseline", "content"])
    parser.add_argument("--anchors", type=int, nargs="+", default=[11, 15, 19])
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=.1)
    parser.add_argument("--gain", type=float, default=.1)
    parser.add_argument("--max-relative-delta", type=float, default=.05)
    parser.add_argument("--retention", choices=("coverage", "fifo"), default="coverage")
    parser.add_argument("--start-chunk", type=int, default=5)
    parser.add_argument("--max-read-sigma", type=float, default=.5)
    parser.add_argument("--allocator-gib", type=float, default=32.)
    parser.add_argument("--minimum-free-gib", type=float, default=40.)
    parser.add_argument("--no-decode", action="store_true")
    parser.add_argument("--decoded-gt-metrics", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare" and args.reference_root is None:
        parser.error("prepare requires --reference-root")
    if args.mode == "worker" and args.group is None:
        parser.error("worker requires --group")
    if args.allocator_gib <= 0 or args.minimum_free_gib < args.allocator_gib:
        parser.error("require0 < allocator-gib <= minimum-free-gib")
    {"prepare": prepare, "worker": worker, "launch": launch, "compare": compare}[args.mode](args)


if __name__ == "__main__":
    main()
