"""Paired SANA/TTN image-to-video inference without reading a training case."""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch

from worldttn.evaluation import (diagnostic_noise, file_sha256, load_evaluation_run, tensor_sha256,
                                validate_inference_interventions)

REPO = Path(__file__).resolve().parents[1]
CONDITIONING_KEYS = ("initial_latent", "y", "mask", "uncondition", "camera_conditions", "chunk_plucker")


def link_or_copy(source, target):
    """Immutable inference artifacts can share an inode; never overwrite a destination."""
    if target.exists():
        raise FileExistsError(target)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def load_shared_inputs(directory, case, prompt, checkpoint_digest, args, flow_shift):
    """Reuse exactly the prepared observation/text/geometry of a completed baseline."""
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    protocol = summary["protocol"]
    expected = dict(case=case, prompt=prompt, checkpoint_sha256=checkpoint_digest, status="completed",
                    steps=args.steps, cfg_scale=args.cfg_scale, cached_blocks=args.cached_blocks,
                    cross_attn_backend=args.cross_attn_backend, flow_shift=flow_shift,
                    execution="reference/reference", camera_attention="sana",
                    history_source="generated", ttn_ablation="full")
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("shared input baseline protocol differs")
    source_sink = protocol.get("tla_sink", {})
    if source_sink.get("mode", "off") != "off" and source_sink.get("gain", 1) != 0:
        raise ValueError("shared input baseline must have sink disabled")
    rows = {row["method"]: row for row in summary["episodes"]}
    if len(summary["episodes"]) != 2 or set(rows) != {"sana", "ttn"}:
        raise ValueError("shared input baseline requires exactly one completed SANA/TTN pair")
    for key in ("input_sha256", "initial_noise_sha256", "case_id", "seed", "base_sha256"):
        if rows["sana"][key] != rows["ttn"][key]:
            raise ValueError("shared input baseline paired identity differs")
    path = directory / "input-bundle.pt"
    if protocol.get("input_bundle_sha256") and file_sha256(path) != protocol["input_bundle_sha256"]:
        raise ValueError("shared conditioning bundle SHA256 differs")
    batch = torch.load(path, map_location="cpu", weights_only=True)
    if ({key: tensor_sha256(batch[key]) for key in CONDITIONING_KEYS} != rows["ttn"]["input_sha256"]
            or any(not torch.isfinite(batch[key]).all() for key in CONDITIONING_KEYS)
            or batch["camera_conditions"].shape != (1, case["latent_frames"], 20)):
        raise ValueError("shared conditioning differs from the completed baseline")
    if "target_size_hw" in case and [batch["height"], batch["width"]] != case["target_size_hw"]:
        raise ValueError("shared conditioning image dimensions differ")
    from tools.ttn_decode_comparison import load_pair
    _, _, latents = load_pair(directory, 0, (case["latent_frames"]-1)//3)
    if not torch.equal(latents[0][:, :, :1], batch["initial_latent"]):
        raise ValueError("shared observed frame differs from saved baseline latents")
    return batch, summary


def load_case(path):
    path = Path(path).resolve()
    case = json.loads(path.read_text(encoding="utf-8"))
    if case.get("format") != "TTN-custom-image-case-v1":
        raise ValueError("unsupported custom case format")
    height, width = case["target_size_hw"]
    frames = case["latent_frames"]
    if (height, width) != (704, 1280) or frames < 4 or frames % 3 != 1:
        raise ValueError("requires native 704x1280 and 1+3n latent frames")
    if case["raw_frames"] != (frames - 1) * 8 + 1 or case["fps"] != 16:
        raise ValueError("raw/latent frame count or 16-fps protocol disagrees")
    camera = case["camera"]
    intr = np.asarray(camera["intrinsics_px"], dtype=np.float32)
    if (camera["intrinsics_grid_hw"] != [height, width]
            or intr.shape != (4,) or not np.isfinite(intr).all() or np.any(intr[:2] <= 0)
            or not 0 <= intr[2] <= width or not 0 <= intr[3] <= height):
        raise ValueError("invalid virtual camera intrinsics")
    camera_poses(case)  # Validate the trajectory before allocating a model.
    image = path.parent / case["image"]
    if file_sha256(image) != case["image_sha256"]:
        raise ValueError("custom first-frame SHA256 mismatch")
    prompt = (path.parent / case["prompt_file"]).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("empty custom prompt")
    return case, image, prompt


def camera_poses(case):
    """OpenCV C2W: horizontal translated orbit, looking at a fixed circle centre."""
    camera, frames = case["camera"], case["raw_frames"]
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
    if camera["trajectory"] == "static":
        return poses
    if camera["trajectory"] != "closed_orbit":
        raise ValueError("unsupported custom camera trajectory")
    radius = camera["radius"]
    start, hold = camera["start_hold_raw_frames"], camera["end_hold_raw_frames"]
    if (not np.isfinite(radius) or radius <= 0 or
            any(type(n) is not int or n < 8 or n % 8 for n in (start, hold)) or
            start + hold >= frames - 1):
        raise ValueError("closed orbit requires positive radius and valid 8-frame-aligned holds")
    end = frames - 1 - hold
    phase = np.clip((np.arange(frames, dtype=np.float64) - start) / (end - start), 0, 1)
    angle = 2 * np.pi * (3 * phase**2 - 2 * phase**3)  # Smooth start/stop.
    c, s = np.cos(angle), np.sin(angle)
    poses[:, 0, 0], poses[:, 0, 2] = c, -s
    poses[:, 2, 0], poses[:, 2, 2] = s, c
    poses[:, 0, 3], poses[:, 2, 3] = radius * s, radius * (1 - c)
    # Exact closure, including intrinsics/rays; avoid sin(2*pi) rounding drift.
    poses[:start + 1] = poses[end:] = np.eye(4, dtype=np.float32)
    return poses


def make_geometry(case, vae_stride):
    from inference_video_scripts.wm.inference_sana_wm import prepare_camera
    height, width = case["target_size_hw"]
    poses = camera_poses(case)
    intr = np.repeat(np.asarray(case["camera"]["intrinsics_px"], dtype=np.float32)[None], len(poses), axis=0)
    packed = prepare_camera(poses, intr, target_size=(height, width), vae_stride=vae_stride)
    camera = packed["raymap"].clone()
    # Native prepare_camera returns latent-grid intrinsics. rollout/TTNSession
    # require pixel-grid intrinsics and perform their own scaling exactly once.
    camera[:, 16:] *= camera.new_tensor([width / (width // vae_stride[-1]),
                                        height / (height // vae_stride[-1])] * 2)
    if camera.shape != (case["latent_frames"], 20):
        raise ValueError("camera/frame count mismatch")
    return {"camera_conditions": camera[None], "chunk_plucker": packed["chunk_plucker"][None]}


def return_latent_metrics(generated, batch, case):
    """Initial-view consistency, not future-GT accuracy or proof of camera following."""
    if case["camera"]["trajectory"] != "closed_orbit":
        return None
    # Exclude the closing latent's raw interval, which still contains movement.
    first_raw = case["raw_frames"] - case["camera"]["end_hold_raw_frames"]
    ids = list(range((first_raw + 7) // 8, case["latent_frames"]))
    camera, rays = batch["camera_conditions"], batch["chunk_plucker"]
    if (not ids or not torch.equal(camera[:, ids], camera[:, :1].expand(-1, len(ids), -1)) or
            not torch.equal(rays[:, :, ids], rays[:, :, :1].expand(-1, -1, len(ids), -1, -1))):
        raise ValueError("return window camera/rays do not match the observed view")
    errors = (generated[:, :, ids].float() - generated[:, :, :1].float()).square().mean((0, 1, 3, 4))
    return {"reference": "observed latent frame 0; no future GT",
            "return_latent_ids": ids, "per_frame_latent_mse": errors.cpu().tolist(),
            "mean_return_to_observed_latent_mse": errors.mean().item(),
            "final_return_to_observed_latent_mse": errors[-1].item()}


def encode_first_frame(case, image_path, config, output):
    from diffusion.model.builder import get_vae, vae_encode
    from inference_video_scripts.wm.inference_sana_wm import resize_and_center_crop
    from sana.tools import resolve_hf_path
    height, width = case["target_size_hw"]
    with Image.open(image_path) as image:
        image, _, _, _ = resize_and_center_crop(image.convert("RGB"), height, width)
        image.save(output / "first_frame_used.png")
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 127.5 - 1
    dtype = getattr(torch, config.vae.weight_dtype)
    vae = get_vae(config.vae.vae_type, resolve_hf_path(config.vae.vae_pretrained),
                  device="cuda", dtype=dtype, config=config.vae)
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    if hasattr(vae, "use_framewise_encoding"):
        vae.use_framewise_encoding = True
    with torch.no_grad():
        initial = vae_encode(config.vae.vae_type, vae, pixels[None, :, None].to("cuda", dtype),
                             sample_posterior=False, device="cuda").float().cpu()
    if (initial.shape != (1, config.vae.vae_latent_dim, 1, height // 32, width // 32)
            or not torch.isfinite(initial).all()):
        raise ValueError("first-frame VAE shape mismatch")
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    return initial


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True, help="immutable completed model snapshot")
    parser.add_argument("--case", type=Path, default=REPO / "assets/worldttn/study_static/case.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO / "configs/worldttn/dual_psi_meta.json")
    parser.add_argument("--sana-config", type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument("--cached-blocks", type=int, default=2)
    parser.add_argument("--cross-attn-backend", choices=("math", "auto"), default="math")
    parser.add_argument("--reuse-baseline", type=Path,
                        help="completed paired custom run; reuse its observation/text/geometry and SANA rollout")
    parser.add_argument("--state-diagnostics", action="store_true", help="detached state/sink telemetry")
    parser.add_argument("--check-sink-smoke", action="store_true",
                        help="after video encoding, require the step100 five-chunk sink acceptance checks")
    from worldttn.sink import add_sink_arguments
    add_sink_arguments(parser)
    parser.set_defaults(adapter=None, stage=None, dataset_root=None, data_dir=None, vae_cache_dir=None)
    args = parser.parse_args(argv)
    case, image_path, prompt = load_case(args.case)
    if args.check_sink_smoke and (case["latent_frames"] != 16 or args.steps != 4 or not args.state_diagnostics):
        parser.error("--check-sink-smoke requires 16 latents, --steps 4 and --state-diagnostics")
    if not torch.cuda.is_available():
        raise ValueError("submit this command with a single CUDA Slurm allocation")
    if args.steps < 1 or not np.isfinite(args.cfg_scale) or args.cfg_scale < 1:
        raise ValueError("invalid sampling steps/CFG")
    run, config, ttn, adapter, digest, last_train = load_evaluation_run(args)
    sink = validate_inference_interventions(args, ttn)
    if ttn.camera_attention != "sana" or list(config.vae.vae_stride) != [8, 32, 32]:
        raise ValueError("custom paired inference requires native SANA camera and LTX [8,32,32]")
    from worldttn.cli import rollout, seed_everything, timed_cuda, to_device
    from worldttn.sana import build_sana, configure_cross_attention
    from worldttn.checkpoint import load_checkpoint
    from worldttn.performance import configure_execution, ExecutionOptions
    from diffusion.model.builder import get_tokenizer_and_text_encoder
    from train_video_scripts.train_sana_wm_stage1 import _encode_prompts

    shared_summary = None
    if args.reuse_baseline:
        batch, shared_summary = load_shared_inputs(args.reuse_baseline, case, prompt, digest, args,
                                                  config.scheduler.inference_flow_shift)
    args.output.mkdir(parents=True, exist_ok=False)
    seed_everything(case["seed"])
    np.save(args.output / "camera_poses.npy", camera_poses(case))
    if shared_summary is None:
        batch = make_geometry(case, config.vae.vae_stride)
        batch["initial_latent"] = encode_first_frame(case, image_path, config, args.output)
        tokenizer, encoder = get_tokenizer_and_text_encoder(config.text_encoder.text_encoder_name, "cpu")
        encoder.eval().requires_grad_(False)
        batch["y"], batch["mask"] = _encode_prompts([prompt], tokenizer, encoder, config, "cpu")
        batch["uncondition"], _ = _encode_prompts([""], tokenizer, encoder, config, "cpu")
        del tokenizer, encoder
        gc.collect()
        batch.update(width=case["target_size_hw"][1], height=case["target_size_hw"][0],
                     data_info={"img_hw": torch.tensor([case["target_size_hw"]], dtype=torch.float32)})
    else:
        for name in ("first_frame_used.png", "case-000-sana.pt"):
            if (args.reuse_baseline / name).is_file():
                link_or_copy(args.reuse_baseline / name, args.output / name)
    shape = (1, config.vae.vae_latent_dim, case["latent_frames"], *batch["initial_latent"].shape[-2:])
    noise = diagnostic_noise(shape, "cuda", case["seed"]).cpu()
    inputs = {key: tensor_sha256(batch[key]) for key in CONDITIONING_KEYS}
    if shared_summary is None:
        torch.save(batch, args.output / "input-bundle.pt")
    else:
        link_or_copy(args.reuse_baseline / "input-bundle.pt", args.output / "input-bundle.pt")
        baseline_noise = shared_summary["episodes"][0]["initial_noise_sha256"]
        if tensor_sha256(noise) != baseline_noise:
            raise ValueError("shared baseline initial noise differs")
    (args.output / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (args.output / "case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    from worldttn.provenance import implementation_identity
    protocol = {"scope": "custom AI-generated first-frame qualitative test; no GT future video",
                "status": "running", "training_run": str(args.training_run.resolve()),
                "checkpoint": str(adapter), "checkpoint_sha256": digest, "step": last_train["step"],
                "frames": case["latent_frames"], "raw_frames": case["raw_frames"], "seed": case["seed"],
                "steps": args.steps, "cfg_scale": args.cfg_scale, "cached_blocks": args.cached_blocks,
                "flow_shift": config.scheduler.inference_flow_shift, "cross_attn_backend": args.cross_attn_backend,
                "camera_attention": "sana", "history_source": "generated", "ttn_ablation": "full",
                "stage": ttn.stage, "tla_sink": asdict(sink), "state_diagnostics": args.state_diagnostics,
                "ttn_config": ttn.to_dict(), "provenance": implementation_identity(),
                "compile": {key: os.environ.get(key, "0") for key in ("GDN_DISABLE_COMPILE", "GDN_DISABLE_COMPLEX_COMPILE")},
                "input_bundle_sha256": file_sha256(args.output / "input-bundle.pt"),
                "shared_baseline": str(args.reuse_baseline.resolve()) if args.reuse_baseline else None,
                "refiner": None, "execution": "reference/reference", "no_training_dataset_read": True,
                "case": case, "prompt": prompt, "metric_space": "none: no ground-truth future",
                "config": {"model": asdict(config.model), "text_encoder": asdict(config.text_encoder),
                           "scheduler": asdict(config.scheduler), "vae": asdict(config.vae),
                           "data": {"target_fps": case["fps"], "vae_ratio": [8, 32]}},
                "torch": torch.__version__, "cuda": torch.version.cuda}
    (args.output / "manifest.json").write_text(json.dumps({"protocol": protocol}, indent=2), encoding="utf-8")
    episodes = [] if shared_summary is None else [next(row for row in shared_summary["episodes"] if row["method"] == "sana")]
    if episodes:
        (args.output / "episodes.jsonl").write_text(json.dumps(episodes[0]) + "\n", encoding="utf-8")
    for method in (("sana", "ttn") if shared_summary is None else ("ttn",)):
        seed_everything(case["seed"])
        options = {"install_adapter": method == "ttn"}
        if method == "ttn" and (last_train.get("weight_scope") == "dit" or
                last_train.get("train_scope") in ("dit", "ttn-visual")):
            options["dtype"] = torch.float32
        model = build_sana(config, ttn, run["base"]["source"], "cuda", **options)
        if model.base_load_report["sha256"] != run["base"]["sha256"]:
            raise ValueError("original SANA base differs from the checkpoint")
        if method == "ttn":
            configure_execution(model, ExecutionOptions())
            load_checkpoint(adapter, model)
        configure_cross_attention(model, args.cross_attn_backend)
        model.eval().requires_grad_(False)
        gpu_batch = to_device(batch, "cuda")
        gpu_noise = noise.to("cuda")
        def progress(row):
            print("[TTN custom chunk] " + json.dumps({"method": method,
                  **{key: row[key] for key in ("chunk", "start", "end", "seconds")}}), flush=True)
        runtime_options = {}
        if method == "ttn" and (sink.active or args.state_diagnostics):
            runtime_options.update(sink_options=sink, state_diagnostics=args.state_diagnostics)
        with torch.no_grad():
            (generated, runtime, chunks), timing = timed_cuda(lambda: rollout(
                model, config, gpu_batch, args.steps, args.cfg_scale, args.cached_blocks,
                initial_noise=gpu_noise, on_chunk=progress, **runtime_options))
        if not torch.equal(generated[:, :, :1].cpu(), batch["initial_latent"]):
            raise ValueError("sampling changed the observed latent")
        if {key: tensor_sha256(gpu_batch[key]) for key in inputs} != inputs or tensor_sha256(gpu_noise) != tensor_sha256(noise):
            raise ValueError("sampling mutated shared conditioning/noise")
        row = {"method": method, "case_id": case["case_id"], "seed": case["seed"],
               "input_sha256": inputs, "initial_noise_sha256": tensor_sha256(noise),
               "base_sha256": model.base_load_report["sha256"], "chunks": chunks, "timing": timing,
               "commits": runtime.commit_count if runtime else None,
               "sink_reference_sha256": getattr(runtime, "sink_reference_sha256", None),
               "sink_reference_verified": getattr(runtime, "sink_reference_verified", None)
                   if getattr(runtime, "sink_reference_sha256", None) is not None else None,
               "return_view": return_latent_metrics(generated, gpu_batch, case)}
        torch.save({"latents": generated.cpu(), "method": method, "case_id": case["case_id"],
                    "seed": case["seed"], "chunks": [{key: c[key] for key in ("chunk", "start", "end")} for c in chunks]},
                   args.output / f"case-000-{method}.pt")
        episodes.append(row)
        with (args.output / "episodes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        del model, gpu_batch, gpu_noise, generated, runtime
        gc.collect()
        torch.cuda.empty_cache()
        print("[TTN custom method] " + json.dumps({"method": method, **timing}), flush=True)
    if file_sha256(adapter) != digest or file_sha256(image_path) != case["image_sha256"]:
        raise ValueError("checkpoint/input image changed during inference")
    protocol["status"] = "completed"
    (args.output / "summary.json").write_text(json.dumps({"protocol": protocol, "episodes": episodes,
        "metrics": None, "metric_note": "No future GT: latent GT MSE is unavailable."}, indent=2), encoding="utf-8")
    from tools.ttn_decode_comparison import main as decode
    source = args.sana_config or run["arguments"].get("sana_config") or json.loads(args.config.read_text())["sana_config"]
    source = Path(source)
    if not source.is_absolute():
        source = REPO / source
    decode_args = ["--evaluation", str(args.output), "--output", str(args.output / "videos"),
                   "--chunks", str((case["latent_frames"] - 1) // 3), "--sana-config", str(source)]
    decode(decode_args)
    if args.reuse_baseline:
        decode([*decode_args[:3], str(args.output / "videos-vs-ttn-baseline"), *decode_args[4:],
                "--left-evaluation", str(args.reuse_baseline), "--left-method", "ttn"])
    if args.check_sink_smoke:
        from tools.ttn_submit_sink import main as validate_smoke
        validate_smoke(["--check-smoke", str(args.output)])
    print(f"[TTN custom completed] {args.output / 'videos/comparison.mp4'}", flush=True)


if __name__ == "__main__":
    main()
