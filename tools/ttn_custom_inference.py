"""Paired SANA/TTN image-to-video inference without reading a training case."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from worldttn.evaluation import diagnostic_noise, file_sha256, load_evaluation_run, tensor_sha256

REPO = Path(__file__).resolve().parents[1]


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
    if (camera["trajectory"] != "static" or camera["intrinsics_grid_hw"] != [height, width]
            or intr.shape != (4,) or not np.isfinite(intr).all() or np.any(intr[:2] <= 0)
            or not 0 <= intr[2] <= width or not 0 <= intr[3] <= height):
        raise ValueError("invalid fixed virtual camera")
    image = path.parent / case["image"]
    if file_sha256(image) != case["image_sha256"]:
        raise ValueError("custom first-frame SHA256 mismatch")
    prompt = (path.parent / case["prompt_file"]).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("empty custom prompt")
    return case, image, prompt


def make_geometry(case, vae_stride):
    from inference_video_scripts.wm.inference_sana_wm import prepare_camera
    height, width = case["target_size_hw"]
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], case["raw_frames"], axis=0)
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
    parser.set_defaults(adapter=None, stage=None, dataset_root=None, data_dir=None, vae_cache_dir=None)
    args = parser.parse_args(argv)
    case, image_path, prompt = load_case(args.case)
    if not torch.cuda.is_available():
        raise ValueError("submit this command with a single CUDA Slurm allocation")
    if args.steps < 1 or not np.isfinite(args.cfg_scale) or args.cfg_scale < 1:
        raise ValueError("invalid sampling steps/CFG")
    run, config, ttn, adapter, digest, last_train = load_evaluation_run(args)
    if ttn.camera_attention != "sana" or list(config.vae.vae_stride) != [8, 32, 32]:
        raise ValueError("custom paired inference requires native SANA camera and LTX [8,32,32]")
    from worldttn.cli import rollout, seed_everything, timed_cuda, to_device
    from worldttn.sana import build_sana, configure_cross_attention
    from worldttn.checkpoint import load_checkpoint
    from worldttn.performance import configure_execution, ExecutionOptions
    from diffusion.model.builder import get_tokenizer_and_text_encoder
    from train_video_scripts.train_sana_wm_stage1 import _encode_prompts

    args.output.mkdir(parents=True, exist_ok=False)
    seed_everything(case["seed"])
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
    shape = (1, config.vae.vae_latent_dim, case["latent_frames"], *batch["initial_latent"].shape[-2:])
    noise = diagnostic_noise(shape, "cuda", case["seed"]).cpu()
    inputs = {key: tensor_sha256(batch[key]) for key in
              ("initial_latent", "y", "mask", "uncondition", "camera_conditions", "chunk_plucker")}
    torch.save(batch, args.output / "input-bundle.pt")
    (args.output / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (args.output / "case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    protocol = {"scope": "custom AI-generated first-frame qualitative test; no GT future video",
                "status": "running", "training_run": str(args.training_run.resolve()),
                "checkpoint": str(adapter), "checkpoint_sha256": digest, "step": last_train["step"],
                "frames": case["latent_frames"], "raw_frames": case["raw_frames"], "seed": case["seed"],
                "steps": args.steps, "cfg_scale": args.cfg_scale, "cached_blocks": args.cached_blocks,
                "flow_shift": config.scheduler.inference_flow_shift, "cross_attn_backend": args.cross_attn_backend,
                "camera_attention": "sana", "history_source": "generated", "ttn_ablation": "full",
                "refiner": None, "execution": "reference/reference", "no_training_dataset_read": True,
                "case": case, "prompt": prompt, "metric_space": "none: no ground-truth future",
                "config": {"model": asdict(config.model), "text_encoder": asdict(config.text_encoder),
                           "scheduler": asdict(config.scheduler), "vae": asdict(config.vae),
                           "data": {"target_fps": case["fps"], "vae_ratio": [8, 32]}},
                "torch": torch.__version__, "cuda": torch.version.cuda}
    (args.output / "manifest.json").write_text(json.dumps({"protocol": protocol}, indent=2), encoding="utf-8")
    episodes = []
    for method in ("sana", "ttn"):
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
        with torch.no_grad():
            (generated, runtime, chunks), timing = timed_cuda(lambda: rollout(
                model, config, gpu_batch, args.steps, args.cfg_scale, args.cached_blocks,
                initial_noise=gpu_noise, on_chunk=progress))
        if not torch.equal(generated[:, :, :1].cpu(), batch["initial_latent"]):
            raise ValueError("sampling changed the observed latent")
        if {key: tensor_sha256(gpu_batch[key]) for key in inputs} != inputs or tensor_sha256(gpu_noise) != tensor_sha256(noise):
            raise ValueError("sampling mutated shared conditioning/noise")
        row = {"method": method, "case_id": case["case_id"], "seed": case["seed"],
               "input_sha256": inputs, "initial_noise_sha256": tensor_sha256(noise),
               "base_sha256": model.base_load_report["sha256"], "chunks": chunks, "timing": timing,
               "commits": runtime.commit_count if runtime else None}
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
    decode(["--evaluation", str(args.output), "--output", str(args.output / "videos"),
            "--chunks", str((case["latent_frames"] - 1) // 3), "--sana-config", str(source)])
    print(f"[TTN custom completed] {args.output / 'videos/comparison.mp4'}", flush=True)


if __name__ == "__main__":
    main()
