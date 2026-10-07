"""Decode a paired evaluation prefix with the native SANA VAE (no DiT sampling)."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


def load_pair(directory, case, chunks):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    protocol = summary["protocol"]
    if protocol.get("history_source") != "generated" or protocol.get("ttn_ablation") != "full":
        raise ValueError("requires full TTN and generated-history paired evaluation")
    if chunks < 1:
        raise ValueError("chunks must be positive")
    frames = 1 + 3 * chunks
    payloads = [torch.load(directory / f"case-{case:03d}-{method}.pt",
                           map_location="cpu", weights_only=True) for method in ("sana", "ttn")]
    episodes = []
    for method, payload in zip(("sana", "ttn"), payloads):
        matching = [row for row in summary["episodes"] if row["method"] == method
                    and row["case_id"] == payload["case_id"] and row["seed"] == payload["seed"]]
        if payload["method"] != method or len(matching) != 1:
            raise ValueError("latent payload and evaluation identity disagree")
        episodes.append(matching[0])
        latent = payload["latents"]
        if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[2] < frames:
            raise ValueError(f"{method}: requires [1,C,T,H,W] with T >= {frames}")
        if not torch.isfinite(latent).all():
            raise ValueError(f"{method}: nonfinite saved latents")
    for key in ("case_id", "seed", "input_sha256", "initial_noise_sha256", "base_sha256"):
        if episodes[0][key] != episodes[1][key]:
            raise ValueError(f"paired inputs disagree: {key}")
    a, b = [p["latents"] for p in payloads]
    if a.shape != b.shape or not torch.equal(a[:, :, :1], b[:, :, :1]):
        raise ValueError("paired latent shapes or observed frame disagree")
    return summary, episodes[0], [p["latents"][:, :, :frames].contiguous() for p in payloads]


def pixels(decoded):
    if isinstance(decoded, list):
        decoded = torch.stack(decoded)
    if decoded.ndim != 5 or decoded.shape[:2] != (1, 3) or not torch.isfinite(decoded).all():
        raise ValueError("VAE must return finite [1,3,T,H,W]")
    # Same fixed [-1,1] -> uint8 conversion as SanaWMPipeline; no per-video rescaling.
    return (127.5 * decoded + 127.5).clamp(0, 255).permute(0, 2, 3, 4, 1).to(
        "cpu", dtype=torch.uint8).numpy()[0]


def comparison_frame(left, right, frame, stride, step, font):
    from PIL import Image, ImageDraw
    if left.shape != right.shape:
        raise ValueError("decoded video sizes disagree")
    height, width, _ = left.shape
    canvas = Image.new("RGB", (width * 2, height + 64))
    canvas.paste(Image.fromarray(left), (0, 64))
    canvas.paste(Image.fromarray(right), (width, 64))
    chunk = 0 if frame == 0 else (frame - 1) // (3 * stride) + 1
    note = "Observed frame" if chunk == 0 else f"Predicted chunk {chunk}"
    draw = ImageDraw.Draw(canvas)
    for x, label in ((0, "SANA pretrained"), (width, f"TTN step {step}")):
        draw.text((x + 12, 5), label, fill="white", font=font)
        draw.text((x + 12, 34), note, fill="white", font=font)
    return np.asarray(canvas)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=5, help="predicted chunks, excluding observation")
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--sana-config", type=Path, default=Path("configs/worldttn/sana_teacher_single_gpu.yaml"))
    args = parser.parse_args(argv)
    summary, episode, latents = load_pair(args.evaluation, args.case, args.chunks)
    from worldttn.sana import load_sana_config
    from diffusion.model.builder import get_vae, vae_decode
    from sana.tools import resolve_hf_path
    import imageio.v2 as iio
    import imageio_ffmpeg
    from PIL import ImageFont

    imageio_ffmpeg.get_ffmpeg_exe()  # Fail before loading the VAE if MP4 encoder is unavailable.
    config = load_sana_config(args.sana_config)
    fps = summary["protocol"]["config"]["data"]["target_fps"]
    stride = config.vae.vae_stride[0]
    if stride != summary["protocol"]["config"]["data"]["vae_ratio"][0]:
        raise ValueError("VAE temporal stride differs from the saved evaluation")
    if not fps or fps <= 0 or not torch.cuda.is_available():
        raise ValueError("requires recorded positive FPS and a CUDA Slurm allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    dtype = getattr(torch, config.vae.weight_dtype)
    source = resolve_hf_path(config.vae.vae_pretrained)
    vae = get_vae(config.vae.vae_type, source, device="cuda", dtype=dtype, config=config.vae)
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    if hasattr(vae, "use_framewise_encoding"):
        vae.use_framewise_encoding = vae.use_framewise_decoding = True
        vae.tile_sample_stride_num_frames = config.vae.tile_sample_stride_num_frames
        vae.tile_sample_min_num_frames = config.vae.tile_sample_min_num_frames
    videos, times = [], []
    with torch.inference_mode():
        for method, latent in zip(("sana", "ttn"), latents):
            torch.cuda.synchronize()
            started = time.perf_counter()
            # Decode the whole prefix together, preserving native temporal decoder behavior.
            video = pixels(vae_decode(config.vae.vae_type, vae, latent.to("cuda", dtype)))
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
            if video.shape[0] != (latent.shape[2] - 1) * stride + 1:
                raise ValueError("decoded raw frame count differs from the VAE stride")
            with iio.get_writer(args.output / f"{method}.mp4", fps=fps, codec="libx264", macro_block_size=1) as writer:
                for frame in video:
                    writer.append_data(frame)
            videos.append(video)
            print(f"[TTN video] {method}: {video.shape}, decode={times[-1]:.1f}s", flush=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    with iio.get_writer(args.output / "comparison.mp4", fps=fps, codec="libx264", macro_block_size=1) as writer:
        for index, (left, right) in enumerate(zip(*videos)):
            writer.append_data(comparison_frame(left, right, index, stride, summary["protocol"]["step"], font))
    metadata = {"status": "completed", "source_evaluation": str(args.evaluation.resolve()),
                "kind": "decoded prefix of existing rollout; no new diffusion sampling or refiner",
                "predicted_chunks": args.chunks, "latent_frames": latents[0].shape[2],
                "raw_frames": len(videos[0]), "fps": fps, "resolution_hw": list(videos[0].shape[1:3]),
                "protocol": summary["protocol"], "case_id": episode["case_id"], "seed": episode["seed"],
                "input_sha256": episode["input_sha256"], "initial_noise_sha256": episode["initial_noise_sha256"],
                "vae": asdict(config.vae), "sana_config_sha256": hashlib.sha256(args.sana_config.read_bytes()).hexdigest(),
                "decode_seconds": dict(zip(("sana", "ttn"), times)), "files": ["sana.mp4", "ttn.mp4", "comparison.mp4"]}
    (args.output / "comparison.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[TTN video] completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
