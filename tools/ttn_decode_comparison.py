"""Decode a paired evaluation prefix with the native SANA VAE (no DiT sampling)."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


def load_pair(directory, case, chunks, *, left_evaluation=None, left_method="sana"):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    protocol = summary["protocol"]
    if protocol.get("history_source") != "generated" or protocol.get("ttn_ablation") != "full":
        raise ValueError("requires full TTN and generated-history paired evaluation")
    if chunks < 1:
        raise ValueError("chunks must be positive")
    frames = 1 + 3 * chunks
    if left_method not in ("sana", "ttn"):
        raise ValueError("left method must be sana or ttn")
    if left_method == "ttn" and left_evaluation is None:
        raise ValueError("TTN baseline comparison requires --left-evaluation")
    left_directory = Path(left_evaluation) if left_evaluation is not None else directory
    left_summary = json.loads((left_directory / "summary.json").read_text())
    if left_evaluation is not None:
        left_protocol = left_summary["protocol"]
        for key in ("history_source", "ttn_ablation", "step", "checkpoint_sha256", "case", "seed",
                    "steps", "cfg_scale", "cached_blocks", "cross_attn_backend", "flow_shift"):
            if left_protocol.get(key) != protocol.get(key):
                raise ValueError(f"left evaluation protocol differs: {key}")
        sink = left_protocol.get("tla_sink", {})
        if sink.get("mode", "off") != "off" and sink.get("gain", 1) != 0:
            raise ValueError("left baseline must have sink disabled")
    payloads = [torch.load(folder / f"case-{case:03d}-{method}.pt", map_location="cpu", weights_only=True)
                for folder, method in ((left_directory, left_method), (directory, "ttn"))]
    episodes = []
    for method, source, payload in zip((left_method, "ttn"), (left_summary, summary), payloads):
        matching = [row for row in source["episodes"] if row["method"] == method
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


def verify_video(path, frames, size_hw, fps):
    """Reject an incomplete MP4 before publishing a completed video result."""
    import imageio.v2 as iio
    with iio.get_reader(path) as reader:
        count, metadata = reader.count_frames(), reader.get_meta_data()
        if count != frames:
            raise ValueError(f"{path}: encoded frame count {count} != {frames}")
        if tuple(metadata["size"]) != tuple(reversed(size_hw)) or abs(metadata["fps"] - fps) > 1e-4:
            raise ValueError(f"{path}: encoded resolution/FPS differs")
        for index in (0, count - 1):
            if reader.get_data(index).shape != (*size_hw, 3):
                raise ValueError(f"{path}: encoded boundary frame cannot be decoded")
    return {"frames": count, "resolution_hw": list(size_hw), "fps": metadata["fps"],
            "bytes": Path(path).stat().st_size}


def comparison_frame(left, right, frame, stride, step, font, *, labels=None):
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
    labels = labels or ("SANA pretrained", f"TTN step {step}")
    for x, label in zip((0, width), labels):
        draw.text((x + 12, 5), label, fill="white", font=font)
        draw.text((x + 12, 34), note, fill="white", font=font)
    return np.asarray(canvas)


def save_return_views(output, videos, protocol, font, *, labels=None, method_names=("sana", "ttn")):
    """Compare decoded initial/returned views before lossy MP4 encoding."""
    case = protocol.get("case", {})
    camera = case.get("camera", {})
    if camera.get("trajectory") != "closed_orbit" or len(videos[0]) != case["raw_frames"]:
        return None  # A prefix that has not returned cannot measure loop consistency.
    from PIL import Image, ImageDraw
    start = case["raw_frames"] - camera["end_hold_raw_frames"]
    height, width, _ = videos[0].shape[1:]
    canvas = Image.new("RGB", (2 * width, 2 * (height + 64)))
    draw = ImageDraw.Draw(canvas)
    metrics = {"reference": "decoded observed frame 0; RGB in [0,1], before MP4 encoding",
               "return_raw_frame_range": [start, case["raw_frames"]],
               "note": "Initial-view consistency only; no future GT. A static video can also score well: inspect camera motion separately.",
               "methods": {}}
    labels = labels or ("SANA pretrained", f"TTN step {protocol['step']}")
    for row, (method, video) in enumerate(zip(method_names, videos)):
        reference = video[0].astype(np.float32) / 255
        errors = [float(np.square(frame.astype(np.float32) / 255 - reference).mean())
                  for frame in video[start:]]
        metrics["methods"][method] = {"mean_return_to_observed_rgb_mse": float(np.mean(errors)),
                                      "final_return_to_observed_rgb_mse": errors[-1]}
        top = row * (height + 64)
        label = labels[row]
        for x, frame, name in ((0, video[0], "Observed reconstruction"),
                                (width, video[-1], "Returned view")):
            canvas.paste(Image.fromarray(frame), (x, top + 64))
            draw.text((x + 12, top + 5), label, fill="white", font=font)
            draw.text((x + 12, top + 34), name, fill="white", font=font)
    canvas.save(output / "return-comparison.png")
    (output / "return-view.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def comparison_labels(protocol, left_method="sana"):
    baseline = "SANA pretrained" if left_method == "sana" else f"TTN step {protocol['step']} | sink off"
    label = f"TTN step {protocol['step']}"
    sink = protocol.get("tla_sink", {})
    if sink.get("mode", "off") != "off" and sink.get("gain", 0) > 0:
        label += f" | {sink['mode']} g={sink['gain']:g} | {sink['position']}"
    return baseline, label


def save_contact_sheet(output, videos, fps, stride, protocol, font, labels):
    from PIL import Image
    indices = [round(second * fps) for second in (5, 10, 13, 15, 20, 23, 25, 30)
               if round(second * fps) < len(videos[0])]
    if not indices:
        return None
    rows = [comparison_frame(videos[0][i], videos[1][i], i, stride, protocol["step"], font,
                             labels=(f"{labels[0]} | t={i/fps:g}s", f"{labels[1]} | t={i/fps:g}s")) for i in indices]
    height, width = rows[0].shape[:2]
    sheet = Image.new("RGB", (width, height * len(rows)))
    for row, pixels_ in enumerate(rows):
        sheet.paste(Image.fromarray(pixels_), (0, row * height))
    sheet.save(output / "contact-sheet.jpg", quality=92)
    return {"file": "contact-sheet.jpg", "raw_frame_indices": indices}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=5, help="predicted chunks, excluding observation")
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--sana-config", type=Path, default=Path("configs/worldttn/sana_teacher_single_gpu.yaml"))
    parser.add_argument("--left-evaluation", type=Path, help="optional completed baseline for the left panel")
    parser.add_argument("--left-method", choices=("sana", "ttn"), default="sana")
    args = parser.parse_args(argv)
    summary, episode, latents = load_pair(args.evaluation, args.case, args.chunks,
                                        left_evaluation=args.left_evaluation, left_method=args.left_method)
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
    method_names = ("sana" if args.left_method == "sana" else "ttn_baseline", "ttn")
    labels = comparison_labels(summary["protocol"], args.left_method)
    with torch.inference_mode():
        for method, latent in zip(method_names, latents):
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
            writer.append_data(comparison_frame(left, right, index, stride, summary["protocol"]["step"], font, labels=labels))
    encoded = {f"{method}.mp4": verify_video(args.output / f"{method}.mp4", len(video), video.shape[1:3], fps)
               for method, video in zip(method_names, videos)}
    encoded["comparison.mp4"] = verify_video(args.output / "comparison.mp4", len(videos[0]),
        (videos[0].shape[1] + 64, videos[0].shape[2] * 2), fps)
    metadata = {"status": "completed", "source_evaluation": str(args.evaluation.resolve()),
                "kind": "decoded prefix of existing rollout; no new diffusion sampling or refiner",
                "predicted_chunks": args.chunks, "latent_frames": latents[0].shape[2],
                "raw_frames": len(videos[0]), "fps": fps, "resolution_hw": list(videos[0].shape[1:3]),
                "protocol": summary["protocol"], "case_id": episode["case_id"], "seed": episode["seed"],
                "input_sha256": episode["input_sha256"], "initial_noise_sha256": episode["initial_noise_sha256"],
                "vae": asdict(config.vae), "sana_config_sha256": hashlib.sha256(args.sana_config.read_bytes()).hexdigest(),
                "decode_seconds": dict(zip(method_names, times)),
                "left_evaluation": str(args.left_evaluation.resolve()) if args.left_evaluation else None,
                "encoded_video_validation": encoded,
                "labels": list(labels), "files": [f"{method}.mp4" for method in method_names] + ["comparison.mp4"]}
    contact = save_contact_sheet(args.output, videos, fps, stride, summary["protocol"], font, labels)
    if contact is not None:
        metadata["contact_sheet"] = contact
        metadata["files"].append(contact["file"])
    returned = save_return_views(args.output, videos, summary["protocol"], font,
                                 labels=labels, method_names=method_names)
    if returned is not None:
        metadata["return_view"] = returned
        metadata["files"] += ["return-comparison.png", "return-view.json"]
    (args.output / "comparison.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[TTN video] completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
