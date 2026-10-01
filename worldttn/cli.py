"""Single-card server entrypoints. --help does not import the CUDA SANA stack."""
import argparse
import json
import os
import random
import time
from pathlib import Path
from dataclasses import replace
import torch
from .core import TTNConfig, BASE_ID, ANCHORS
from .checkpoint import make_optimizer, save_checkpoint, load_checkpoint
from .training import train_clip, SANAFlowLoss, chunk_ranges
from .session import TTNSession

ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def read_reference(path, stage=None):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    config = TTNConfig(**settings["ttn"])
    if stage: config = replace(config, stage=stage)
    return config, settings


def json_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def to_device(value, device):
    if isinstance(value, torch.Tensor): return value.to(device)
    if isinstance(value, dict): return {k: to_device(v, device) for k, v in value.items()}
    return value


def load_bundle(path, device):
    batch = torch.load(path, map_location="cpu", weights_only=False)
    batch = to_device(batch, device)
    if "y" not in batch or "camera_conditions" not in batch: raise ValueError("bundle requires y and camera_conditions")
    batch["camera_conditions"] = batch["camera_conditions"].float()
    return batch


def synthetic_bundle(config, device, frames=13, height=22, width=40):
    clean = torch.randn(1, 128, frames, height, width, device=device)
    pose = torch.eye(4, device=device).expand(1, frames, 4, 4).clone()
    pose[..., 0, 3] = torch.arange(frames, device=device) * .01
    image_width = width * 32
    image_height = height * 32
    intr = clean.new_tensor([image_width * .8, image_height * .8, image_width * .5,
                             image_height * .5]).expand(1, frames, 4)
    return {
        "clean_latents": clean,
        "initial_latent": clean[:, :, :1],
        "y": torch.randn(1, 1, 300, 2304, device=device),
        "uncondition": torch.zeros(1, 1, 300, 2304, device=device),
        "mask": torch.ones(1, 300, device=device, dtype=torch.long),
        "camera_conditions": torch.cat((pose.flatten(-2), intr), -1),
        "chunk_plucker": torch.zeros(1, 48, frames, height, width, device=device),
        "width": image_width,
        "height": image_height,
        "data_info": {}
    }


def build(args, stage=None, sana_path=None):
    from .sana import load_sana_config, build_sana
    ttn, settings = read_reference(args.config, stage or args.stage)
    source = sana_path or args.sana_config or settings["sana_config"]
    source = Path(source)
    if not source.is_absolute(): source = ROOT / source
    config = load_sana_config(source)
    model = build_sana(config, ttn, args.base_weights, device=args.device)
    return model, config, settings


def train_update(model, config, batch, optimizer, k):
    from train_video_scripts.train_sana_wm_stage1 import _build_timesteps, _build_time_sampler
    clean = batch["clean_latents"].float()
    sampler = _build_time_sampler(config, clean.shape[2], clean.device)
    timesteps, _ = _build_timesteps(config, clean, clean.shape[2], True, time_sampler=sampler)
    noise = torch.randn_like(clean)
    extras = {"chunk_plucker": batch["chunk_plucker"]} if "chunk_plucker" in batch else {}
    with torch.autocast(device_type=clean.device.type, dtype=torch.bfloat16, enabled=clean.device.type == "cuda"):
        return train_clip(model,
                          clean,
                          batch["y"],
                          batch["camera_conditions"],
                          optimizer,
                          SANAFlowLoss(config),
                          timesteps,
                          noise,
                          width=batch["width"],
                          height=batch["height"],
                          mask=batch.get("mask"),
                          data_info=batch.get("data_info"),
                          extras=extras,
                          valid_mask=batch.get("frame_valid_mask"),
                          tbptt=k)


@torch.no_grad()
def rollout(model, config, batch, steps=4, cfg_scale=4.5, cached_blocks=-1):
    from diffusion.scheduler.self_forcing_flow_euler_sampler import SelfForcingFlowEulerCamCtrl
    initial = batch.get("initial_latent")
    if initial is None: initial = batch["clean_latents"][:, :, :1]
    if initial.shape[0] != 1 or initial.shape[2] != 1:
        raise ValueError("reference rollout accepts batch 1, one initial latent frame")
    frames = batch["camera_conditions"].shape[1]
    if frames < 4 or frames % 3 != 1: raise ValueError("native first-plus-one rollout requires 1+3n latent frames")
    noise = torch.randn(initial.shape[0], initial.shape[1], frames, *initial.shape[-2:], device=initial.device)
    noise[:, :, :1] = initial.float()
    extras = {"chunk_plucker": batch["chunk_plucker"]} if "chunk_plucker" in batch else {}
    session = TTNSession(model, batch["camera_conditions"], batch["width"], batch["height"],
                         batch.get("frame_valid_mask"), extras)
    kwargs = {
        "mask": batch.get("mask"),
        "data_info": dict(batch.get("data_info", {})),
        "camera_conditions": batch["camera_conditions"],
        "ttn_session": session,
        **extras
    }
    kwargs["data_info"]["condition_frame_info"] = {0: 0.}
    # Restore forward_long after this episode so repeated rollouts cannot stack sampler patches.
    original = model.forward_long
    records = []
    last = time.perf_counter()
    try:
        solver = SelfForcingFlowEulerCamCtrl(model,
                                             batch["y"],
                                             batch.get("uncondition", torch.zeros_like(batch["y"])),
                                             cfg_scale=cfg_scale,
                                             flow_shift=config.scheduler.inference_flow_shift,
                                             model_kwargs=kwargs,
                                             base_chunk_frames=3,
                                             num_cached_blocks=cached_blocks)
        with torch.autocast(device_type=initial.device.type,
                            dtype=torch.bfloat16,
                            enabled=initial.device.type == "cuda"):
            for index, chunk, start, end in solver.sample_chunks(noise, steps=steps):
                if not torch.isfinite(chunk).all(): raise FloatingPointError("nonfinite generated chunk")
                if initial.device.type == "cuda": torch.cuda.synchronize()
                now = time.perf_counter()
                records.append({
                    "chunk": index,
                    "start": start,
                    "end": end,
                    "seconds": now - last,
                    **session.runtime.last_stats
                })
                last = now
    finally:
        model.forward_long = original
    expected = len(chunk_ranges(frames))
    if session.runtime.commit_count != expected + 1 or session.runtime.predict_count != expected + 1:
        raise AssertionError("prefill/chunk state counters disagree")
    if any(len(ids) != frames for ids in session.runtime.committed_frame_ids):
        raise AssertionError("duplicate/missing frame writes")
    if not torch.equal(noise[:, :, :1], initial.float()): raise AssertionError("initial frame was modified")
    return noise, session.runtime, records


def timed_cuda(call):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, {
        "seconds": time.perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved()
    }


def train_command(args):
    model, config, settings = build(args)
    optimizer = make_optimizer(model, settings.get("learning_rate", 1e-5))
    step = load_checkpoint(args.adapter, model, optimizer, args.resume) if args.adapter else 0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.batch_file:
        batch = load_bundle(args.batch_file, args.device)
        batches = (batch for _ in range(args.max_steps))
    else:
        from train_video_scripts.train_sana_wm_stage1 import _build_dataloader, _extract_batch, _encode_prompts
        from diffusion.model.builder import get_tokenizer_and_text_encoder
        config.train.train_batch_size = 1
        loader = _build_dataloader(config, config.text_encoder.model_max_length, 1, 0)
        tokenizer = encoder = None
        if not config.data.load_text_feat:
            tokenizer, encoder = get_tokenizer_and_text_encoder(config.text_encoder.text_encoder_name, args.device)
            encoder.eval().requires_grad_(False)

        def dataset_batches():
            while True:
                seen = False
                for raw in loader:
                    seen = True
                    clean, y, mask, info, camera, plucker = _extract_batch(
                        raw, args.device, torch.float32, config.data.load_text_feat,
                        getattr(config.data, "return_chunk_plucker", False))
                    if not config.data.load_text_feat:
                        with torch.no_grad():
                            y, mask = _encode_prompts(y, tokenizer, encoder, config, args.device)
                    if camera is None: raise ValueError("TTN training requires camera_conditions from dataset")
                    b = {
                        "clean_latents": clean,
                        "y": y,
                        "mask": mask,
                        "data_info": to_device(info, args.device),
                        "camera_conditions": camera,
                        "width": clean.shape[-1],
                        "height": clean.shape[-2]
                    }
                    if plucker is not None: b["chunk_plucker"] = plucker
                    yield b
                if not seen: raise ValueError("empty training dataset")

        batches = dataset_batches()
    k = args.tbptt or settings.get("tbptt", 2)
    for batch in batches:
        if step >= args.max_steps: break
        result, timing = timed_cuda(lambda: train_update(model, config, batch, optimizer, k))
        step += 1
        record = {
            "step": step,
            "stage": model.ttn_system.config.stage,
            "loss": result["loss"],
            "outer_grad_norm": result["outer_grad_norm"],
            "commits": result["runtime"].commit_count,
            "predictions": result["runtime"].predict_count,
            "chunks": result["chunks"],
            "tbptt": k,
            "config": model.ttn_system.config.to_dict(),
            "base": model.base_load_report,
            **timing
        }
        json_record(output / "train.jsonl", record)
        print(json.dumps(record))
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(output / "last.pt", model, optimizer, step)


def infer_command(args):
    if not args.batch_file:
        raise ValueError("infer requires --batch-file with one initial_latent and known camera trajectory")
    model, config, settings = build(args)
    if args.adapter: load_checkpoint(args.adapter, model)
    batch = load_bundle(args.batch_file, args.device)
    result, timing = timed_cuda(lambda: rollout(model, config, batch, args.steps, args.cfg_scale, args.cached_blocks))
    latents, runtime, records = result
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": latents.cpu()}, output / "rollout.pt")
    report = {
        "mode": "inference",
        "stage": model.ttn_system.config.stage,
        "config": model.ttn_system.config.to_dict(),
        "base": model.base_load_report,
        "adapter": args.adapter,
        "steps": args.steps,
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "commits": runtime.commit_count,
        "predictions": runtime.predict_count,
        "chunks": records,
        **timing
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


def smoke_command(args):
    if args.frames < 4 or args.frames % 3 != 1: raise ValueError("smoke uses 1+3n latent frames")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    previous = None
    all_reports = []
    for stage in args.stages:
        seed_everything(args.seed)
        model, config, settings = build(args, stage)
        optimizer = make_optimizer(model)
        if previous: load_checkpoint(previous, model, optimizer, resume=False)
        batch = load_bundle(args.batch_file, args.device) if args.batch_file else synthetic_bundle(
            config, args.device, args.frames, args.latent_height, args.latent_width)
        result, train_timing = timed_cuda(
            lambda: train_update(model, config, batch, optimizer, args.tbptt or settings.get("tbptt", 2)))
        gradients = {
            name: float(p.grad.float().norm())
            for name, p in model.named_parameters() if p.requires_grad and p.grad is not None
        }
        if not gradients or not all(torch.isfinite(torch.tensor(v)) for v in gradients.values()):
            raise AssertionError("invalid training gradients")
        previous = output / stage / "last.pt"
        save_checkpoint(previous, model, optimizer, 1)
        seed_everything(args.seed)
        sampled, sample_timing = timed_cuda(
            lambda: rollout(model, config, batch, args.steps, args.cfg_scale, args.cached_blocks))
        latent, runtime, records = sampled
        report = {
            "stage": stage,
            "synthetic_inputs": not bool(args.batch_file),
            "config": model.ttn_system.config.to_dict(),
            "base": model.base_load_report,
            "seed": args.seed,
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "anchors": [i for i, b in enumerate(model.blocks) if b.attn.__class__.__name__ == "TTNAnchor"],
            "train": {
                "loss": result["loss"],
                "outer_grad_norm": result["outer_grad_norm"],
                "grad_norms": gradients,
                "chunks": result["chunks"],
                **train_timing
            },
            "rollout": {
                "finite": bool(torch.isfinite(latent).all()),
                "commits": runtime.commit_count,
                "predictions": runtime.predict_count,
                "branch_state_norms": runtime.world_state.flatten(1).norm(dim=-1).tolist(),
                "chunks": records,
                **sample_timing
            }
        }
        if report["anchors"] != list(ANCHORS): raise AssertionError("anchor replacement mismatch")
        (output / stage / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        torch.save({"latents": latent.cpu()}, output / stage / "rollout.pt")
        all_reports.append(report)
        print(json.dumps(report))
        del model, optimizer, result, sampled, latent, runtime
        torch.cuda.empty_cache()
    (output / "summary.json").write_text(json.dumps(all_reports, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "infer", "smoke"))
    parser.add_argument("--config", default=str(ROOT / "configs/worldttn/reference.json"))
    parser.add_argument("--sana-config")
    parser.add_argument("--base-weights", help="local mirror of the specified SANA teacher, or hf:// URI")
    parser.add_argument("--adapter", help="TTN checkpoint; stage comes from --stage or reference config")
    parser.add_argument("--stage", choices=("A", "B", "C"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", default="output/worldttn")
    parser.add_argument("--batch-file", help="precomputed tensor bundle; see docs/ttn/RUNNING.md")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--tbptt", type=int, choices=(1, 2, 4))
    parser.add_argument("--steps", type=int, default=4, help="4 is smoke only; set an explicit quality schedule")
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument("--cached-blocks", type=int, default=-1)
    parser.add_argument("--stages", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"])
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--latent-height", type=int, default=22)
    parser.add_argument("--latent-width", type=int, default=40)
    args = parser.parse_args()
    os.environ.setdefault("DISABLE_XFORMERS", "1")
    if args.resume and not args.adapter: parser.error("--resume requires --adapter")
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        parser.error("server entrypoints require CUDA; local CPU validation is python -m pytest tests/ttn")
    if min(args.steps, args.max_steps, args.save_every) < 1: parser.error("step counts must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("SANA_CP_SIZE", "1")) > 1:
        parser.error("TTN v0.1 reference is single-card, CP=1, without FSDP")
    torch.cuda.set_device(
        torch.device(args.device) if torch.device(args.device).index is not None else torch.cuda.current_device())
    seed_everything(args.seed)
    {"train": train_command, "infer": infer_command, "smoke": smoke_command}[args.command](args)


if __name__ == "__main__": main()
