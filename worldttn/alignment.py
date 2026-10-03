"""One-chunk GT teacher-forcing diagnostic; no rollout, optimizer or distillation.

Same-input replay isolates each replaced attention module using SANA's hidden
inputs and geometry. Native TTN flow prediction additionally includes upstream
feature drift. Neither measurement establishes long-horizon memory quality.
"""
import gc
import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch

from .anchor import TTNAnchor
from .core import ANCHORS
from .evaluation import file_sha256, tensor_sha256, load_evaluation_run, load_evaluation_cases
from .session import TTNSession, validate_clean_output
from .training import SANAFlowLoss


def comparison_metrics(teacher, student, mask=None):
    if teacher.shape != student.shape:
        raise ValueError("alignment output shapes disagree")
    a, b = teacher.detach().float(), student.detach().float()
    if mask is not None:
        if mask.shape != a.shape[:-1]: raise ValueError("alignment mask must select tokens")
        a, b = a[mask.bool()], b[mask.bool()]
    if not a.numel() or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("alignment requires nonempty finite outputs")
    a, b = a.reshape(-1), b.reshape(-1)
    norm_a, norm_b = a.norm().item(), b.norm().item()
    mse = (a - b).square().mean().item()
    energy = a.square().mean().item()
    # Undefined ratios/cosines remain null rather than hiding a zero teacher.
    return {"mse": mse, "relative_l2": (mse / energy)**.5 if energy > 0 else None,
            "cosine": (a * b).sum().item() / (norm_a * norm_b) if norm_a * norm_b > 0 else None,
            "teacher_rms": energy**.5, "student_rms": b.square().mean().item()**.5,
            "rms_ratio": norm_b / norm_a if norm_a > 0 else None,
            "elements": a.numel()}


def attention_output(value):
    return value[0] if isinstance(value, tuple) else value


@contextmanager
def replay_anchors(teacher, student, context, records=None):
    """Observe the teacher without replacing its output or propagating TTN drift."""
    handles, seen = [], set()
    def hook(index):
        def replay(module, inputs, kwargs, output):
            if index in seen: raise RuntimeError("teacher anchor executed twice in one probe")
            seen.add(index)
            kwargs = dict(kwargs, ttn_chunk_context=context)
            if kwargs.get("kv_cache") is not None: kwargs["kv_cache"] = list(kwargs["kv_cache"])
            replacement = student.blocks[index].attn(*inputs, **kwargs)
            if records is not None:
                a, b = attention_output(output), attention_output(replacement)
                n = a.shape[1]
                read, _ = context.token_masks(n)
                generated = (context.frame_ids != 0).repeat_interleave(n // context.frame_ids.shape[1], -1)
                records.append({"block": index, "same_input": True,
                                "metric_tokens": int((read & generated).sum()),
                                **comparison_metrics(a, b, read & generated)})
            # Forward hooks returning None leave SANA's downstream features intact.
        return replay
    try:
        for i in ANCHORS:
            if not isinstance(student.blocks[i].attn, TTNAnchor) or isinstance(teacher.blocks[i].attn, TTNAnchor):
                raise ValueError("replay needs original SANA and five actual TTN anchors")
            handles.append(teacher.blocks[i].attn.register_forward_hook(hook(i), with_kwargs=True))
        yield
        if seen != set(ANCHORS): raise RuntimeError(f"not all five anchors executed: {sorted(seen)}")
    finally:
        for handle in handles: handle.remove()


@torch.no_grad()
def probe_chunk(teacher, student, batch, flow_loss, timesteps, noise):
    """Compare a fixed noised GT chunk with only frame 0 committed as history."""
    if teacher.training or student.training or any(p.requires_grad for m in (teacher, student) for p in m.parameters()):
        raise ValueError("alignment models must be eval/frozen")
    if student.ttn_system.config.stage != "A": raise ValueError("this isolation diagnostic is Stage A only")
    clean = batch["clean_latents"].float()
    if clean.ndim != 5 or clean.shape[0] != 1 or clean.shape[2] != 4 or noise.shape != clean.shape:
        raise ValueError("alignment needs one batch, one native first chunk (4 latents), and matching noise")
    if not timesteps or len(set(timesteps)) != len(timesteps): raise ValueError("use distinct probe timesteps")
    extras = {"chunk_plucker": batch["chunk_plucker"]} if "chunk_plucker" in batch else {}
    camera, y, mask, info = batch["camera_conditions"], batch["y"], batch.get("mask"), batch.get("data_info")
    if mask is not None: mask = mask.reshape(mask.shape[0], -1)  # preserve true padding
    width, height = clean.shape[-1], clean.shape[-2]
    native = TTNSession(student, camera, width, height, extras=extras)
    local = TTNSession(student, camera, width, height, extras=extras)
    teacher_session = TTNSession(teacher, camera, width, height, extras=extras)
    native.reset(1)
    local.reset(1)
    scratch = lambda model: [[None] * 10 for _ in model.blocks]
    with torch.autocast(device_type=clean.device.type, dtype=torch.bfloat16, enabled=clean.device.type == "cuda"):
        native.prefill(clean[:, :, :1], y, mask, info)
        prefill = local.begin_chunk(0, 1, prefill=True).for_clean()
        with replay_anchors(teacher, student, prefill):
            out, _ = teacher_session.forward(clean[:, :, :1], torch.zeros(1, device=clean.device), y,
                                             prefill, scratch(teacher), 0, 1, mask, info, save=True)
        validate_clean_output(out, clean[:, :, :1])
        local.runtime.prefill(prefill)
        local_context, native_context = local.begin_chunk(0, 4), native.begin_chunk(0, 4)
        snapshots = [s.runtime.world_state.clone() for s in (local, native)]
        loss_mask = torch.ones(1, 1, 4, 1, 1, device=clean.device, dtype=torch.bool)
        loss_mask[:, :, 0] = False
        rows = []
        for timestep in timesteps:
            t = torch.full((1, 1, 4), timestep, device=clean.device, dtype=torch.long)
            t[:, :, 0] = 0
            anchors, predictions = [], {}
            with replay_anchors(teacher, student, local_context, anchors):
                teacher_loss = flow_loss(teacher_session, clean, t, noise, y, local_context, scratch(teacher),
                                         0, 4, mask, info, loss_mask,
                                         on_prediction=lambda x: predictions.update(teacher=x.detach()))
            student_loss = flow_loss(native, clean, t, noise, y, native_context, scratch(student),
                                     0, 4, mask, info, loss_mask,
                                     on_prediction=lambda x: predictions.update(student=x.detach()))
            for output in predictions.values(): validate_clean_output(output, clean)
            if not torch.isfinite(teacher_loss).all() or not torch.isfinite(student_loss).all():
                raise FloatingPointError("nonfinite diagnostic flow loss")
            rows.append({"timestep": timestep, "anchors": anchors,
                         "native_flow": {"sana_loss": teacher_loss.mean().item(),
                                         "ttn_loss": student_loss.mean().item(),
                                         "ttn_minus_sana_loss": (student_loss.mean() - teacher_loss.mean()).item(),
                                         **comparison_metrics(predictions["teacher"][:, :, 1:], predictions["student"][:, :, 1:])}})
    states = {}
    for label, session, context, snapshot in zip(("teacher_feature_prefill", "native_prefill"),
                                                (local, native), (local_context, native_context), snapshots):
        r = session.runtime
        if r.commit_count != 1 or r.predict_count != 2 or r.committed_frame_ids != [{0}]:
            raise AssertionError("probe changed history/counters")
        if not torch.equal(r.world_state, snapshot) or r.transition_fast.count_nonzero():
            raise AssertionError("temporary probe changed persistent S/psi")
        states[label] = {"commits": r.commit_count, "predictions": r.predict_count,
                         "committed_frames": [0], "write_mask": context.write_mask[0].tolist(),
                         "state_norm": r.world_state.norm().item()}
    return {"probes": rows, "states": states, "noise_sha256": tensor_sha256(noise)}


def alignment_command(args):
    from .cli import seed_everything, to_device, timed_cuda, json_record
    from .sana import build_sana, configure_cross_attention
    from .checkpoint import load_checkpoint
    from diffusion.model.builder import get_tokenizer_and_text_encoder
    from train_video_scripts.train_sana_wm_stage1 import _encode_prompts
    from torch.utils.data import default_collate

    output = Path(args.output).resolve()
    if output.exists(): raise ValueError(f"use a new alignment output directory: {output}")
    if args.frames != 4: raise ValueError("align-chunk requires exactly 4 latent frames")
    run, config, ttn, adapter, digest, last_train = load_evaluation_run(args)
    if ttn.stage != "A": raise ValueError("align-chunk currently diagnoses Stage A only")
    if any(t < 0 or t >= config.scheduler.train_sampling_steps for t in args.alignment_timesteps):
        raise ValueError("probe timesteps must lie inside the training schedule")
    cases, rejected, _ = load_evaluation_cases(config, args)
    tokenizer, encoder = get_tokenizer_and_text_encoder(config.text_encoder.text_encoder_name, "cpu")
    encoder.eval().requires_grad_(False)
    with torch.no_grad():
        for case in cases:
            case["y"], case["mask"] = _encode_prompts([case["prompt"]], tokenizer, encoder, config, "cpu")
            case["y"], case["mask"] = case["y"].cpu(), case["mask"].cpu()
    del tokenizer, encoder
    gc.collect()
    output.mkdir(parents=True)
    protocol = {"scope": "single-chunk training-example isolation diagnostic, not a rollout",
                "training_run": str(Path(args.training_run).resolve()), "checkpoint": str(adapter),
                "checkpoint_sha256": digest, "base_sha256": run["base"]["sha256"],
                "stage": ttn.stage, "step": last_train["step"], "frames": 4,
                "timesteps": args.alignment_timesteps, "seed": args.seed, "cfg": False,
                "local_replay": "same SANA hidden inputs/geometry; S prefilled once from teacher features",
                "native_flow": "normal TTN features and native GT-frame-0 prefill; SANAFlowLoss/noise-clean target",
                "history": "only GT frame 0 committed; current input is noised GT; no generated history",
                "metric_scope": "attention output after camera fusion/shared gate/proj, before residual; exclude frame 0",
                "training_changed": False, "threshold": None,
                "torch": torch.__version__, "cuda": torch.version.cuda, "launch": args.launch,
                "config": {name: asdict(getattr(config, name)) for name in ("model", "scheduler", "text_encoder", "data")}}
    (output / "manifest.json").write_text(json.dumps({"protocol": protocol, "rejected": rejected}, indent=2))
    teacher = build_sana(config, ttn, args.base_weights or run["base"]["source"], args.device, install_adapter=False)
    student = build_sana(config, ttn, args.base_weights or run["base"]["source"], args.device)
    for label, model in (("sana", teacher), ("ttn", student)):
        if model.base_load_report["sha256"] != run["base"]["sha256"]: raise ValueError("base weights differ from training")
        if label == "ttn": load_checkpoint(adapter, model)
        policy = configure_cross_attention(model, args.cross_attn_backend)
        model.eval().requires_grad_(False)
        print("[TTN alignment model] " + json.dumps({"method": label, "cross_attention": policy}), flush=True)
    if file_sha256(adapter) != digest: raise ValueError("checkpoint changed during alignment")
    loss_fn, records = SANAFlowLoss(config), []
    for index, case in enumerate(cases):
        seed_everything(case["seed"])
        batch = {"clean_latents": case["reference"].to(args.device), "camera_conditions": case["camera"].to(args.device),
                 "y": case["y"].to(args.device), "mask": case["mask"].to(args.device),
                 "data_info": to_device(default_collate([case["info"]]), args.device)}
        if case["plucker"] is not None: batch["chunk_plucker"] = case["plucker"][None].to(args.device)
        generator = torch.Generator(device=args.device).manual_seed(case["seed"])
        noise = torch.randn(batch["clean_latents"].shape, generator=generator, device=args.device)
        result, timing = timed_cuda(lambda: probe_chunk(teacher, student, batch, loss_fn, args.alignment_timesteps, noise))
        row = {"case_id": case["case_id"], "seed": case["seed"], "prompt": case["prompt"],
               "input_sha256": {k: tensor_sha256(v) for k, v in batch.items() if isinstance(v, torch.Tensor)},
               "timing": timing, **result}
        records.append(row)
        json_record(output / "episodes.jsonl", row)
        print("[TTN alignment case] " + json.dumps(row), flush=True)
    (output / "summary.json").write_text(json.dumps({"protocol": protocol, "episodes": records}, indent=2))
    print("[TTN alignment done] " + json.dumps({"output": str(output), "step": last_train["step"]}), flush=True)
