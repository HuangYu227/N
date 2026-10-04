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
from .evaluation import file_sha256, tensor_sha256, load_evaluation_run, load_evaluation_cases, diagnostic_noise
from .session import TTNSession, validate_clean_output
from .training import SANAFlowLoss
from .alignment_detail import (comparison_metrics, stage_metrics, block_drift_trace, parameter_fingerprint,
                               parameter_changes, gradient_health, diagnostic_summary, print_concise_report)


def attention_output(value):
    return value[0] if isinstance(value, tuple) else value


@contextmanager
def matched_softmax_teacher(teacher, student):
    """Private diagnostic: same learned mappings/backbone, original Softmax kernels.

    This hybrid teacher is never used as the original-SANA rollout baseline.
    Restore original parameter values AND dtypes after the diagnostic.
    """
    source, target = student.state_dict(), teacher.state_dict()
    shared = {k: v for k, v in source.items() if not k.startswith("ttn_system.") and ".beta_proj." not in k}
    if any(k not in target or target[k].shape != v.shape for k, v in shared.items()):
        raise ValueError("matched teacher requires the same SANA backbone and inherited mappings")
    backup = {k: target[k].detach().cpu().clone() for k in shared}
    try:
        # Match master precision as well as values; do not round FP32 TTN projections to BF16.
        for name, parameter in teacher.named_parameters():
            if name in shared: parameter.data = shared[name].detach().to(parameter.device, copy=True)
        for name, buffer in teacher.named_buffers():
            if name in shared: buffer.copy_(shared[name])
        yield {"shared_tensors": len(shared), "reference": "student mappings/backbone with original Softmax; diagnostic hybrid"}
    finally:
        for name, parameter in teacher.named_parameters():
            if name in backup: parameter.data = backup[name].to(parameter.device)
        for name, buffer in teacher.named_buffers():
            if name in backup: buffer.copy_(backup[name])


@contextmanager
def replay_anchors(teacher, student, context, records=None):
    """Observe the teacher without replacing its output or propagating TTN drift."""
    handles, seen = [], set()
    teacher_stages, student_stages = {i: {} for i in ANCHORS}, {i: {} for i in ANCHORS}
    def emit(values):
        def save(name, value):
            values[name] = tuple(t.detach() for t in value) if isinstance(value, tuple) else value.detach()
        return save
    def teacher_pre(index):
        def prepare(module, inputs, kwargs):
            teacher_stages[index]["input"] = inputs[0].detach()
            return inputs, dict(kwargs, ttn_diagnostic=emit(teacher_stages[index]))
        return prepare
    def qkv_hook(values):
        def save(module, inputs, output): emit(values)("projected_qkv", output.chunk(3, -1))
        return save
    def hook(index):
        def replay(module, inputs, kwargs, output):
            if index in seen: raise RuntimeError("teacher anchor executed twice in one probe")
            seen.add(index)
            kwargs = dict(kwargs, ttn_chunk_context=context)
            if records is not None: kwargs["ttn_diagnostic"] = emit(student_stages[index])
            if kwargs.get("kv_cache") is not None: kwargs["kv_cache"] = list(kwargs["kv_cache"])
            replacement = student.blocks[index].attn(*inputs, **kwargs)
            if records is not None:
                a, b = attention_output(output), attention_output(replacement)
                n = a.shape[1]
                read, _ = context.token_masks(n)
                generated = (context.frame_ids != 0).repeat_interleave(n // context.frame_ids.shape[1], -1)
                stages = stage_metrics(teacher_stages[index], student_stages[index], context, student.ttn_system.config)
                records.append({"block": index, "same_input": stages["input"]["exact_equal"], "stages": stages,
                                "metric_tokens": int((read & generated).sum()),
                                **comparison_metrics(a, b, read & generated)})
                teacher_stages[index].clear()
                student_stages[index].clear()
            # Forward hooks returning None leave SANA's downstream features intact.
        return replay
    try:
        for i in ANCHORS:
            if not isinstance(student.blocks[i].attn, TTNAnchor) or isinstance(teacher.blocks[i].attn, TTNAnchor):
                raise ValueError("replay needs original SANA and five actual TTN anchors")
            if records is not None:
                handles.append(teacher.blocks[i].attn.register_forward_pre_hook(teacher_pre(i), with_kwargs=True))
                handles.append(teacher.blocks[i].attn.qkv.register_forward_hook(qkv_hook(teacher_stages[i])))
                handles.append(student.blocks[i].attn.qkv.register_forward_hook(qkv_hook(student_stages[i])))
            handles.append(teacher.blocks[i].attn.register_forward_hook(hook(i), with_kwargs=True))
        yield
        if seen != set(ANCHORS): raise RuntimeError(f"not all five anchors executed: {sorted(seen)}")
    finally:
        for handle in handles: handle.remove()


@torch.no_grad()
def probe_chunk(teacher, student, batch, flow_loss, timesteps, noise, *, gradient_timestep=None):
    """Compare a fixed noised GT chunk with only frame 0 committed as history."""
    if teacher.training or student.training or any(p.requires_grad for m in (teacher, student) for p in m.parameters()):
        raise ValueError("alignment models must be eval/frozen")
    clean = batch["clean_latents"].float()
    if clean.ndim != 5 or clean.shape[0] != 1 or clean.shape[2] != 4 or noise.shape != clean.shape:
        raise ValueError("alignment needs one batch, one native first chunk (4 latents), and matching noise")
    if not timesteps or len(set(timesteps)) != len(timesteps): raise ValueError("use distinct probe timesteps")
    parameter_hashes = [parameter_fingerprint(m) for m in (teacher, student)]
    flags = [[p.requires_grad for p in m.parameters()] for m in (teacher, student)]
    changes = parameter_changes(teacher, student)
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
        psi_snapshots = [s.runtime.transition_fast.clone() for s in (local, native)]
        pose_snapshots = [s.runtime.previous_committed_pose.clone() for s in (local, native)]
        loss_mask = torch.ones(1, 1, 4, 1, 1, device=clean.device, dtype=torch.bool)
        loss_mask[:, :, 0] = False
        rows = []
        for timestep in timesteps:
            t = torch.full((1, 1, 4), timestep, device=clean.device, dtype=torch.long)
            t[:, :, 0] = 0
            anchors, predictions = [], {}
            with block_drift_trace(teacher, student, local_context) as drift:
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
            rows.append({"timestep": timestep, "anchors": anchors, "representation_drift": drift,
                         "native_flow": {"sana_loss": teacher_loss.mean().item(),
                                         "ttn_loss": student_loss.mean().item(),
                                         "ttn_minus_sana_loss": (student_loss.mean() - teacher_loss.mean()).item(),
                                         **comparison_metrics(predictions["teacher"][:, :, 1:], predictions["student"][:, :, 1:])}})
        health = None if gradient_timestep is None else gradient_health(
            native, native_context, clean, y, mask, info, noise, flow_loss, loss_mask, gradient_timestep)
    states = {}
    for label, session, context, snapshot, psi, pose in zip(("teacher_feature_prefill", "native_prefill"),
              (local, native), (local_context, native_context), snapshots, psi_snapshots, pose_snapshots):
        r = session.runtime
        if r.commit_count != 1 or r.predict_count != 2 or r.committed_frame_ids != [{0}]:
            raise AssertionError("probe changed history/counters")
        if not torch.equal(r.world_state, snapshot) or not torch.equal(r.transition_fast, psi) or not torch.equal(r.previous_committed_pose, pose):
            raise AssertionError("temporary probe changed persistent S/psi")
        states[label] = {"commits": r.commit_count, "predictions": r.predict_count,
                         "committed_frames": [0], "write_mask": context.write_mask[0].tolist(),
                         "state_norm": r.world_state.norm().item()}
    if parameter_hashes != [parameter_fingerprint(m) for m in (teacher, student)]:
        raise AssertionError("diagnostic changed model parameters")
    if flags != [[p.requires_grad for p in m.parameters()] for m in (teacher, student)]:
        raise AssertionError("gradient probe did not restore requires_grad flags")
    return {"probes": rows, "states": states, "noise_sha256": tensor_sha256(noise),
            "parameter_changes": changes, "gradient_health": health, "diagnostic_summary": diagnostic_summary(rows),
            "read_only_verified": {"parameter_sha256_before": parameter_hashes, "parameters_unchanged": True,
                                   "requires_grad_restored": True, "persistent_S_psi_pose_unchanged_after_private_prefill": True}}


def alignment_command(args):
    from .cli import seed_everything, to_device, timed_cuda, json_record
    from .sana import build_sana, configure_cross_attention
    from .checkpoint import load_checkpoint
    from .provenance import implementation_identity
    from diffusion.model.builder import get_tokenizer_and_text_encoder
    from train_video_scripts.train_sana_wm_stage1 import _encode_prompts
    from torch.utils.data import default_collate

    output = Path(args.output).resolve()
    if output.exists(): raise ValueError(f"use a new alignment output directory: {output}")
    if args.frames != 4: raise ValueError("align-chunk requires exactly 4 latent frames")
    run, config, ttn, adapter, digest, last_train = load_evaluation_run(args)
    gradient_timestep = getattr(args, "alignment_grad_timestep", 500)
    if any(t < 0 or t >= config.scheduler.train_sampling_steps for t in [*args.alignment_timesteps, gradient_timestep]):
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
                "provenance": implementation_identity(),
                "training_provenance": run.get("provenance"),
                "train_scope": last_train.get("train_scope", run["arguments"].get("train_scope", "ttn")),
                "camera_attention": ttn.camera_attention,
                "teacher_modes": ["original_sana", "matched_backbone_softmax"],
                "fixed_cases_sha256": file_sha256(args.fixed_cases) if getattr(args, "fixed_cases", None) else None,
                "noise_frames": getattr(args, "noise_frames", None) or args.frames,
                "training_run": str(Path(args.training_run).resolve()), "checkpoint": str(adapter),
                "checkpoint_sha256": digest, "base_sha256": run["base"]["sha256"],
                "ttn_config": ttn.to_dict(), "base_initialization_seed": args.seed,
                "stage": ttn.stage, "step": last_train["step"], "frames": 4,
                "timesteps": args.alignment_timesteps, "seed": args.seed, "cfg": False,
                "local_replay": "same SANA hidden inputs/geometry; S prefilled once from teacher features",
                "native_flow": "normal TTN features and native GT-frame-0 prefill; SANAFlowLoss/noise-clean target",
                "history": "only GT frame 0 committed; current input is noised GT; no generated history",
                "metric_scope": "attention output after camera fusion/shared gate/proj, before residual; exclude frame 0",
                "training_changed": False, "threshold": None,
                "gradient_probe": {"timestep": gradient_timestep, "scope": "fixed single-chunk flow loss; no optimizer step, raw gradients not historical logs",
                                   "activation_offload": "cpu"},
                "parameter_reference": "base SANA projections cast to FP32 and TTN beta zero initialization; no invented initial checkpoint",
                "timing_scope": "instrumented probes/CPU metrics/hashing/backward; not a throughput benchmark",
                "torch": torch.__version__, "cuda": torch.version.cuda, "launch": args.launch,
                "config": {name: asdict(getattr(config, name)) for name in ("model", "scheduler", "text_encoder", "data")}}
    (output / "manifest.json").write_text(json.dumps({"protocol": protocol, "rejected": rejected}, indent=2))
    seed_everything(args.seed)
    teacher = build_sana(config, ttn, args.base_weights or run["base"]["source"], args.device, install_adapter=False)
    seed_everything(args.seed)  # also match any allowed missing base buffer/embedding initialization
    kwargs = {"dtype": torch.float32} if (last_train.get("weight_scope") == "dit" or
                run["arguments"].get("train_scope") in ("dit", "ttn-visual")) else {}
    student = build_sana(config, ttn, args.base_weights or run["base"]["source"], args.device, **kwargs)
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
        noise = diagnostic_noise(batch["clean_latents"].shape, args.device, case["seed"], getattr(args, "noise_frames", None))
        result, timing = timed_cuda(lambda: probe_chunk(teacher, student, batch, loss_fn, args.alignment_timesteps, noise,
                                                        gradient_timestep=gradient_timestep))
        row = {"case_id": case["case_id"], "seed": case["seed"], "prompt": case["prompt"],
               "input_sha256": {k: tensor_sha256(v) for k, v in batch.items() if isinstance(v, torch.Tensor)},
               "timing": timing, **result}
        records.append(row)
        with matched_softmax_teacher(teacher, student) as matched:
            isolated, isolated_timing = timed_cuda(lambda: probe_chunk(
                teacher, student, batch, loss_fn, args.alignment_timesteps, noise))
        row["matched_backbone_softmax"] = {**matched, "timing": isolated_timing, **isolated,
                                          "parameter_reference": "matched mappings, not initialization; beta remains zero-reference"}
        json_record(output / "episodes.jsonl", row)
        print("[TTN alignment case] " + json.dumps({"case_id": case["case_id"], "seed": case["seed"], "timing": timing,
                                                  "read_only_verified": result["read_only_verified"]}), flush=True)
        print_concise_report(result)
    (output / "summary.json").write_text(json.dumps({"protocol": protocol, "episodes": records}, indent=2))
    print("[TTN alignment done] " + json.dumps({"output": str(output), "step": last_train["step"]}), flush=True)
