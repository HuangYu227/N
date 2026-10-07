"""Single-card inference and causal DDP/FSDP2 training; --help needs no CUDA SANA imports."""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from dataclasses import replace, asdict
import torch
from .core import TTNConfig, BASE_ID, ANCHORS
from .checkpoint import make_optimizer, save_checkpoint, load_checkpoint, read_checkpoint, apply_checkpoint_weights
from .training import train_clip, SANAFlowLoss, chunk_ranges
from .session import TTNSession
from .performance import ExecutionOptions, configure_from_args, execution_report, precision_audit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "configs/worldttn/reference_sana_camera.json"


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
    from .sana import load_sana_config, build_sana, configure_cross_attention
    ttn, settings = read_reference(args.config, stage or args.stage)
    source = sana_path or args.sana_config or settings["sana_config"]
    source = Path(source)
    if not source.is_absolute(): source = ROOT / source
    config = load_sana_config(source)
    root = getattr(args, "dataset_root", None)
    if root:
        from .sana import resolve_data_paths
        resolve_data_paths(config, root)
    if getattr(args, "data_dir", None):
        names = list(config.data.data_dir) if isinstance(config.data.data_dir, dict) else ["sekai_game"]
        if len(names) != 1: raise ValueError("--data-dir requires a single named dataset in the SANA config")
        config.data.data_dir = {names[0]: str(Path(args.data_dir).resolve())}
    if getattr(args, "vae_cache_dir", None): config.data.vae_cache_dir = str(Path(args.vae_cache_dir).resolve())
    # Do not quantize inherited master weights to BF16 before joint fine-tuning.
    options = {"dtype": torch.float32} if getattr(args, "train_scope", "ttn") in ("ttn-visual", "ttn-new", "dit") else {}
    model = build_sana(config, ttn, args.base_weights, device=args.device, **options)
    configure_from_args(model, args)
    policy = configure_cross_attention(model, getattr(args, "cross_attn_backend", "auto"),
                                       diagnostic_unmask_all_valid=getattr(args, "diagnostic_unmask_all_valid", False))
    from .distributed import rank_world
    if rank_world()[0] == 0: print("[TTN SDPA] " + json.dumps(policy), flush=True)
    return model, config, settings


def train_update(model, config, batch, optimizer, k, parallel=None, *, activation_offload="none", memory_trace=False):
    from .cuda_debug import cuda_diagnostics
    from train_video_scripts.train_sana_wm_stage1 import _build_timesteps, _build_time_sampler
    clean = batch["clean_latents"].float()
    sampler = _build_time_sampler(config, clean.shape[2], clean.device)
    timesteps, _ = _build_timesteps(config, clean, clean.shape[2], True, time_sampler=sampler)
    noise = torch.randn_like(clean)
    extras = {"chunk_plucker": batch["chunk_plucker"]} if "chunk_plucker" in batch else {}
    if parallel is not None:
        activation_offload, memory_trace = parallel.activation_offload, parallel.memory_trace
    memory_records = []
    def trace(phase, **info):
        from .distributed import rank_world
        record = {"phase": phase, "rank": rank_world()[0], "activation_offload": activation_offload, **info}
        if parallel is not None and phase in ("prefill_begin", "backward_end", "optimizer_end", "oom"):
            record["storage"] = parallel.storage_record(optimizer)
        if clean.device.type == "cuda":
            record.update(allocated_bytes=torch.cuda.memory_allocated(clean.device),
                          reserved_bytes=torch.cuda.memory_reserved(clean.device),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(clean.device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(clean.device))
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    record["host_rss_bytes"] = int(line.split()[1]) * 1024
                    break
        except OSError: pass  # Optional Linux diagnostic; also runs in local Windows tests.
        memory_records.append(record)
        # Explicit per-rank diagnostics; ordinary training metrics remain rank 0.
        print("[TTN memory] " + json.dumps(record), flush=True)
    try:
        with cuda_diagnostics(os.environ.get("TTN_CUDA_TRACE_DIR"), clean.device), torch.autocast(
                device_type=clean.device.type, dtype=torch.bfloat16, enabled=clean.device.type == "cuda"):
            result = train_clip(model, clean, batch["y"], batch["camera_conditions"], optimizer,
                                parallel.loss_fn if parallel else SANAFlowLoss(config), timesteps, noise,
                                width=batch["width"], height=batch["height"], mask=batch.get("mask"),
                                data_info=batch.get("data_info"), extras=extras,
                                valid_mask=batch.get("frame_valid_mask"), tbptt=k, parallel=parallel,
                                activation_offload=activation_offload,
                                memory_callback=trace if memory_trace else None)
    except torch.OutOfMemoryError:
        trace("oom", failed_after=memory_records[-1]["phase"] if memory_records else "unknown")
        raise
    result.update(activation_offload=activation_offload, memory_phases=memory_records)
    return result


@torch.no_grad()
def rollout(model, config, batch, steps=4, cfg_scale=4.5, cached_blocks=-1, *, initial_noise=None, on_chunk=None,
            ttn_ablation="full", state_diagnostics=False, history_reference=None, on_state=None,
            history_source=None, sink_options=None, replay_options=None):
    from diffusion.scheduler.self_forcing_flow_euler_sampler import SelfForcingFlowEulerCamCtrl
    initial = batch.get("initial_latent")
    if initial is None: initial = batch["clean_latents"][:, :, :1]
    if initial.shape[0] != 1 or initial.shape[2] != 1:
        raise ValueError("reference rollout accepts batch 1, one initial latent frame")
    frames = batch["camera_conditions"].shape[1]
    if frames < 4 or frames % 3 != 1: raise ValueError("native first-plus-one rollout requires 1+3n latent frames")
    shape = (initial.shape[0], initial.shape[1], frames, *initial.shape[-2:])
    history_source = history_source or ("gt" if history_reference is not None else "generated")
    if history_source not in ("generated", "gt", "ttn-gt", "native-gt"):
        raise ValueError("unknown clean history source")
    if (history_source != "generated") != (history_reference is not None):
        raise ValueError("GT/mixed history requires an explicit reference; generated history cannot access it")
    if history_source in ("ttn-gt", "native-gt") and not hasattr(model, "ttn_system"):
        raise ValueError("mixed history requires a TTN model")
    if sink_options is not None and sink_options.active:
        if not hasattr(model, "ttn_system"):
            raise ValueError("TLA sink requires a TTN model")
        if history_source != "generated":
            raise ValueError("TLA sink requires generated history")
    if history_reference is not None:
        if history_reference.device.type != "cpu" or tuple(history_reference.shape) != shape:
            raise ValueError("GT-history reference must stay on CPU with the exact rollout shape")
        if not torch.isfinite(history_reference).all() or not torch.equal(history_reference[:, :, :1], initial.cpu()):
            raise ValueError("GT-history reference is nonfinite or has a different observed frame")
    if replay_options is not None and replay_options.active:
        if not hasattr(model, "ttn_system") or history_source != "generated":
            raise ValueError("observed replay requires a TTN model with generated history")
    if initial_noise is not None:
        if tuple(initial_noise.shape) != shape or initial_noise.device != initial.device:
            raise ValueError("initial noise shape/device must match the rollout")
        noise = initial_noise.clone()  # methods must never share an in-place-mutated noise tensor
    else:
        noise = torch.randn(*shape, device=initial.device)
    noise[:, :, :1] = initial.float()
    extras = {"chunk_plucker": batch["chunk_plucker"]} if "chunk_plucker" in batch else {}
    session = TTNSession(model, batch["camera_conditions"], batch["width"], batch["height"],
                         batch.get("frame_valid_mask"), extras, ablation=ttn_ablation,
                         diagnostics=state_diagnostics,
                         **({"sink_options": sink_options} if sink_options is not None else {}),
                         **({"replay_options": replay_options} if replay_options is not None else {})) if hasattr(model, "ttn_system") else None
    from .session import repeat_batch
    cfg_batch = initial.shape[0] * (2 if cfg_scale > 1 else 1)
    camera = repeat_batch(batch["camera_conditions"], cfg_batch).clone()
    camera[..., 16:] *= camera.new_tensor([initial.shape[-1] / batch["width"],
                                          initial.shape[-2] / batch["height"]] * 2)
    mask = batch.get("mask")
    if isinstance(mask, torch.Tensor):
        if mask.ndim not in (2, 4) or (mask.ndim == 4 and mask.shape[1:3] != (1, 1)):
            raise ValueError("rollout text mask must be [B,L] or [B,1,1,L]")
        # SANA forward_long repeats a 2D token mask for CFG BEFORE squeezing;
        # _encode_prompts returns [B,1,1,L]. Preserve all padding bits.
        mask = mask.reshape(mask.shape[0], -1)
    kwargs = {
        "mask": mask,
        "data_info": dict(batch.get("data_info", {})),
        "camera_conditions": camera,
        **{key: repeat_batch(value, cfg_batch) for key, value in extras.items()}
    }
    if session is not None: kwargs["ttn_session"] = session
    kwargs["data_info"]["condition_frame_info"] = {0: 0.}
    # Restore forward_long after this episode so repeated rollouts cannot stack sampler patches.
    original = model.forward_long
    records = []
    last = time.perf_counter()
    try:
        history_kwargs = {} if history_reference is None else {
            "clean_history_provider": lambda start, end: history_reference[:, :, start:end].to(initial.device),
            "clean_history_source": history_source}
        solver = SelfForcingFlowEulerCamCtrl(model,
                                             batch["y"],
                                             batch.get("uncondition", torch.zeros_like(batch["y"])),
                                             cfg_scale=cfg_scale,
                                             flow_shift=config.scheduler.inference_flow_shift,
                                             model_kwargs=kwargs,
                                             base_chunk_frames=3,
                                             num_cached_blocks=cached_blocks, **history_kwargs)
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
                    **(session.runtime.last_stats if session is not None else {})
                })
                if history_reference is not None:
                    records[-1]["clean_history_source"] = history_source
                if on_chunk is not None: on_chunk(records[-1])
                if on_state is not None and session is not None: on_state(index, session.runtime)
                last = now
    finally:
        model.forward_long = original
    if session is not None:
        expected = len(chunk_ranges(frames))
        if session.runtime.commit_count != expected + 1 or session.runtime.predict_count != expected + 1:
            raise AssertionError("prefill/chunk state counters disagree")
        if any(len(ids) != frames for ids in session.runtime.committed_frame_ids):
            raise AssertionError("duplicate/missing frame writes")
        if session.runtime.sink_options.active and session.runtime.sink_reference is not None:
            session.runtime.verify_sink_reference()
            session.runtime.sink_reference_verified = True
        if session.runtime.replay_observations:
            session.runtime.verify_replay_reference()
            session.runtime.replay_reference_verified = True
    if not torch.equal(noise[:, :, :1], initial.float()): raise AssertionError("initial frame was modified")
    return noise, session.runtime if session is not None else None, records


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


def _dataset_batch(raw, config, args, tokenizer=None, encoder=None, encoder_device=None):
    from train_video_scripts.train_sana_wm_stage1 import _extract_batch, _encode_prompts
    clean, y, mask, info, camera, plucker = _extract_batch(
        raw, args.device, torch.float32, config.data.load_text_feat,
        getattr(config.data, "return_chunk_plucker", False))
    if not config.data.load_text_feat:
        with torch.no_grad():
            y, mask = _encode_prompts(y, tokenizer, encoder, config, encoder_device)
        y, mask = y.to(args.device), mask.to(args.device)
    if camera is None: raise ValueError("TTN training requires camera_conditions from dataset")
    batch = {"clean_latents": clean, "y": y, "mask": mask,
             "data_info": to_device(info, args.device), "camera_conditions": camera.float(),
             "width": clean.shape[-1], "height": clean.shape[-2]}
    if plucker is not None: batch["chunk_plucker"] = plucker
    return batch


def _training_identity(args, config, settings, k):
    def section(name):
        value = getattr(config, name, None)
        return asdict(value) if value is not None else None
    train = getattr(config, "train", None)
    identity = {"tbptt": k, "seed": args.seed, "per_rank_batch": 1,
            "learning_rate": settings.get("learning_rate", 1e-5),
            "scheduler": asdict(config.scheduler),
            "task": getattr(config, "task", None), "sana_model": section("model"),
            "text_encoder": section("text_encoder"),
            "text_encoder_device": getattr(args, "text_encoder_device", "auto"),
            "timestep": {name: getattr(train, name, None) for name in (
                "chunk_sampling_strategy", "same_timestep_prob", "chunk_mixture_probs", "noise_multiplier")},
            "batch_file": args.batch_file,
            "execution": {"core_backend": getattr(args, "ttn_core_backend", "reference"),
                          "psi_backend": getattr(args, "ttn_psi_backend", "reference")},
            "data": None if args.batch_file else asdict(config.data)}
    # Preserve existing frozen-backbone resume identities exactly.
    if getattr(args, "train_scope", "ttn") == "dit":
        identity.update(train_scope="dit", backbone_lr=getattr(args, "backbone_lr", 1e-6), optimizer_foreach=False)
    elif getattr(args, "train_scope", "ttn") in ("ttn-visual", "ttn-new"):
        identity.update(train_scope=args.train_scope, optimizer_foreach=False)
    if getattr(args, "optimizer_policy", None) == "origin": identity["optimizer_policy"] = "origin"
    ttn = settings.get("ttn", {})
    if ttn.get("local_update", False) or ttn.get("persistent_meta", False):
        identity["meta_ttt"] = {"version": "dual-psi-v1", "local_update": ttn.get("local_update", False),
                                "persistent_meta": ttn.get("persistent_meta", False),
                                "local_objective": "weighted_v_energy_support", "persistent_objective": "raw_weighted_innovation",
                                "support": "absolute_frame_xy_checkerboard", "local_eta_init": .01, "local_delta": 1.,
                                "noise_gate": False}
    return identity


def _training_record(model, result, timing, parallel):
    local = {"rank": parallel.rank, "loss": result["loss"], "outer_grad_norm": result["outer_grad_norm"],
             "commits": result["runtime"].commit_count, "predictions": result["runtime"].predict_count,
             "chunks": result["chunks"], "activation_offload": result.get("activation_offload", "none"),
             "prefill": result.get("prefill"),
             "memory_phases": result.get("memory_phases", []), **timing}
    local["storage"] = result.get("storage")
    if "anchor_gradients" in result: local["anchor_gradients"] = result["anchor_gradients"]
    from .distributed import gather_records
    if "parameter_update" in result: local["parameter_update"] = result["parameter_update"]
    for key in ("optimizer_updates", "exposure"):
        if key in result: local[key] = result[key]
    ranks = gather_records(local)
    return {"loss": sum(record["loss"] for record in ranks) / parallel.world,
            "outer_grad_norm": result["outer_grad_norm"], "ranks": ranks,
            "seconds": max(record["seconds"] for record in ranks),
            "world_size": parallel.world, "global_batch": parallel.world, "parallel": parallel.mode,
            "train_scope": getattr(model, "ttn_train_scope", "ttn"),
            "weight_scope": getattr(model, "ttn_weight_scope", "ttn"),
            "config": model.ttn_system.config.to_dict(), "base": model.base_load_report,
            "cross_attention": getattr(model, "cross_attention_report", None), "execution": execution_report(model)}


def benchmark_run(args):
    return bool(getattr(args, "ttn_benchmark_stable_steps", 0) or getattr(args, "ttn_profiler_trace", None)
                or getattr(args, "ttn_layout_audit", False))


def checkpoint_due(args, step):
    enabled = not benchmark_run(args) or getattr(args, "benchmark_save_checkpoint", False)
    return enabled and (step % args.save_every == 0 or step == args.max_steps)


def resolve_train_scope(args):
    if benchmark_run(args) and args.adapter:
        stored = torch.load(args.adapter, map_location="cpu", weights_only=False, mmap=True).get("train_scope", "ttn")
        if getattr(args, "train_scope", None) is not None and args.train_scope != stored:
            raise ValueError("benchmark train-scope differs from checkpoint; exact resume is required")
        args.train_scope = stored
    elif getattr(args, "train_scope", None) is None:
        args.train_scope = "ttn"


def resolve_optimizer_policy(args, payload):
    """New joint runs use origins; exact resumes never reassign old Adam states."""
    stored = payload.get("optimizer_policy", "legacy") if payload else None
    requested = getattr(args, "optimizer_policy", None)
    if payload and (args.resume or getattr(args, "unfreeze", False)):
        if requested is not None and requested != stored:
            raise ValueError("resume/unfreeze optimizer parameter policy mismatch")
        args.optimizer_policy = stored
    else:
        args.optimizer_policy = requested or ("origin" if args.train_scope in ("dit", "ttn-new") else "legacy")


def train_command(args):
    from .distributed import ParallelTraining, save_training_checkpoint, restore_training_checkpoint, rank_world
    from .parallel_checkpoint import validate_training_checkpoint, validate_unfreeze_checkpoint, restore_training_progress
    from .parallel_data import ResumableBatchStream
    from .training_health import audit_training_parameters, FirstUpdateProbe
    from .provenance import implementation_identity
    from .stability import stability_rows, progress_line, anchor_gradient_scales
    from .sana import build_sana
    import hashlib
    import inspect
    resolve_train_scope(args)
    args._failure_phase = "model-build"
    model, config, settings = build(args)
    from .anchor import configure_train_scope, is_ttn_parameter
    if getattr(args, "train_scope", "ttn") != "ttn":
        configure_train_scope(model, args.train_scope)
    mode = getattr(args, "parallel", "single")
    rank, world = rank_world()
    k = args.tbptt or settings.get("tbptt", 2)
    payload = read_checkpoint(args.adapter, model, resume=args.resume) if args.adapter else None
    resolve_optimizer_policy(args, payload)
    identity = _training_identity(args, config, settings, k)
    unfreeze = getattr(args, "unfreeze", False)
    resume_execution = None
    if unfreeze:
        if args.resume or not payload or getattr(args, "train_scope", "ttn") != "dit":
            raise ValueError("unfreeze requires --adapter, --train-scope dit and no --resume")
        if Path(args.output).resolve() == Path(args.adapter).resolve().parent:
            raise ValueError("unfreeze requires a separate output directory")
        validate_unfreeze_checkpoint(payload, mode, world, identity)
    if args.resume and payload.get("distributed"):
        resume_execution = validate_training_checkpoint(payload, mode, world, identity, benchmark=benchmark_run(args))
    elif args.resume and mode != "single":
        raise ValueError("multi-GPU resume requires optimizer shards; use --adapter without --resume for initialization")
    if payload: apply_checkpoint_weights(model, payload)
    parallel = ParallelTraining(model, SANAFlowLoss(config), mode,
                                activation_offload=getattr(args, "activation_offload", "none"),
                                memory_trace=getattr(args, "memory_trace", False))
    # FSDP2 optimizer must be constructed AFTER parameters become DTensors.
    optimizer = make_optimizer(model, settings.get("learning_rate", 1e-5),
                               backbone_lr=getattr(args, "backbone_lr", 1e-6), policy=args.optimizer_policy)
    if not benchmark_run(args) or getattr(args, "benchmark_save_checkpoint", False):
        from .parallel_checkpoint import preflight_checkpoint
        args._failure_phase = "startup-disk-budget"
        budget = preflight_checkpoint(Path(args.output) / "last.pt", parallel, optimizer)
        if rank == 0: print("[TTN checkpoint budget] " + json.dumps(budget), flush=True)
    step, cursor = 0, None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stream = None
    tokenizer = encoder = None
    encoder_device = None
    if args.batch_file:
        bundle_path = args.batch_file.replace("{rank}", str(rank))
        batch = load_bundle(bundle_path, args.device)
        if batch["clean_latents"].shape[0] != 1: raise ValueError("reference training uses one clip per rank")
    else:
        from train_video_scripts.train_sana_wm_stage1 import _build_dataloader
        from diffusion.model.builder import get_tokenizer_and_text_encoder
        config.train.train_batch_size = 1
        # Dataset construction may shuffle its internal item list. Use identical
        # seeds on all ranks before constructing it, then separate training RNGs.
        seed_everything(args.seed)
        loader = _build_dataloader(config, config.text_encoder.model_max_length, world, rank)
        if not config.data.load_text_feat:
            requested = getattr(args, "text_encoder_device", "auto")
            encoder_device = ("cpu" if world > 1 else args.device) if requested == "auto" else requested
            if encoder_device == "cuda": encoder_device = args.device
            tokenizer, encoder = get_tokenizer_and_text_encoder(config.text_encoder.text_encoder_name, encoder_device)
            encoder.eval().requires_grad_(False)
    seed_everything(args.seed + rank)
    if args.resume:
        args._failure_phase = "checkpoint-restore"
        if payload.get("distributed"):
            step, cursor = restore_training_checkpoint(args.adapter, parallel, optimizer, identity,
                                                      benchmark=benchmark_run(args))
        else:
            step = load_checkpoint(args.adapter, model, optimizer, resume=True)
    elif unfreeze:
        step, cursor = restore_training_progress(args.adapter, parallel, identity)
    if not args.batch_file:
        stream = ResumableBatchStream(loader, seed=args.seed, rank=rank, world=world, state=cursor)
    parameter_report = audit_training_parameters(model, optimizer)
    benchmark_steps = getattr(args, "ttn_benchmark_stable_steps", 0)
    benchmark_records = []
    if benchmark_steps:
        if args.ttn_profile or args.ttn_layout_audit or os.getenv("CUDA_LAUNCH_BLOCKING") == "1" or os.getenv("TORCH_LOGS"):
            raise ValueError("throughput benchmark requires separate profiler/layout/recompile/synchronous diagnosis runs")
        args.max_steps = step + benchmark_steps + 1
        args.save_every = args.max_steps + 1  # Benchmark checkpoint IO only after the measurement window.
        if args.ttn_core_backend == "compiled":
            from .compiled import benchmark_warmup
            benchmark_warmup(True)
    elif getattr(args, "ttn_profiler_trace", None):
        args.max_steps = step + 1
    if rank == 0:
        sources = {"cli": Path(__file__), "anchor": Path(inspect.getfile(type(model.blocks[3].attn))),
                   "sana": Path(inspect.getfile(build_sana)), "optimizer": Path(inspect.getfile(make_optimizer)),
                   "training": Path(inspect.getfile(train_clip)), "distributed": Path(inspect.getfile(ParallelTraining))}
        run = {"arguments": vars(args), "parallel": mode, "world_size": world, "global_batch": world,
               "training": identity, "base": model.base_load_report, "torch": torch.__version__,
               "launch": getattr(args, "launch", None), "parameters": parameter_report,
               "storage": parallel.storage_record(optimizer),
               "provenance": implementation_identity(),
               "execution": execution_report(model), "resume_execution": resume_execution, "precision": precision_audit(),
               "camera_attention": model.ttn_system.config.camera_attention,
               "history_protocol": {"clean_commits": "GT after current noisy loss", "camera_cache": "all previous chunks within clip",
                                    "prefill": "initial frame once; independent GDN/FFN scratch caches", "telemetry": "detached statistics; rank/head separated",
                                    "persistent_gradient": "live inside TBPTT" if model.ttn_system.config.persistent_meta else "detached",
                                    "local_lifecycle": "ephemeral per noisy call" if model.ttn_system.config.local_update else "disabled"},
               "implementation": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                  for name, path in sources.items()}}
        if unfreeze:
            run["initialization"] = {"checkpoint": str(Path(args.adapter).resolve()), "step": step,
                                     "optimizer_reset": True, "rng_and_data_cursor_restored": True}
            print(f"[TTN unfreeze] source_step={step} next_step={step + 1} optimizer=reset rng/data=restored", flush=True)
        elif payload:
            run["initialization"] = {"source": "checkpoint", "step": payload["step"],
                                     "checkpoint": str(Path(args.adapter).resolve()), "resume": args.resume}
        else:
            run["initialization"] = {"source": "pretrained SANA + fresh TTN", "step": 0,
                "sana_inherited": "original base SHA256; visual/camera mappings preserved",
                "ttn_new": "beta/controller heads zero; controller body random; normalized random generators",
                "episode_state": "S/psi zero; never optimizer parameters", "seed": args.seed}
        (output / "run_config.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        json_record(output / "implementations.jsonl", {"next_step": step + 1, "train_scope": args.train_scope if hasattr(args, "train_scope") else "ttn",
                                                     **run["provenance"]})
        print("[TTN trainable] " + json.dumps({k: parameter_report[k] for k in
              ("stage", "train_scope", "weight_scope", "trainable_numel", "frozen_numel", "optimizer_groups")}), flush=True)
    first_update = True
    while step < args.max_steps:
        args._failure_step = step + 1
        args._failure_phase = "training-update"
        iteration_started = time.perf_counter()
        if stream is not None: batch = _dataset_batch(next(stream), config, args, tokenizer, encoder, encoder_device)
        update_probe = FirstUpdateProbe(model) if first_update else None
        from .benchmark import profiler_update
        with profiler_update(getattr(args, "ttn_profiler_trace", None) if first_update else None, rank):
            result, timing = timed_cuda(lambda: train_update(model, config, batch, optimizer, k, parallel))
        result["anchor_gradients"] = anchor_gradient_scales(model)
        result["storage"] = parallel.storage_record(optimizer)
        if update_probe is not None:
            result["parameter_update"] = update_probe.report(optimizer)
            first_update = False
        elif first_update and "optimizer_updates" in result:
            result["parameter_update"] = result["optimizer_updates"]
            first_update = False
        step += 1
        record = {"step": step, "stage": model.ttn_system.config.stage, "tbptt": k,
                  **_training_record(model, result, timing, parallel)}
        if "parameter_update" in result:
            health = {"step": step, "stage": model.ttn_system.config.stage,
                      "ranks": [{"rank": r["rank"], **r["parameter_update"]} for r in record["ranks"]]}
            if rank == 0:
                (output / "first_update.json").write_text(json.dumps(health, indent=2), encoding="utf-8")
                for group, values in result["parameter_update"]["groups"].items():
                    if not is_ttn_parameter(group + "."): continue
                    print(f"[TTN update] {group}: grad={values['grad_norm']:.6g}, "
                          f"delta={values['delta_norm']:.6g}, changed={values['changed_elements']}, "
                          f"optimizer_states={values['optimizer_state_parameters']}", flush=True)
                if getattr(model, "ttn_train_scope", "ttn") == "dit":
                    print("[DiT update] " + json.dumps(result["parameter_update"]["backbone"]), flush=True)
            if any(r["missing_core_gradients"] for r in health["ranks"]):
                raise RuntimeError("first training update disconnected core TTN projections; inspect first_update.json")
        if rank == 0:
            record["implementation_id"] = run["provenance"]["source_fingerprint"]
            json_record(output / "train.jsonl", record)
            for row in stability_rows(record): json_record(output / "stability.jsonl", row)
            print(progress_line(record, args.max_steps), flush=True)
        if benchmark_steps:
            record["local_iteration_seconds"] = time.perf_counter() - iteration_started
            benchmark_records.append(record)
            if len(benchmark_records) == 1:
                # Cold update validates real backward after independent helper warmup.
                if torch.cuda.is_available(): torch.cuda.synchronize()
                if torch.distributed.is_initialized(): torch.distributed.barrier()
                if args.ttn_core_backend == "compiled": benchmark_warmup(True, stable=True)
        if benchmark_steps and step == args.max_steps:
            from .distributed import gather_records
            # One gather after the whole window, never an added per-step synchronization.
            iteration_times = gather_records({"rank": rank, "seconds": [r["local_iteration_seconds"] for r in benchmark_records]})
            for i, sample in enumerate(benchmark_records):
                sample["iteration_seconds"] = max(r["seconds"][i] for r in iteration_times)
                for r in sample["ranks"]:
                    r["iteration_seconds"] = next(t["seconds"][i] for t in iteration_times if t["rank"] == r["rank"])
        if benchmark_steps and step == args.max_steps and rank == 0:
            from .benchmark import stable_summary
            from .evaluation import file_sha256
            performance = {**stable_summary(benchmark_records, benchmark_steps), "execution": execution_report(model),
                           "precision": precision_audit(), "provenance": run["provenance"],
                           "checkpoint_input": args.adapter,
                           "checkpoint_sha256": file_sha256(args.adapter) if args.adapter else None, "training_identity": identity,
                           "scope": args.train_scope, "world_size": world}
            if args.ttn_core_backend == "compiled":
                from .compiled import warmup_report
                performance["compile_warmup"] = warmup_report()
            (output / "performance.json").write_text(json.dumps(performance, indent=2), encoding="utf-8")
        if checkpoint_due(args, step):
            args._failure_phase = "checkpoint-save"
            save_training_checkpoint(output / "last.pt", parallel, optimizer, step,
                                     stream.state_dict() if stream is not None else {}, identity)
    if benchmark_steps and args.ttn_core_backend == "compiled": benchmark_warmup(False)


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
        "execution": execution_report(model), "precision": precision_audit(),
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
            lambda: train_update(model, config, batch, optimizer, args.tbptt or settings.get("tbptt", 2),
                                 activation_offload=args.activation_offload, memory_trace=args.memory_trace))
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
                "activation_offload": result["activation_offload"],
                "memory_phases": result["memory_phases"],
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


def distributed_smoke_command(args):
    """A/B/C two-rank updates, plain checkpoint exports and rank-zero CFG rollouts."""
    import gc
    from torch import distributed as dist
    from .distributed import ParallelTraining, save_training_checkpoint, rank_world
    rank, world = rank_world()
    if world < 2: raise ValueError("distributed-smoke requires at least two processes")
    if args.frames < 4 or args.frames % 3 != 1: raise ValueError("smoke uses 1+3n latent frames")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    previous = None
    reports = []
    for stage in args.stages:
        seed_everything(args.seed)
        model, config, settings = build(args, stage)
        if previous: load_checkpoint(previous, model)
        parallel = ParallelTraining(model, SANAFlowLoss(config), args.parallel,
                                    activation_offload=args.activation_offload, memory_trace=args.memory_trace)
        optimizer = make_optimizer(model, settings.get("learning_rate", 1e-5))
        seed_everything(args.seed + rank)
        batch = load_bundle(args.batch_file.replace("{rank}", str(rank)), args.device) if args.batch_file else (
            synthetic_bundle(config, args.device, args.frames, args.latent_height, args.latent_width))
        if batch["clean_latents"].shape[0] != 1: raise ValueError("smoke training uses one clip per rank")
        k = args.tbptt or settings.get("tbptt", 2)
        result, timing = timed_cuda(lambda: train_update(model, config, batch, optimizer, k, parallel))
        expected = len(chunk_ranges(batch["clean_latents"].shape[2])) + 1
        if result["runtime"].commit_count != expected or result["runtime"].predict_count != expected:
            raise AssertionError("training state counters disagree")
        if any(len(ids) != batch["clean_latents"].shape[2] for ids in result["runtime"].committed_frame_ids):
            raise AssertionError("training duplicate/missing writes")
        gradients = [p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                     for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise AssertionError("invalid training gradients")
        train_record = _training_record(model, result, timing, parallel)
        previous = output / stage / "last.pt"
        save_training_checkpoint(previous, parallel, optimizer, 1, {}, {"tbptt": k, "seed": args.seed})
        # Release the training model before rebuilding a plain inference model;
        # otherwise a rank-zero rollout could double the model memory footprint.
        del model, parallel, optimizer, result, batch, gradients
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        if rank == 0:
            seed_everything(args.seed)
            model, config, _ = build(args, stage)
            load_checkpoint(previous, model)
            batch = load_bundle(args.batch_file.replace("{rank}", "0"), args.device) if args.batch_file else (
                synthetic_bundle(config, args.device, args.frames, args.latent_height, args.latent_width))
            sampled, timing = timed_cuda(lambda: rollout(model, config, batch, args.steps, args.cfg_scale,
                                                         args.cached_blocks))
            latents, runtime, chunks = sampled
            report = {"stage": stage, "parallel": args.parallel, "world_size": world, "seed": args.seed,
                      "synthetic_inputs": not bool(args.batch_file), "tbptt": k, "steps": args.steps,
                      "cfg_scale": args.cfg_scale, "train": train_record,
                      "rollout": {"finite": bool(torch.isfinite(latents).all()), "commits": runtime.commit_count,
                                  "predictions": runtime.predict_count, "chunks": chunks,
                                  "branch_state_norms": runtime.world_state.flatten(1).norm(dim=-1).tolist(), **timing}}
            (output / stage / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            torch.save({"latents": latents.cpu()}, output / stage / "rollout.pt")
            reports.append(report)
            print(json.dumps(report), flush=True)
            del model, batch, sampled, latents, runtime
            gc.collect()
            torch.cuda.empty_cache()
        dist.barrier()
    if rank == 0: (output / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


def distributed_check_command(args):
    from .distributed import rank_world
    if rank_world()[0] == 0:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "distributed.json").write_text(json.dumps(args.launch, indent=2), encoding="utf-8")
        print(json.dumps(args.launch), flush=True)


def diagnose_update_command(args):
    """Exactly one distributed-smoke training update, including its rank-local seeds."""
    from .distributed import ParallelTraining, rank_world
    rank, world = rank_world()
    if args.frames < 4 or args.frames % 3 != 1: raise ValueError("diagnosis uses 1+3n latent frames")
    seed_everything(args.seed)
    model, config, settings = build(args)
    parallel = ParallelTraining(model, SANAFlowLoss(config), args.parallel,
                                activation_offload=args.activation_offload, memory_trace=args.memory_trace)
    optimizer = make_optimizer(model, settings.get("learning_rate", 1e-5))
    seed_everything(args.seed + rank)  # Match distributed-smoke rank0 on the single-card test.
    batch = load_bundle(args.batch_file.replace("{rank}", str(rank)), args.device) if args.batch_file else (
        synthetic_bundle(config, args.device, args.frames, args.latent_height, args.latent_width))
    if batch["clean_latents"].shape[0] != 1: raise ValueError("diagnosis uses one clip per rank")
    k = args.tbptt or settings.get("tbptt", 2)
    result, timing = timed_cuda(lambda: train_update(model, config, batch, optimizer, k, parallel))
    expected = len(chunk_ranges(batch["clean_latents"].shape[2])) + 1
    runtime = result["runtime"]
    if runtime.commit_count != expected or runtime.predict_count != expected:
        raise AssertionError("training state counters disagree")
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise AssertionError("invalid training gradients")
    report = {"stage": model.ttn_system.config.stage, "tbptt": k, "seed": args.seed,
              "synthetic_inputs": not bool(args.batch_file), "distributed": args.launch,
              "train": _training_record(model, result, timing, parallel)}
    if rank == 0:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "update.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "infer", "smoke", "distributed-smoke", "distributed-check", "diagnose-update", "evaluate", "align-chunk", "stage-evaluate"))
    parser.add_argument("--training-run", help="completed training output directory; evaluate restores its config")
    parser.add_argument("--alignment-timesteps", type=int, nargs="+", default=[0, 250, 500, 750, 999],
                        help="align-chunk: fixed training-schedule indices, paired GT/noise, no CFG")
    parser.add_argument("--alignment-grad-timestep", type=int, default=500,
                        help="align-chunk: read-only autograd.grad probe at this index; no optimizer step")
    parser.add_argument("--eval-cases", type=int, default=1, help="deterministic unique example clips, diagnostic only")
    parser.add_argument("--fixed-cases", help="immutable CPU case bundle pinned by stage evaluation (shared full horizon)")
    parser.add_argument("--ttn-ablation", choices=("full", "no-ttt", "identity", "no-local", "no-persistent"), default="full",
                        help="evaluate only: runtime intervention on the same C checkpoint, without changing stage/weights")
    parser.add_argument("--history-source", choices=("generated", "gt", "ttn-gt", "native-gt"), default="generated",
                        help="evaluate only: gt commits current GT after prediction to ALL caches; output stays generated")
    parser.add_argument("--state-diagnostics", action="store_true", help="evaluate only: detached per-head spectra/transport; adds diagnostic overhead")
    parser.add_argument("--eval-methods", nargs="+", choices=("sana", "ttn"), default=["sana", "ttn"],
                        help="evaluate only: allow TTN-only mechanism runs without repeating the SANA baseline")
    parser.add_argument("--noise-frames", type=int, help="generate noise at this horizon, then take a prefix; use 61 for paired 13/61 diagnostics")
    parser.add_argument("--revisit-min-gap", type=int, default=30, help="minimum revisit separation in latent frames")
    parser.add_argument("--revisit-distance-fraction", type=float, default=.02, help="pose distance / trajectory extent")
    parser.add_argument("--revisit-angle-deg", type=float, default=5.)
    parser.add_argument("--revisit-max-pairs", type=int, default=5)
    parser.add_argument("--parallel", choices=("auto", "single", "ddp", "fsdp2"), default="auto",
                        help="auto: single with one process, DDP under Slurm, otherwise native FSDP2")
    parser.add_argument("--distributed-timeout", type=int, default=600,
                        help="process group timeout in seconds (default: 600)")
    parser.add_argument("--ttn-core-backend", choices=("reference", "reuse", "compiled"), default="reference")
    parser.add_argument("--ttn-psi-backend", choices=("reference", "projected"), default="reference",
                        help="Triton is conditional on target-GPU profile; not implemented yet")
    parser.add_argument("--ttn-profile", action="store_true", help="annotate profiler ranges; separate from throughput runs")
    parser.add_argument("--ttn-layout-audit", action="store_true", help="record unique production tensor layouts per rank")
    parser.add_argument("--ttn-benchmark-stable-steps", type=int, default=0,
                        help="train only: one cold update plus N stable updates in a private output directory")
    parser.add_argument("--ttn-profiler-trace", help="train only: export first-update CUDA traces, separate from throughput")
    parser.add_argument("--benchmark-save-checkpoint", action="store_true",
                        help="explicitly save resume bundles in benchmark/profiler/layout runs (default: no checkpoint IO)")
    parser.add_argument("--ttn-compare-reference", action="store_true",
                        help="evaluate: add same-weight TTN reference rollout and direct latent/S/psi differences")
    parser.add_argument("--activation-offload", choices=("none", "cpu"), default="none",
                        help="store backward saved tensors in pinned host RAM; single/DDP/FSDP2, no forward replay")
    parser.add_argument("--memory-trace", action="store_true",
                        help="print per-rank prefill/noisy/clean/backward/optimizer memory diagnostics")
    parser.add_argument("--cuda-trace", action="store_true",
                        help="synchronized ATen/autograd layout logs in OUTPUT/cuda-trace; use eager mode and blocking=1")
    parser.add_argument("--cross-attn-backend", choices=("auto", "math", "flash", "efficient"), default="auto",
                        help="backend for the standard text cross-attention SDPA call only; forced choices have no fallback")
    parser.add_argument("--diagnostic-unmask-all-valid", action="store_true",
                        help="synthetic diagnose-update only: omit verified all-one text masks at SDPA, retaining upstream masks")
    parser.add_argument("--dataset-root", help="root for SANA-config relative raw data and VAE cache paths")
    parser.add_argument("--data-dir", help="override the single configured raw zip dataset directory")
    parser.add_argument("--vae-cache-dir", help="override the latent cache directory")
    parser.add_argument("--text-encoder-device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="auto keeps the frozen text encoder on CPU for multi-GPU training")
    parser.add_argument("--config", default=str(DEFAULT_REFERENCE),
                        help="new runs default to C/original SANA camera; legacy linear-camera checkpoints need their explicit config")
    parser.add_argument("--sana-config")
    parser.add_argument("--base-weights", help="local mirror of the specified SANA teacher, or hf:// URI")
    parser.add_argument("--adapter", help="TTN checkpoint; stage comes from --stage or reference config")
    parser.add_argument("--camera-attention", choices=("linear", "sana"),
                        help="evaluate only: explicitly override the loaded TTN camera mixer/cache semantics")
    parser.add_argument("--camera-ablation", action="store_true", help="evaluate only: allow/label a camera operator different from training")
    parser.add_argument("--train-scope", choices=("ttn", "ttn-visual", "ttn-new", "dit"), default=None,
                        help="train: ttn-new trains only added parameters; dit trains the complete DiT")
    parser.add_argument("--optimizer-policy", choices=("origin", "legacy"),
                        help="train: joint defaults to origin (new TTN vs inherited SANA); resumes inherit saved policy")
    parser.add_argument("--backbone-lr", type=float, default=1e-6,
                        help="joint origin policy: LR for all inherited SANA visual/camera/DiT parameters")
    parser.add_argument("--stage", choices=("A", "B", "C"),
                        help="A: identity Predict + Correct/Read; B: camera-conditioned Predict + Correct/Read; "
                             "C: B plus per-clean-chunk psi adaptation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", default="output/worldttn")
    parser.add_argument("--batch-file", help="precomputed tensor bundle; optional {rank} expands per process")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--unfreeze", action="store_true",
                        help="train: initialize DiT from C/sana-camera ttn-new or legacy visual warmup; reset optimizer, restore progress")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--tbptt", type=int, choices=(1, 2, 4))
    parser.add_argument("--steps", type=int, help="default 4 for smoke; evaluate requires an explicit quality schedule")
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument("--cached-blocks", type=int, default=-1)
    parser.add_argument("--stages", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"])
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--latent-height", type=int, default=22)
    parser.add_argument("--latent-width", type=int, default=40)
    from .sink import add_sink_arguments, sink_options_from_args
    add_sink_arguments(parser)
    from .replay import add_replay_arguments, replay_options_from_args
    add_replay_arguments(parser)
    args = parser.parse_args(argv)
    try:
        sink = sink_options_from_args(args)
        replay = replay_options_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    try:
        resolve_train_scope(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.benchmark_save_checkpoint and (args.command != "train" or not benchmark_run(args)):
        parser.error("--benchmark-save-checkpoint requires a train benchmark/profiler/layout run")
    if args.command != "evaluate" and (args.ttn_ablation != "full" or args.history_source != "generated"
            or args.state_diagnostics or args.eval_methods != ["sana", "ttn"] or args.tla_sink != "off"
            or args.tla_replay != "off"):
        parser.error("mechanism interventions are evaluate-only; training semantics are unchanged")
    if args.history_source in ("ttn-gt", "native-gt") and args.eval_methods != ["ttn"]:
        parser.error("mixed history interventions require TTN-only --eval-methods ttn")
    if sink.active and args.history_source != "generated":
        parser.error("sink experiments require generated history; run H1/H2 separately")
    if len(set(args.eval_methods)) != len(args.eval_methods): parser.error("--eval-methods must be distinct")
    if args.camera_attention is not None and args.command != "evaluate":
        parser.error("--camera-attention is an evaluate-only ablation; train with an explicit reference config")
    if args.camera_ablation and args.command != "evaluate": parser.error("--camera-ablation is evaluate-only")
    if args.noise_frames is not None and (args.command not in ("evaluate", "align-chunk", "stage-evaluate") or args.noise_frames < args.frames):
        parser.error("--noise-frames must cover the diagnostic horizon")
    if args.fixed_cases and args.command not in ("evaluate", "align-chunk", "stage-evaluate"): parser.error("--fixed-cases is diagnostic-only")
    if args.train_scope != "ttn" and args.command != "train":
        parser.error("--train-scope is only supported for train; inference reads weight scope from checkpoint")
    if args.optimizer_policy is not None and args.command != "train": parser.error("--optimizer-policy is train-only")
    if args.unfreeze and (args.command != "train" or args.train_scope != "dit" or not args.adapter or args.resume):
        parser.error("--unfreeze requires train --train-scope dit --adapter and no --resume")
    if not math.isfinite(args.backbone_lr) or args.backbone_lr <= 0:
        parser.error("--backbone-lr must be finite and positive")
    if args.command in ("evaluate", "stage-evaluate"):
        if not args.training_run or args.steps is None:
            parser.error("evaluate requires --training-run and explicit --steps")
        if args.batch_file or args.resume:
            parser.error("evaluate reads real zip data from --training-run; no --batch-file or --resume")
        if args.frames < 4 or args.frames % 3 != 1 or args.eval_cases < 1:
            parser.error("evaluate requires positive cases and 1+3n latent frames")
    if args.command == "stage-evaluate" and (args.frames != 61 or not args.fixed_cases or args.adapter):
        parser.error("stage-evaluate requires --frames 61 --fixed-cases and the snapshot's last.pt (no --adapter)")
    if args.command == "align-chunk":
        if not args.training_run or args.frames != 4 or args.eval_cases < 1:
            parser.error("align-chunk requires --training-run, --frames 4 and positive --eval-cases")
        if args.batch_file or args.resume or args.steps is not None:
            parser.error("align-chunk uses real GT data without a sampler, --batch-file, --steps or --resume")
        if min(*args.alignment_timesteps, args.alignment_grad_timestep) < 0 or len(set(args.alignment_timesteps)) != len(args.alignment_timesteps):
            parser.error("align-chunk requires distinct nonnegative timesteps")
    try:
        ExecutionOptions(args.ttn_core_backend, args.ttn_psi_backend)
    except ValueError as error:
        parser.error(str(error))
    if args.ttn_core_backend != "reference" and (args.ttn_ablation != "full" or args.state_diagnostics or
            args.history_source != "generated" or args.command in ("align-chunk", "stage-evaluate")):
        parser.error("mechanism and alignment diagnostics require reference/reference")
    if args.ttn_benchmark_stable_steps < 0 or (args.ttn_benchmark_stable_steps or args.ttn_profiler_trace) and args.command != "train":
        parser.error("benchmark/profiler update flags require train and nonnegative stable steps")
    if args.ttn_benchmark_stable_steps and args.ttn_profiler_trace:
        parser.error("profiler and stable throughput runs must be separate")
    if args.ttn_compare_reference and args.command not in ("evaluate", "stage-evaluate"):
        parser.error("--ttn-compare-reference requires evaluate or stage-evaluate")
    if args.steps is None: args.steps = 4
    if args.diagnostic_unmask_all_valid and (args.command != "diagnose-update" or args.batch_file):
        parser.error("--diagnostic-unmask-all-valid requires synthetic diagnose-update without --batch-file")
    os.environ.setdefault("DISABLE_XFORMERS", "1")
    os.environ["SANA_LOG_GLOBAL_RANK_ONLY"] = "1"
    if args.cuda_trace:
        if os.environ.get("CUDA_LAUNCH_BLOCKING") != "1": parser.error("--cuda-trace requires CUDA_LAUNCH_BLOCKING=1 before Python starts")
        if os.environ.get("GDN_DISABLE_COMPILE", "0") in ("0", "false"):
            parser.error("--cuda-trace requires GDN_DISABLE_COMPILE=1; operator tracing is an eager-only diagnostic")
        os.environ["TTN_CUDA_TRACE_DIR"] = str(Path(args.output) / "cuda-trace")
    else:
        os.environ.pop("TTN_CUDA_TRACE_DIR", None)
    if args.resume and not args.adapter: parser.error("--resume requires --adapter")
    from .distributed import initialize, resolve_launch_environment, check_distributed, rank_world
    try:
        launch = resolve_launch_environment()
    except ValueError as error:
        parser.error(str(error))
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        parser.error("server entrypoints require CUDA; local CPU validation is python -m pytest tests/ttn")
    if min(args.steps, args.max_steps, args.save_every) < 1: parser.error("step counts must be positive")
    if int(os.environ.get("SANA_CP_SIZE", "1")) > 1: parser.error("TTN reference requires CP=1")
    world = launch["world_size"]
    if args.command in ("infer", "smoke", "evaluate", "align-chunk", "stage-evaluate") and (world > 1 or args.parallel not in ("auto", "single")):
        parser.error("infer/smoke/evaluate/align-chunk are single-card; use distributed-smoke for multi-rank training checks")
    if args.command in ("distributed-smoke", "distributed-check") and (world < 2 or args.parallel == "single"):
        parser.error(f"{args.command} requires srun, accelerate launch or torchrun with at least two processes")
    try:
        args.parallel, device = initialize(args.parallel, args.device, timeout_seconds=args.distributed_timeout)
    except ValueError as error:
        parser.error(str(error))
    args.device = str(device)
    seed_everything(args.seed)
    try:
        args.launch = check_distributed(device)
        if world > 1 and args.command != "distributed-check" and rank_world()[0] == 0:
            print(json.dumps({"distributed": args.launch}), flush=True)
        from .evaluation import evaluate_command
        from .alignment import alignment_command
        from .stage_evaluation import stage_evaluate_command
        {"train": train_command, "infer": infer_command, "smoke": smoke_command, "evaluate": evaluate_command,
         "align-chunk": alignment_command, "stage-evaluate": stage_evaluate_command,
         "distributed-smoke": distributed_smoke_command, "distributed-check": distributed_check_command,
         "diagnose-update": diagnose_update_command}[args.command](args)
    except Exception as error:
        from .failure import record_failure
        record_failure(args.output, getattr(args, "_failure_phase", args.command), error,
                       step=getattr(args, "_failure_step", None))
        raise
    finally:
        if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == "__main__": main()
