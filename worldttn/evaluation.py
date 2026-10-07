"""Paired autoregressive latent diagnostics, not a decoded SANA-WM benchmark.

Free-rollout future GT stays on CPU and is used only after sampling. Explicit
GT-history diagnostics reveal each clean chunk only AFTER its prediction. Pose-selected returns
are candidates, not proof that the same visible scene was actually revisited.
"""
import gc
import hashlib
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .core import TTNConfig, BASE_REVISION
from .training import chunk_ranges


HISTORY_ACCESS = {
    "generated": "one observed frame; generated chunks update all history",
    "gt": "GT slice after current denoising; commits TTN + GDN/camera/FFN caches; output remains generated",
    "ttn-gt": "GT clean pass supplies TTN S/psi; generated clean pass supplies native GDN/camera/FFN caches",
    "native-gt": "generated clean pass supplies TTN S/psi; GT clean pass supplies native GDN/camera/FFN caches",
}


def validate_inference_interventions(args, config):
    """Reject incompatible diagnostic combinations before loading cases/models."""
    from .sink import sink_options_from_args
    sink = sink_options_from_args(args)
    history = getattr(args, "history_source", "generated")
    if history not in HISTORY_ACCESS:
        raise ValueError("unknown clean history source")
    mixed = history in ("ttn-gt", "native-gt")
    if sink.active and "ttn" not in getattr(args, "eval_methods", ["sana", "ttn"]):
        raise ValueError("active sink requires a TTN evaluation method")
    if mixed and tuple(getattr(args, "eval_methods", ["sana", "ttn"])) != ("ttn",):
        raise ValueError("mixed history interventions require TTN-only evaluation")
    if sink.active or mixed:
        if config.stage != "C" or getattr(args, "ttn_ablation", "full") != "full":
            raise ValueError("sink and mixed history require Full Stage C")
        if (config.camera_attention != "sana" or getattr(args, "camera_ablation", False)
                or getattr(args, "camera_attention", None) not in (None, "sana")):
            raise ValueError("sink and mixed history require native SANA camera without overrides")
        if (getattr(args, "ttn_core_backend", "reference"),
                getattr(args, "ttn_psi_backend", "reference")) != ("reference", "reference"):
            raise ValueError("sink and mixed history require reference/reference")
        if getattr(args, "ttn_compare_reference", False):
            raise ValueError("sink/history interventions cannot be used for backend comparison")
    if sink.active and history != "generated":
        raise ValueError("sink experiments require generated history; run H1/H2 separately")
    return sink


def latent_metrics(generated, reference, revisit_pairs, *, training_frames=None):
    if generated.shape != reference.shape or generated.ndim != 5 or generated.shape[0] != 1:
        raise ValueError("paired latents must have identical [1,C,F,H,W] shapes")
    frames = generated.shape[2]
    if frames < 4 or frames % 3 != 1:
        raise ValueError("metrics require 1+3n latent frames")
    if not torch.isfinite(generated).all() or not torch.isfinite(reference).all():
        raise ValueError("nonfinite paired latents")
    generated, reference = generated.float().cpu(), reference.float().cpu()
    errors = (generated - reference).square().mean((0, 1, 3, 4))
    chunks = [{"chunk": i, "start": a, "end": b, "metric_start": max(a, 1),
               "latent_mse": errors[max(a, 1):b].mean().item()}
              for i, (a, b) in enumerate(chunk_ranges(frames))]
    revisits = []
    for pair in revisit_pairs:
        a, b = pair["frame_a"], pair["frame_b"]
        if not isinstance(a, int) or not isinstance(b, int) or not 0 < a < b < frames:
            raise ValueError("revisit pairs must index two generated frames in this rollout")
        revisits.append({**pair, "return_gt_latent_mse": errors[b].item(),
                         "generated_pair_latent_mse": (generated[:, :, a] - generated[:, :, b]).square().mean().item(),
                         "gt_pair_latent_mse": (reference[:, :, a] - reference[:, :, b]).square().mean().item()})
    t = torch.arange(1, frames, dtype=torch.float32)
    t -= t.mean()
    slope = (t * (errors[1:] - errors[1:].mean())).sum() / t.square().sum()
    result = {"mean_future_latent_mse": errors[1:].mean().item(),
            "final_chunk_latent_mse": chunks[-1]["latent_mse"],
            "error_slope_per_latent_frame": slope.item(),
            "per_frame_latent_mse": [None, *errors[1:].tolist()], "chunks": chunks,
            "revisits": revisits,
            "revisit_return_gt_latent_mse": _mean([p["return_gt_latent_mse"] for p in revisits]),
            "revisit_generated_pair_latent_mse": _mean([p["generated_pair_latent_mse"] for p in revisits])}
    if training_frames is not None:
        if training_frames < 1: raise ValueError("training horizon must be positive")
        tail = errors[training_frames:]
        indices = torch.arange(training_frames, frames, dtype=torch.float32)
        indices -= indices.mean() if indices.numel() else 0
        result.update(after_training_horizon_frame_count=tail.numel(),
            after_training_horizon_mean_latent_mse=tail.mean().item() if tail.numel() else None,
            after_training_horizon_error_slope=((indices * (tail - tail.mean())).sum() / indices.square().sum()).item()
                if tail.numel() > 1 else None)
    return result


def _mean(values):
    return sum(values) / len(values) if values else None


def find_revisits(camera, *, min_gap=30, distance_fraction=.02, angle_deg=5., max_pairs=5):
    """Select near-equal poses/intrinsics with an intervening excursion.

    Ignore observed frame 0 and static/adjacent frames. Distance is relative to
    the trajectory's spatial extent; leave-and-return requires >=10% extent or
    >=15 degrees rotation. These explicit heuristic thresholds are diagnostic.
    """
    if camera.ndim != 2 or camera.shape[1] != 20 or not torch.isfinite(camera).all():
        raise ValueError("revisits require finite [F,20] C2W + intrinsics")
    if min_gap < 1 or max_pairs < 1 or not 0 <= distance_fraction <= 1 or not 0 <= angle_deg <= 180:
        raise ValueError("invalid revisit thresholds")
    camera = camera.double().cpu()
    poses = camera[:, :16].reshape(-1, 4, 4)
    rotations, positions = poses[:, :3, :3], poses[:, :3, 3]
    if not torch.allclose(rotations @ rotations.transpose(-1, -2), torch.eye(3, dtype=torch.float64).expand_as(rotations), atol=1e-3):
        raise ValueError("camera rotations are not orthogonal")
    if (camera[:, 16:18] <= 0).any(): raise ValueError("invalid focal lengths")
    extent = torch.cdist(positions, positions).max().item()
    # Trace(R_a^T R_b) is the Frobenius inner product of the two rotations.
    angles = torch.rad2deg(torch.acos(((rotations.flatten(1) @ rotations.flatten(1).T - 1) / 2).clamp(-1, 1)))
    distances = torch.cdist(positions, positions)
    candidates = []
    for a in range(1, len(camera) - min_gap):
        for b in range(a + min_gap, len(camera)):
            distance = distances[a, b].item()
            angle = angles[a, b].item()
            fraction = distance / extent if extent > 1e-8 else 0.
            if distance > max(1e-8, extent * distance_fraction) or angle > angle_deg:
                continue
            if not torch.allclose(camera[a, 16:], camera[b, 16:], rtol=1e-3, atol=1e-4):
                continue
            excursion = distances[a, a:b + 1].max().item() / extent if extent > 1e-8 else 0.
            angle_excursion = angles[a, a:b + 1].max().item()
            if excursion < .1 and angle_excursion < 15.: continue
            candidates.append({"frame_a": a, "frame_b": b, "distance": distance,
                               "distance_fraction": fraction, "angle_deg": angle,
                               "trajectory_extent": extent, "excursion_fraction": excursion,
                               "excursion_angle_deg": angle_excursion})
    candidates.sort(key=lambda p: (p["distance_fraction"] + p["angle_deg"] / 180, p["frame_b"], p["frame_a"]))
    selected = []
    for pair in candidates:
        if any(abs(pair["frame_b"] - old["frame_b"]) < 3 for old in selected): continue
        selected.append(pair)
        if len(selected) == max_pairs: break
    return selected


def paired_summary(records):
    methods = {"sana": {}, "ttn": {}}
    for row in records:
        key = (row["case_id"], row["seed"])
        if row["method"] not in methods or key in methods[row["method"]]:
            raise ValueError("unknown method or duplicate case/seed")
        methods[row["method"]][key] = row["metrics"]
    if not methods["sana"] or methods["sana"].keys() != methods["ttn"].keys():
        raise ValueError("summary requires a complete SANA/TTN pair for every case/seed")
    names = metric_names(records[0]["metrics"])
    means, deltas = {}, {}
    for name in names:
        pairs = [(methods["sana"][key][name], methods["ttn"][key][name]) for key in methods["sana"]]
        if any((a is None) != (b is None) for a, b in pairs):
            raise ValueError("methods disagree on missing revisit metrics")
        values = [(a, b) for a, b in pairs if a is not None]
        if not all(math.isfinite(a) and math.isfinite(b) for a, b in values):
            raise ValueError("nonfinite paired metrics")
        means[name] = {"sana": _mean([a for a, _ in values]), "ttn": _mean([b for _, b in values]),
                       "paired_count": len(values)}
        deltas[name] = _mean([b - a for a, b in values])
    return {"common_case_seed_count": len(methods["sana"]), "metrics": means,
            "ttn_minus_sana": deltas, "delta_direction": "lower is better; slope is descriptive"}


def metric_names(metrics):
    names = ("mean_future_latent_mse", "final_chunk_latent_mse", "error_slope_per_latent_frame",
             "revisit_return_gt_latent_mse", "revisit_generated_pair_latent_mse",
             "after_training_horizon_mean_latent_mse", "after_training_horizon_error_slope")
    return [name for name in names if name in metrics]


def evaluation_summary(records):
    if {r["method"] for r in records} == {"sana", "ttn"}: return paired_summary(records)
    method = records[0]["method"]
    if len({(r["case_id"], r["seed"]) for r in records}) != len(records):
        raise ValueError("duplicate single-method case/seed")
    return {"case_seed_count": len(records), "metrics": {
        name: {method: _mean([r["metrics"][name] for r in records if r["metrics"][name] is not None]),
               "count": sum(r["metrics"][name] is not None for r in records)}
        for name in metric_names(records[0]["metrics"])}}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    header = json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}).encode()
    return hashlib.sha256(header + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def evaluation_config(run, source):
    """Restore the exact stored model, text, scheduler and data configuration."""
    from .sana import load_sana_config
    config = load_sana_config(source)
    identity = run["training"]
    for current, stored in (("model", "sana_model"), ("text_encoder", "text_encoder"),
                            ("scheduler", "scheduler"), ("data", "data")):
        if not isinstance(identity.get(stored), dict):
            raise ValueError(f"real-data evaluation requires training.{stored}")
        setattr(config, current, replace(getattr(config, current), **identity[stored]))
    config.task = identity["task"]
    if config.data.type != "SanaWMZipLatentDataset" or config.data.load_text_feat:
        raise ValueError("reference evaluation requires raw zip captions and cached SANA latents")
    return config


def select_cases(dataset, frames, count, seed, revisit_options):
    """Reject missing cameras/short samples instead of identity fallback or padding."""
    from .cli import seed_everything
    cases, rejected, seen = [], [], set()
    required_raw = (frames - 1) * dataset.vae_time_stride + 1
    for index, item in enumerate(dataset.dataset):
        key = (item["raw_zip"], item["key"])
        if key in seen: continue
        seen.add(key)
        sidecar = dataset.load_camera_sidecar(item["camera_npz"])
        ids = sidecar["ids"].tolist() if sidecar is not None else []
        reason = None
        if item["key"] not in ids:
            reason = "missing camera ID/sidecar"
        else:
            start, raw_count = map(int, sidecar["ranges"][ids.index(item["key"])])
            if start < 0 or raw_count < required_raw or start + raw_count > min(len(sidecar["pose"]), len(sidecar["intrinsics"])):
                reason = "insufficient raw camera frames"
        if reason:
            rejected.append({"key": item["key"], "reason": reason})
            continue
        seed_everything(seed + len(cases))
        sample = dataset.getdata(index)  # __getitem__ may silently retry another sample
        gt, prompt, _, info, _, _, camera = sample[:7]
        if gt.ndim != 4 or gt.shape[1] != frames:
            rejected.append({"key": item["key"], "reason": "insufficient cached latent horizon"})
            continue
        if not torch.isfinite(gt).all() or camera.shape != (frames, 20):
            raise ValueError(f"invalid reference/camera for {item['key']}")
        plucker = sample[7] if dataset.return_chunk_plucker else None
        if plucker is not None and (plucker.shape[1:] != gt.shape[1:] or not torch.isfinite(plucker).all()):
            raise ValueError(f"invalid chunk Plucker for {item['key']}")
        cases.append({"case_id": f"{item['dataset_name']}/{item['key']}", "seed": seed + len(cases),
                      "reference": gt[None], "camera": camera[None], "plucker": plucker,
                      "prompt": prompt, "info": info, "raw_camera_frames": raw_count,
                      "camera_sidecar": item["camera_npz"], "cache_zip": item["cache_zip"],
                      "revisit_pairs": find_revisits(camera, **revisit_options)})
        if len(cases) == count: break
    if len(cases) != count:
        raise ValueError(f"requested {count} complete real cases; found {len(cases)}; rejected={rejected[:10]}")
    return cases, rejected


def load_evaluation_run(args):
    """Shared identity/config validation for rollout and teacher-forcing probes."""
    from .cli import ROOT, read_reference
    from .sana import resolve_data_paths
    training_run = Path(args.training_run).resolve()
    run = json.loads((training_run / "run_config.json").read_text())
    adapter = Path(args.adapter or training_run / "last.pt").resolve()
    adapter_digest = file_sha256(adapter)
    payload = torch.load(adapter, map_location="cpu", weights_only=False)
    if payload.get("format") != "TTN-SANA-WM-v0.1" or payload.get("base_revision") != BASE_REVISION:
        raise ValueError("not a compatible TTN checkpoint")
    ttn = TTNConfig(**payload["config"])
    if args.stage and args.stage != ttn.stage: raise ValueError("diagnostics must retain the checkpoint stage")
    if payload["stage"] != ttn.stage or run["base"]["sha256"] != payload["base_sha256"]:
        raise ValueError("training run and checkpoint identities disagree")
    last_train = json.loads((training_run / "train.jsonl").read_text().splitlines()[-1])
    if last_train["stage"] != ttn.stage or last_train["step"] != payload["step"]:
        raise ValueError("checkpoint must match the completed training run's final step")
    _, settings = read_reference(args.config)
    source = Path(args.sana_config or run["arguments"].get("sana_config") or settings["sana_config"])
    if not source.is_absolute(): source = ROOT / source
    config = evaluation_config(run, source)
    if args.dataset_root: resolve_data_paths(config, args.dataset_root)
    if args.data_dir or args.vae_cache_dir:
        raise ValueError("evaluate restores training data paths; only --dataset-root is supported")
    return run, config, ttn, adapter, adapter_digest, last_train


def load_evaluation_cases(config, args):
    fixed = getattr(args, "fixed_cases", None)
    if fixed and Path(fixed).is_file():
        bundle = torch.load(fixed, map_location="cpu", weights_only=False)
        if bundle.get("format") != "TTN-fixed-cases-v1" or bundle["seed"] != args.seed or len(bundle["cases"]) != args.eval_cases:
            raise ValueError("fixed case identity/seed/count differs")
        if bundle["data_identity"] != fixed_data_identity(config):
            raise ValueError("fixed cases were selected with a different data/text/geometry configuration")
        cases = []
        for source in bundle["cases"]:
            if source["reference"].shape[2] < args.frames: raise ValueError("fixed case is shorter than requested horizon")
            case = dict(source, reference=source["reference"][:, :, :args.frames].clone(),
                        camera=source["camera"][:, :args.frames].clone())
            if source["plucker"] is not None: case["plucker"] = source["plucker"][:, :args.frames].clone()
            cases.append(case)
        options = {"min_gap": args.revisit_min_gap, "distance_fraction": args.revisit_distance_fraction,
                   "angle_deg": args.revisit_angle_deg, "max_pairs": args.revisit_max_pairs}
        for case in cases: case["revisit_pairs"] = find_revisits(case["camera"][0], **options)
        return cases, bundle["rejected"], options
    from diffusion.data.datasets.video.sana_wm_zip_latent_data import SanaWMZipLatentDataset
    data = asdict(config.data)
    data.update(num_frames=(args.frames - 1) * config.data.vae_ratio[0] + 1,
                data_repeat=1, sort_dataset=True, shuffle_dataset=False)
    dataset = SanaWMZipLatentDataset(**data, resolution=config.data.image_size)
    revisit_options = {"min_gap": args.revisit_min_gap, "distance_fraction": args.revisit_distance_fraction,
                       "angle_deg": args.revisit_angle_deg, "max_pairs": args.revisit_max_pairs}
    cases, rejected = select_cases(dataset, args.frames, args.eval_cases, args.seed, revisit_options)
    if fixed:
        # Atomic exclusive publication also handles two queued evaluators starting together.
        import uuid
        path = Path(fixed).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        torch.save({"format": "TTN-fixed-cases-v1", "seed": args.seed, "data_identity": fixed_data_identity(config),
                    "cases": cases, "rejected": rejected}, temporary)
        try:
            os.link(temporary, path)
        except FileExistsError:
            return load_evaluation_cases(config, args)
        finally:
            temporary.unlink()
    return cases, rejected, revisit_options


def fixed_data_identity(config):
    data = asdict(config.data)
    # A longer training clip must not silently select a different diagnostic case.
    data.pop("num_frames", None)
    return {"data": data, "text_encoder": asdict(config.text_encoder), "vae_ratio": list(config.data.vae_ratio)}


def diagnostic_noise(shape, device, seed, horizon=None):
    horizon = horizon or shape[2]
    if horizon < shape[2]: raise ValueError("noise horizon is shorter than the diagnostic input")
    full_shape = (*shape[:2], horizon, *shape[3:])
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(full_shape, generator=generator, device=device)[:, :, :shape[2]].contiguous()


def evaluate_command(args):
    from .cli import rollout, timed_cuda, seed_everything, json_record, to_device
    from .sana import build_sana, configure_cross_attention
    from .checkpoint import load_checkpoint
    from diffusion.model.builder import get_tokenizer_and_text_encoder
    from train_video_scripts.train_sana_wm_stage1 import _encode_prompts
    from torch.utils.data import default_collate

    output, training_run = Path(args.output).resolve(), Path(args.training_run).resolve()
    if output.exists(): raise ValueError(f"use a new evaluation output directory: {output}")
    if os.environ.get("SANA_WM_STAGE1_KV_SAVE_STRIDE", "1") != "1":
        raise ValueError("paired TTN evaluation requires SANA_WM_STAGE1_KV_SAVE_STRIDE=1")
    if args.cached_blocks == 0 or args.cached_blocks < -1:
        raise ValueError("--cached-blocks must be -1 (unbounded) or positive")
    run, config, ttn, adapter, adapter_digest, last_train = load_evaluation_run(args)
    ablation = getattr(args, "ttn_ablation", "full")
    history = getattr(args, "history_source", "generated")
    methods = getattr(args, "eval_methods", ("sana", "ttn"))
    diagnostics = getattr(args, "state_diagnostics", False)
    sink = validate_inference_interventions(args, ttn)
    if ablation != "full" and ttn.stage != "C": raise ValueError("mechanism ablations require a Stage C checkpoint")
    if getattr(args, "ttn_compare_reference", False):
        if tuple(methods) != ("sana", "ttn") or ablation != "full" or diagnostics:
            raise ValueError("backend comparison requires paired Full evaluation without mechanism diagnostics")
        methods = ("sana", "ttn", "ttn_reference")
    if not methods or len(set(methods)) != len(methods) or any(m not in ("sana", "ttn", "ttn_reference") for m in methods):
        raise ValueError("evaluate requires distinct sana/ttn methods")
    from .provenance import camera_contract, implementation_identity
    camera_policy = camera_contract(ttn.camera_attention, getattr(args, "camera_attention", None),
                                    ablation=getattr(args, "camera_ablation", False))
    cases, rejected, revisit_options = load_evaluation_cases(config, args)
    fixed_digest = file_sha256(args.fixed_cases) if getattr(args, "fixed_cases", None) else None
    output.mkdir(parents=True)
    tokenizer, encoder = get_tokenizer_and_text_encoder(config.text_encoder.text_encoder_name, "cpu")
    encoder.eval().requires_grad_(False)
    # Match SANA's _ensure_null_embed: encode the empty prompt with the same
    # text encoder/template. Keep the existing smoke's zero-text default alone.
    null_y, _ = _encode_prompts([""], tokenizer, encoder, config, "cpu")
    null_y = null_y.cpu()
    for index, case in enumerate(cases):
        y, mask = _encode_prompts([case["prompt"]], tokenizer, encoder, config, "cpu")
        case["y"], case["mask"] = y.cpu(), mask.cpu()
        case["input_sha256"] = {name: tensor_sha256(value) for name, value in (
            ("initial_latent", case["reference"][:, :, :1]), ("camera", case["camera"]),
            ("y", y), ("mask", mask), ("uncondition", null_y))}
        if case["plucker"] is not None: case["input_sha256"]["plucker"] = tensor_sha256(case["plucker"])
        torch.save({"clean_latents": case["reference"], "camera_conditions": case["camera"],
                    "prompt": case["prompt"], "revisit_pairs": case["revisit_pairs"]}, output / f"case-{index:03d}-reference.pt")
    del tokenizer, encoder
    gc.collect()
    manifest = [{k: v for k, v in case.items() if k not in ("reference", "camera", "plucker", "y", "mask", "info")}
                for case in cases]
    protocol = {"scope": "training-example diagnostic; not a verified scene-held-out test",
                "metric_space": "cached LTX latents; no decoded PSNR/SSIM/LPIPS/VBench",
                "interpretation": "GT error includes sample ambiguity; pose candidates require visual validation",
                "training_run": str(training_run), "checkpoint": str(adapter), "checkpoint_sha256": adapter_digest,
                "stage": ttn.stage, "step": last_train["step"], "frames": args.frames, "steps": args.steps,
                "ttn_ablation": ablation, "history_source": history, "eval_methods": list(methods),
                "tla_sink": asdict(sink),
                "meta_ttt": {"local_update": ttn.local_update, "persistent_meta": ttn.persistent_meta},
                "state_diagnostics": diagnostics,
                "history_access": HISTORY_ACCESS[history],
                "train_scope": last_train.get("train_scope", run["arguments"].get("train_scope", "ttn")),
                "weight_scope": last_train.get("weight_scope", "ttn"),
                "provenance": implementation_identity(), "training_provenance": run.get("provenance"),
                "fixed_cases_sha256": fixed_digest,
                "noise_frames": getattr(args, "noise_frames", None) or args.frames,
                **camera_policy,
                "checkpoint_camera_attention": ttn.camera_attention,
                "ttn_camera_attention": getattr(args, "camera_attention", None) or ttn.camera_attention,
                "ttn_camera_backend": "flash_or_math" if (getattr(args, "camera_attention", None) or ttn.camera_attention) == "sana" else None,
                "cfg_scale": args.cfg_scale, "unconditional_text": "encoded empty prompt via SANA _encode_prompts",
                "cross_attn_backend": args.cross_attn_backend,
                "cached_blocks": args.cached_blocks, "kv_save_stride": 1, "refiner": None,
                "training_history_protocol": run.get("history_protocol", {"clean_commits": "GT", "camera_cache": "all previous chunks within clip"}),
                "rollout_history": f"{history} clean chunks; original SANA cache window; TTN S/psi persistent",
                "flow_shift": config.scheduler.inference_flow_shift, "revisit_options": revisit_options,
                "excursion_fraction_min": .1, "excursion_angle_deg_min": 15,
                "training_latent_frames": (run["training"]["data"]["num_frames"] - 1) // config.data.vae_ratio[0] + 1,
                "config": {name: asdict(getattr(config, name)) for name in ("model", "scheduler", "text_encoder", "data")},
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "timing_scope": "instrumented sampler including clean-commit telemetry/progress logs/cold kernels; excludes model/text load and final metrics; not a throughput benchmark",
                "launch": args.launch, "compile": {name: os.getenv(name) for name in ("GDN_DISABLE_COMPILE", "GDN_DISABLE_COMPLEX_COMPILE")}}
    (output / "manifest.json").write_text(json.dumps({"protocol": protocol, "cases": manifest, "rejected": rejected}, indent=2))
    records = []
    for method in methods:
        seed_everything(args.seed)
        kwargs = {"install_adapter": method != "sana"}
        if method != "sana" and (last_train.get("weight_scope") == "dit" or
                run["arguments"].get("train_scope") in ("dit", "ttn-visual")):
            kwargs["dtype"] = torch.float32  # retain trained masters; BF16 CUDA compute still uses autocast
        model = build_sana(config, ttn, args.base_weights or run["base"]["source"], args.device, **kwargs)
        if model.base_load_report["sha256"] != run["base"]["sha256"]:
            raise ValueError("paired evaluation base weights differ from training")
        if method != "sana":
            from .performance import configure_from_args, configure_execution, ExecutionOptions, precision_audit
            execution = configure_execution(model, ExecutionOptions()) if method == "ttn_reference" else configure_from_args(model, args)
            if method == "ttn": protocol.update(execution=execution, precision=precision_audit())
            load_checkpoint(adapter, model)
            if file_sha256(adapter) != adapter_digest: raise ValueError("checkpoint changed during evaluation")
            if getattr(args, "camera_attention", None) is not None:
                from .anchor import configure_camera_attention
                configure_camera_attention(model, args.camera_attention)
        policy = configure_cross_attention(model, args.cross_attn_backend)
        model.eval().requires_grad_(False)
        print("[TTN eval model] " + json.dumps({"method": method, "base": model.base_load_report, "cross_attention": policy}), flush=True)
        for index, case in enumerate(cases):
            seed_everything(case["seed"])
            gt = case["reference"]  # GT future remains on CPU and outside model kwargs
            batch = {"initial_latent": gt[:, :, :1].to(args.device), "y": case["y"].to(args.device),
                     "uncondition": null_y.to(args.device),
                     "mask": case["mask"].to(args.device), "camera_conditions": case["camera"].to(args.device),
                     "width": gt.shape[-1], "height": gt.shape[-2], "data_info": to_device(default_collate([case["info"]]), args.device)}
            if case["plucker"] is not None: batch["chunk_plucker"] = case["plucker"][None].to(args.device)
            noise = diagnostic_noise(gt.shape, args.device, case["seed"], getattr(args, "noise_frames", None))
            noise_hash = tensor_sha256(noise)
            def progress(chunk):
                if diagnostics:
                    display = {key: chunk[key] for key in ("chunk", "start", "end", "seconds", "state_norm", "psi_norm") if key in chunk}
                    print("[TTN mechanism chunk] " + json.dumps({"method": method, "case": index,
                          "ablation": ablation, "history": history, **display}), flush=True)
                else:
                    print("[TTN eval chunk] " + json.dumps({"method": method, "case": index, **chunk}), flush=True)
            def save_state(chunk, state):
                torch.save({"world_state": state.world_state.detach().cpu(),
                            "transition_fast": state.transition_fast.detach().cpu(),
                            "commits": state.commit_count, "predictions": state.predict_count},
                           output / f"case-{index:03d}-{method}-state-{chunk:03d}.pt")
            runtime_options = {"on_state": save_state} if getattr(args, "ttn_compare_reference", False) else {}
            if method == "ttn" and (ablation != "full" or diagnostics):
                runtime_options.update(ttn_ablation=ablation, state_diagnostics=diagnostics)
            if method == "ttn" and sink.active:
                runtime_options["sink_options"] = sink
            if history != "generated":
                runtime_options.update(history_reference=gt, history_source=history)
            (generated, runtime, chunks), timing = timed_cuda(lambda: rollout(
                model, config, batch, args.steps, args.cfg_scale, args.cached_blocks, initial_noise=noise,
                on_chunk=progress, **runtime_options))
            generated = generated.cpu()
            metric_options = {"training_frames": protocol["training_latent_frames"]} if diagnostics else {}
            metrics = latent_metrics(generated, gt, case["revisit_pairs"], **metric_options)
            row = {"case_id": case["case_id"], "seed": case["seed"], "method": method,
                   "input_sha256": case["input_sha256"], "initial_noise_sha256": noise_hash,
                   "mask_valid": int(batch["mask"].sum()), "mask_tokens": batch["mask"].numel(),
                   "metrics": metrics, "timing": timing, "chunks": chunks,
                   "base_sha256": model.base_load_report["sha256"],
                   "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
                   "commits": runtime.commit_count if runtime else None,
                   "predictions": runtime.predict_count if runtime else None,
                   "sink_reference_sha256": getattr(runtime, "sink_reference_sha256", None),
                   "sink_reference_verified": getattr(runtime, "sink_reference_verified", None)
                       if getattr(runtime, "sink_reference_sha256", None) is not None else None}
            if diagnostics and args.frames >= 13:
                row["prefix_13_metrics"] = latent_metrics(generated[:, :, :13], gt[:, :, :13],
                    [p for p in case["revisit_pairs"] if p["frame_b"] < 13], **metric_options)
                row["prefix_protocol"] = "causal first 13 frames of this rollout, same full-horizon noise; not a separate run"
            torch.save({"latents": generated, "method": method, "case_id": case["case_id"], "seed": case["seed"],
                        "chunks": [{k: chunk[k] for k in ("chunk", "start", "end")} for chunk in chunks]},
                       output / f"case-{index:03d}-{method}.pt")
            json_record(output / "episodes.jsonl", row)
            records.append(row)
            print("[TTN eval case] " + json.dumps({"method": method, "case": index, "final_chunk_latent_mse": metrics["final_chunk_latent_mse"],
                                                   "revisit_return_gt_latent_mse": metrics["revisit_return_gt_latent_mse"], **timing}), flush=True)
            del batch, noise, runtime, generated
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if "sana" in methods and "ttn" in methods:
        for index in range(len(cases)):
            a, b = records[index], records[index + len(cases)]
            if any(a[name] != b[name] for name in ("case_id", "seed", "input_sha256", "initial_noise_sha256", "base_sha256")):
                raise AssertionError("paired evaluation inputs disagree")
    if file_sha256(adapter) != adapter_digest: raise ValueError("checkpoint changed during evaluation")
    if fixed_digest is not None and file_sha256(args.fixed_cases) != fixed_digest:
        raise ValueError("fixed cases changed during evaluation")
    paired_records = [r for r in records if r["method"] != "ttn_reference"]
    result = {"protocol": protocol, **evaluation_summary(paired_records), "episodes": records}
    if getattr(args, "ttn_compare_reference", False):
        from .benchmark import compare_saved_rollouts
        result["backend_comparison"] = compare_saved_rollouts(output, len(cases))
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    print("[TTN eval summary] " + json.dumps({"output": str(output), **evaluation_summary(paired_records)}), flush=True)
