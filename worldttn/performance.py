"""Execution policy and diagnostics, deliberately outside model/checkpoint schema."""
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import os
import socket
import sys
import torch


@dataclass(frozen=True)
class ExecutionOptions:
    core_backend: str = "reference"
    psi_backend: str = "reference"
    profile: bool = False
    layout_audit: str | None = None

    def __post_init__(self):
        if self.core_backend not in ("reference", "reuse", "compiled"):
            raise ValueError("unknown TTN core backend")
        if self.psi_backend == "triton":
            raise ValueError("Triton not implemented: target-GPU profiling stop/go gate must pass first")
        if self.psi_backend not in ("reference", "projected"):
            raise ValueError("unknown TTN psi backend")
        if self.core_backend == "reference" and self.psi_backend != "reference":
            raise ValueError("projected psi requires reuse or compiled core")


DEFAULT_EXECUTION = ExecutionOptions()


def validate_execution(options, stage, ablation="full", diagnostics=False, *, meta=False):
    if meta and (options.core_backend != "reference" or options.psi_backend != "reference"):
        raise ValueError("Meta-TTT requires reference/reference; detached optimized backends are incompatible")
    if options.core_backend != "reference" and (stage != "C" or ablation != "full" or diagnostics):
        raise ValueError("optimized TTN execution requires Full Stage C without mechanism diagnostics; use reference/reference")


def execution_report(model):
    options = getattr(getattr(model, "ttn_system", None), "ttn_execution", DEFAULT_EXECUTION)
    implementation = ("original_dense" if options.core_backend == "reference" else
                      "dense_reused" if options.psi_backend == "reference" else "projected")
    cfg = getattr(getattr(model, "ttn_system", None), "config", None)
    if cfg is not None and (cfg.local_update or cfg.persistent_meta): implementation = "live_projected_meta"
    if cfg is not None and cfg.memory_update == "proximal": implementation = "disabled_proximal_s"
    return {**asdict(options), "psi_implementation": implementation, "triton_available": False,
            "activation_checkpointing": getattr(model, "ttn_activation_checkpointing", "none"),
            "activation_offload": getattr(model, "ttn_activation_offload", "none"),
            "activation_gpu_budget_gib": getattr(model, "ttn_activation_gpu_budget_gib", 0.0),
            "state_update": "live_proximal_s" if cfg is not None and cfg.memory_update == "proximal" else "delta"}


def configure_execution(model, options=None):
    options = options or DEFAULT_EXECUTION
    cfg = model.ttn_system.config
    validate_execution(options, cfg.stage, meta=cfg.local_update or cfg.persistent_meta or cfg.memory_update == "proximal")
    model.ttn_system.ttn_execution = options
    for block in model.blocks:
        if hasattr(block.attn, "beta_proj") and hasattr(block.attn, "index"):
            block.attn.ttn_execution = options
            block.attn._ttn_layouts_seen = set()
    return execution_report(model)


def configure_from_args(model, args):
    path = None
    if getattr(args, "ttn_layout_audit", False):
        rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))
        path = str(Path(args.output) / f"layouts-rank{rank}.jsonl")
    options = ExecutionOptions(getattr(args, "ttn_core_backend", "reference"),
        getattr(args, "ttn_psi_backend", "reference"), getattr(args, "ttn_profile", False), path)
    # The untouched reference entrypoint needs no execution attributes at all.
    if options == DEFAULT_EXECUTION: return execution_report(model)
    diagnostic = (getattr(args, "state_diagnostics", False) or getattr(args, "command", "") in
                  ("mechanism-evaluate", "align-chunk", "stage-evaluate") or
                  getattr(args, "history_source", "generated") != "generated")
    cfg = model.ttn_system.config
    validate_execution(options, cfg.stage, getattr(args, "ttn_ablation", "full"), diagnostic,
                       meta=cfg.local_update or cfg.persistent_meta or cfg.memory_update == "proximal")
    return configure_execution(model, options)


def annotation(options, name):
    return torch.profiler.record_function("ttn/" + name) if options.profile else nullcontext()


def tensor_signature(value):
    return {"shape": list(value.shape), "stride": list(value.stride()), "storage_offset": value.storage_offset(),
            "dtype": str(value.dtype), "device": str(value.device), "requires_grad": value.requires_grad,
            "contiguous": value.is_contiguous()}


def audit_layout(anchor, mode, **tensors):
    options = getattr(anchor, "ttn_execution", DEFAULT_EXECUTION)
    if not options.layout_audit: return
    record = {"anchor": anchor.index, "mode": mode, "grad_enabled": torch.is_grad_enabled(),
              "inputs": {name: tensor_signature(value) for name, value in tensors.items()}}
    key = json.dumps(record, sort_keys=True)
    if key in anchor._ttn_layouts_seen: return
    anchor._ttn_layouts_seen.add(key)
    path = Path(options.layout_audit)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle: handle.write(key + "\n")


def precision_audit():
    """Call after complete imports/model build; never change production precision."""
    result = {"python": sys.executable, "torch": torch.__version__, "cuda": torch.version.cuda,
              "hostname": socket.gethostname(), "module": __file__,
              "matmul_precision": torch.get_float32_matmul_precision(),
              "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "scope": "Torch flags only; original GDN handwritten kernel precision is independent",
              "environment": {name: os.environ.get(name) for name in (
                  "GDN_DISABLE_COMPILE", "GDN_DISABLE_COMPLEX_COMPILE", "TORCH_LOGS",
                  "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_LAUNCH_BLOCKING")}}
    try:
        from importlib.metadata import version, PackageNotFoundError
        result["triton"] = version("triton")
    except PackageNotFoundError:
        result["triton"] = None
    return result
