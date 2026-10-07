"""Inference-only clean-history interventions, after the current prediction."""
from copy import copy
from dataclasses import fields, is_dataclass, replace

import torch

from .core import CayleyFactors
from .performance import DEFAULT_EXECUTION
from .session import validate_clean_output


HISTORY_SOURCES = ("generated", "gt", "ttn-gt", "native-gt")
MIXED_SOURCES = ("ttn-gt", "native-gt")


def _validate_mixed_mode(system, ablation):
    if torch.is_grad_enabled():
        raise ValueError("mixed clean history is inference-only")
    cfg = system.config
    if cfg.stage != "C" or cfg.camera_attention != "sana" or ablation != "full":
        raise ValueError("mixed clean history requires Full Stage C with native SANA camera")
    execution = getattr(system, "ttn_execution", DEFAULT_EXECUTION)
    if execution.core_backend != "reference" or execution.psi_backend != "reference":
        raise ValueError("mixed clean history requires reference/reference")


def resolve_clean_history_source(source, provider, session):
    """A provider alone retains the previous all-GT diagnostic behavior."""
    source = ("gt" if provider is not None else "generated") if source is None else source
    if source not in HISTORY_SOURCES:
        raise ValueError("unknown clean history source")
    if source != "generated" and provider is None:
        raise ValueError("GT clean history requires a provider")
    if source in MIXED_SOURCES:
        system = getattr(getattr(session, "model", None), "ttn_system", None)
        if system is None:
            raise ValueError("mixed clean history requires a TTN session")
        _validate_mixed_mode(system, session.ablation)
    return source


def validate_history_reference(reference, generated):
    if (not isinstance(reference, torch.Tensor) or reference.shape != generated.shape
            or reference.device != generated.device or reference.dtype != generated.dtype
            or not torch.isfinite(reference).all()):
        raise ValueError("invalid diagnostic clean history chunk (shape/dtype/device/finite)")


def _clone_tree(value):
    """Clone fast tensor storage, never the shared model or its parameters."""
    if isinstance(value, torch.Tensor): return value.detach().clone()
    if isinstance(value, list): return [_clone_tree(item) for item in value]
    if isinstance(value, tuple): return tuple(_clone_tree(item) for item in value)
    if isinstance(value, dict): return {key: _clone_tree(item) for key, item in value.items()}
    if is_dataclass(value):
        return replace(value, **{field.name: _clone_tree(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, CayleyFactors):
        cloned = copy(value)
        cloned.p, cloned.l = _clone_tree(value.p), _clone_tree(value.l)
        return cloned
    return value


def _isolated_clean_context(context):
    clean = context.for_clean()
    return replace(clean, **{field.name: _clone_tree(getattr(clean, field.name))
                            for field in fields(clean) if field.name != "system"})


def _validate_candidates(context, runtime):
    if set(context.candidates) != set(range(5)):
        raise RuntimeError("all five anchors must finish each clean history pass")
    for i in range(5):
        state, gradient, _ = context.candidates[i]
        if (not isinstance(state, torch.Tensor) or not isinstance(gradient, torch.Tensor)
                or state.shape != runtime.world_state[:, i].shape
                or gradient.shape != runtime.transition_fast[:, i].shape
                or state.device != runtime.world_state.device or gradient.device != runtime.transition_fast.device
                or state.dtype != torch.float32 or gradient.dtype != torch.float32
                or not torch.isfinite(state).all() or not torch.isfinite(gradient).all()):
            raise ValueError("invalid candidate in clean history pass; transaction not committed")


def _validate_native_cache(updated, incoming):
    if not isinstance(updated, list) or len(updated) != len(incoming):
        raise ValueError("native clean history cache block count mismatch")
    for slots, previous in zip(updated, incoming):
        if not isinstance(slots, list) or len(slots) != len(previous):
            raise ValueError("native clean history cache slot count mismatch")
        for index, value in enumerate(slots):
            if index == 6 and isinstance(value, (int, float)) and value in (0., 1.): continue
            if value is not None and (not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()):
                raise ValueError("invalid native clean history cache; transaction not committed")


def _input_identity(value):
    # Diagnostic identity only: copy 32 deterministic elements instead of a full CUDA latent.
    from .evaluation import tensor_sha256
    flat = value.detach().reshape(-1)
    indices = torch.linspace(0, flat.numel()-1, min(32, flat.numel()), device=value.device).long()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sample_sha256": tensor_sha256(flat[indices]), "sample_elements": int(indices.numel()),
            "identity": "detached evenly spaced sample; not a full latent checksum"}


def clean_history_transaction(forward, generated, cache, *, source="generated", reference=None,
                              context=None, runtime=None):
    """Run clean calls and return the selected native cache after one atomic commit.

    ``forward(input, clean_context, native_cache)`` returns output/cache. Inputs
    already contain any CFG branches. Mixed calls start from the same incoming
    hybrid history; they are not donors from separate full-GT trajectories.
    """
    if source not in HISTORY_SOURCES: raise ValueError("unknown clean history source")
    if source != "generated": validate_history_reference(reference, generated)
    mixed = source in MIXED_SOURCES
    if mixed:
        if context is None or runtime is None:
            raise ValueError("mixed clean history requires a TTN context/runtime")
        _validate_mixed_mode(context.system, context.ablation)
        clean_calls = []
        for clean_input in (generated, reference):
            clean = _isolated_clean_context(context)
            output, updated = forward(clean_input.detach().clone(), clean, _clone_tree(cache))
            validate_clean_output(output, clean_input)
            _validate_candidates(clean, runtime)
            _validate_native_cache(updated, cache)
            clean_calls.append((output, updated, clean))
        state_index, native_index = (1, 0) if source == "ttn-gt" else (0, 1)
        _, _, clean = clean_calls[state_index]
        output, updated, _ = clean_calls[native_index]
    else:
        clean_input = generated if source == "generated" else reference
        clean = context.for_clean() if context is not None else None
        output, updated = forward(clean_input, clean, cache)
        if clean is not None: validate_clean_output(output, clean_input)

    if clean is not None:
        origins = {"ttn": "gt" if source in ("gt", "ttn-gt") else "generated",
                   "native": "gt" if source in ("gt", "native-gt") else "generated"}
        # Prepare diagnostic data before publishing; a diagnostic failure must not commit.
        identities = ({"generated": _input_identity(generated),
                       **({"gt": _input_identity(reference)} if reference is not None else {})}
                      if runtime.diagnostics else None)
        runtime.commit_chunk(clean)
        runtime.last_stats.update(history_source=source, history_origins=origins, clean_forward_passes=2 if mixed else 1)
        if identities is not None: runtime.last_stats["clean_input_identity"] = identities
    return output, updated
