"""Offline adapter or full DiT checkpoints. Runtime S/psi are never serialized."""
import math
import os
import random
import tempfile
from pathlib import Path
import torch
from .anchor import is_ttn_parameter, offline_state_dict
from .core import BASE_REVISION
from .training_health import optimizer_parameter_names, audit_training_parameters


def make_optimizer(model, lr=1e-5, *, backbone_lr=1e-6):
    if not math.isfinite(lr) or lr <= 0: raise ValueError("learning rate must be finite and positive")
    parameters = [p for p in model.parameters() if p.requires_grad]
    options = {}
    if getattr(model, "ttn_train_scope", "ttn") == "dit":
        if not math.isfinite(backbone_lr) or backbone_lr <= 0:
            raise ValueError("backbone learning rate must be finite and positive")
        parameters = [{"name": name, "lr": rate,
                       "params": [p for n, p in model.named_parameters() if p.requires_grad and is_ttn_parameter(n) == ttn]}
                      for name, rate, ttn in (("ttn", lr, True), ("backbone", backbone_lr, False))]
        options["foreach"] = False  # Avoid a full-parameter-sized CUDA AdamW temporary.
    optimizer = torch.optim.AdamW(parameters, lr=lr,
                                 betas=(.9, .999),
                                 eps=1e-10,
                                 weight_decay=0., **options)
    audit_training_parameters(model, optimizer)
    return optimizer


def rng_state():
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available(): state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        import numpy as np
        np.random.set_state(state["numpy"])


def checkpoint_payload(model, adapter, optimizer, step):
    return {
        "format": "TTN-SANA-WM-v0.1",
        "base_revision": BASE_REVISION,
        "base_sha256": getattr(model, "base_load_report", {}).get("sha256"),
        "config": model.ttn_system.config.to_dict(),
        "stage": model.ttn_system.config.stage,
        "train_scope": getattr(model, "ttn_train_scope", "ttn"),
        "weight_scope": getattr(model, "ttn_weight_scope", "ttn"),
        "adapter": adapter,
        "optimizer": optimizer.state_dict() if optimizer else None,
        "optimizer_parameter_names": optimizer_parameter_names(model, optimizer) if optimizer else None,
        "step": step,
        "rng": rng_state()
    }


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(handle)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def save_checkpoint(path, model, optimizer, step):
    adapter = {k: v.detach().cpu() for k, v in offline_state_dict(model).items()}
    atomic_save(checkpoint_payload(model, adapter, optimizer, step), path)


def read_checkpoint(path, model, resume=False):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "TTN-SANA-WM-v0.1" or payload.get("base_revision") != BASE_REVISION:
        raise ValueError("not a compatible TTN reference checkpoint")
    if payload.get("base_sha256") != getattr(model, "base_load_report", {}).get("sha256"):
        raise ValueError("base weight SHA256 mismatch")
    current = model.ttn_system.config.to_dict()
    stored = dict(payload["config"])
    stage = stored.pop("stage")
    current_stage = current.pop("stage")
    if stored != current: raise ValueError("checkpoint architecture/config/base identity mismatch")
    if resume and stage != current_stage: raise ValueError("optimizer resume requires the same stage")
    if stage != current_stage and (stage, current_stage) not in (("A", "B"), ("B", "C")):
        raise ValueError("stage initialization must advance A->B or B->C")
    train_scope, weight_scope = payload.get("train_scope", "ttn"), payload.get("weight_scope", "ttn")
    if train_scope not in ("ttn", "dit") or weight_scope not in ("ttn", "dit"):
        raise ValueError("invalid checkpoint train-scope/weight scope")
    if train_scope == "dit" and weight_scope != "dit":
        raise ValueError("joint DiT checkpoint must include full weights")
    if resume and train_scope != getattr(model, "ttn_train_scope", "ttn"):
        raise ValueError("optimizer resume requires the same train-scope; use --adapter for initialization")
    expected = offline_state_dict(model, weight_scope)
    actual = payload["adapter"]
    if expected.keys() != actual.keys() or any(expected[k].shape != actual[k].shape for k in expected):
        raise ValueError("checkpoint adapter keys/shapes mismatch")
    return payload


def apply_checkpoint_weights(model, payload):
    """Apply a validated payload before parallel wrapping, retaining its export scope."""
    if payload.get("weight_scope", "ttn") == "dit":
        model.float()  # Preserve saved FP32 backbone weights even in a frozen inference model.
        model.ttn_weight_scope = "dit"
    model.load_state_dict(payload["adapter"], strict=False)


def load_checkpoint(path, model, optimizer=None, resume=False):
    payload = read_checkpoint(path, model, resume)
    # All validation happens before loading any parameter.
    if resume and (optimizer is None or payload.get("optimizer") is None):
        raise ValueError("resume requires optimizer state")
    if resume and payload.get("optimizer_parameter_names") is not None and (
            payload["optimizer_parameter_names"] != optimizer_parameter_names(model, optimizer)):
        raise ValueError("resume optimizer parameter names/order mismatch")
    if resume and getattr(model, "ttn_train_scope", "ttn") == "dit":
        stored_groups = payload["optimizer"]["param_groups"]
        if [g["lr"] for g in stored_groups] != [g["lr"] for g in optimizer.param_groups]:
            raise ValueError("resume learning rate mismatch")
    apply_checkpoint_weights(model, payload)
    if resume:
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng"])
        return int(payload["step"])
    return 0
