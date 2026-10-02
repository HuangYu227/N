"""Adapter-only offline checkpoints. Runtime S/psi are never serialized."""
import os
import random
import tempfile
from pathlib import Path
import torch
from .anchor import adapter_state_dict
from .core import BASE_REVISION


def make_optimizer(model, lr=1e-5):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=lr,
                             betas=(.9, .999),
                             eps=1e-10,
                             weight_decay=0.)


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
        "adapter": adapter,
        "optimizer": optimizer.state_dict() if optimizer else None,
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
    adapter = {k: v.detach().cpu() for k, v in adapter_state_dict(model).items()}
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
    expected = adapter_state_dict(model)
    actual = payload["adapter"]
    if expected.keys() != actual.keys() or any(expected[k].shape != actual[k].shape for k in expected):
        raise ValueError("checkpoint adapter keys/shapes mismatch")
    return payload


def load_checkpoint(path, model, optimizer=None, resume=False):
    payload = read_checkpoint(path, model, resume)
    actual = payload["adapter"]
    # All validation happens before loading any parameter.
    if resume and (optimizer is None or payload.get("optimizer") is None):
        raise ValueError("resume requires optimizer state")
    model.load_state_dict(actual, strict=False)
    if resume:
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng"])
        return int(payload["step"])
    return 0
