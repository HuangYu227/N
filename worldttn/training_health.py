"""Check the actual optimizer membership and first training update, not probe gradients."""
import math
import torch
from .core import ANCHORS
from .anchor import trainable_parameter_names, is_ttn_parameter


def optimizer_parameter_names(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]


def audit_training_parameters(model, optimizer):
    stage = model.ttn_system.config.stage
    named = dict(model.named_parameters())
    required = {f"blocks.{i}.attn.{group}.weight" for i in ANCHORS
                for group in ("qkv", "beta_proj", "proj", "output_gate")}
    if stage != "A":
        required.update(("ttn_system.generators.u", "ttn_system.generators.v"))
        required.update(f"ttn_system.controller.heads.{i}.weight" for i in range(5))
    if required - named.keys(): raise ValueError(f"missing registered TTN parameters: {sorted(required - named.keys())}")
    expected = trainable_parameter_names(model)
    actual = {name for name, p in named.items() if p.requires_grad}
    if actual != expected:
        raise ValueError(f"TTN trainable parameter mismatch: missing={sorted(expected - actual)}, "
                         f"unexpected={sorted(actual - expected)}")
    optimized = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(optimized) != len(set(optimized)) or set(optimized) != {id(named[n]) for n in expected}:
        raise ValueError("optimizer must contain every TTN trainable parameter exactly once and no frozen parameter")
    wrong_dtype = [name for name in expected if named[name].dtype != torch.float32]
    if wrong_dtype: raise ValueError(f"TTN optimizer parameters must be FP32: {wrong_dtype}")
    return {"stage": stage, "train_scope": getattr(model, "ttn_train_scope", "ttn"),
            "weight_scope": getattr(model, "ttn_weight_scope", "ttn"),
            "trainable_numel": sum(named[n].numel() for n in expected),
            "frozen_numel": sum(p.numel() for name, p in named.items() if name not in expected),
            "optimizer_groups": [{"name": g.get("name", "ttn"), "lr": g["lr"],
                                  "numel": sum(p.numel() for p in g["params"])} for g in optimizer.param_groups],
            "optimizer_parameter_names": optimizer_parameter_names(model, optimizer),
            "trainable": [{"name": n, "shape": list(named[n].shape), "dtype": str(named[n].dtype)}
                          for n in sorted(expected)]}


def _local_cpu(value):
    if hasattr(value, "to_local"): value = value.to_local()
    return value.detach().cpu()


class FirstUpdateProbe:
    def __init__(self, model):
        self.parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.before = {n: _local_cpu(p).clone() for n, p in self.parameters.items()}

    def report(self, optimizer):
        groups, missing, missing_backbone, components = {}, [], [], {}
        backbone = {"parameters": 0, "parameters_with_grad": 0, "updated_parameters": 0,
                    "changed_elements": 0, "optimizer_state_parameters": 0}
        core = {f"blocks.{i}.attn.{group}.weight" for i in ANCHORS for group in ("qkv", "proj", "output_gate")}
        for name, p in self.parameters.items():
            group = ".".join(name.split(".")[:4] if name.startswith("blocks.") else name.split(".")[:2])
            row = groups.setdefault(group, {"initial_squared": 0., "delta_squared": 0., "grad_squared": 0.,
                                           "changed_elements": 0, "missing_gradients": 0,
                                           "optimizer_state_parameters": 0})
            initial, current = self.before[name].double(), _local_cpu(p).double()
            delta = current - initial
            changed = int(delta.count_nonzero())
            row["initial_squared"] += initial.square().sum().item()
            row["delta_squared"] += delta.square().sum().item()
            row["changed_elements"] += changed
            row["optimizer_state_parameters"] += bool(optimizer.state.get(p))
            if not is_ttn_parameter(name):
                part = name.split(".")[2] if name.startswith("blocks.") else ""
                kind = {"attn": "gdn", "mlp": "ffn", "ffn": "ffn", "cross_attn": "text_cross_attention"}.get(part, "other_dit")
                component = components.setdefault(kind, {key: 0 for key in backbone})
                for counts in (backbone, component):
                    counts["parameters"] += 1
                    counts["parameters_with_grad"] += p.grad is not None
                    counts["updated_parameters"] += changed > 0
                    counts["changed_elements"] += changed
                    counts["optimizer_state_parameters"] += bool(optimizer.state.get(p))
                if p.grad is None: missing_backbone.append(name)
            if p.grad is None:
                row["missing_gradients"] += 1
                if name in core: missing.append(name)
            else:
                grad = _local_cpu(p.grad).double()
                row["grad_squared"] += grad.square().sum().item()
        for row in groups.values():
            initial = math.sqrt(row.pop("initial_squared"))
            row["delta_norm"] = math.sqrt(row.pop("delta_squared"))
            row["grad_norm"] = math.sqrt(row.pop("grad_squared"))
            row["initial_norm"] = initial
            row["update_ratio"] = row["delta_norm"] / initial if initial else None
        self.before.clear()
        return {"scope": "actual first optimizer step of this invocation; gradients after clipping; rank-local shards for FSDP2",
                "groups": groups, "missing_core_gradients": missing,
                "backbone": {**backbone, "by_component": components}, "missing_backbone_gradients": missing_backbone}
