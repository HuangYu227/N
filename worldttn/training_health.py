"""Check the actual optimizer membership and first training update, not probe gradients."""
import math
import torch
from .core import ANCHORS
from .anchor import trainable_parameter_names, is_ttn_parameter, is_ttn_new_parameter, is_ttn_camera_parameter


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
    if model.ttn_system.config.state_ttt_rank:
        required.update(f'blocks.{i}.attn.state_ttt.{p}' for i in ANCHORS for p in ('raw_basis','step_logits'))
    elif model.ttn_system.config.local_update: required.add("ttn_system.local_eta_logits")
    if model.ttn_system.config.ttt_backend == 'inplace-ffn':
        required.update(f'blocks.{i}.mlp.inplace_ttt.{p}.weight'
                        for i in model.ttn_system.config.inplace_layers for p in ('ttt_conv', 'ttt_proj'))
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
    policy = getattr(model, "ttn_optimizer_policy", "legacy")
    if policy == "origin" and getattr(model, "ttn_train_scope", "ttn") == "dit":
        for group, names in zip(optimizer.param_groups, optimizer_parameter_names(model, optimizer)):
            if group.get("name") == "local_eta":
                if names != ["ttn_system.local_eta_logits"]: raise ValueError("invalid Local eta optimizer group")
                continue
            expected_origin = {"ttn_new": True, "sana_inherited": False}.get(group.get("name"))
            if expected_origin is None or any(is_ttn_new_parameter(n) != expected_origin for n in names):
                raise ValueError("optimizer parameter origin mismatch")
    return {"stage": stage, "train_scope": getattr(model, "ttn_train_scope", "ttn"),
            "optimizer_policy": policy,
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


def local_eta_snapshot(model):
    parameter = getattr(model.ttn_system, "local_eta_logits", None)
    return None if parameter is None else _local_cpu(parameter).clone()


def local_eta_update(model, before, optimizer):
    """Five actual slow parameter updates; never infer shrinkage from output ratios."""
    if before is None: return None
    parameter = model.ttn_system.local_eta_logits
    after = _local_cpu(parameter)
    rate = next(g["lr"] for g in optimizer.param_groups if any(p is parameter for p in g["params"]))
    return {"anchors": list(ANCHORS), "logits_before": before.tolist(), "logits_after": after.tolist(),
            "eta_before": torch.nn.functional.softplus(before).tolist(),
            "eta_after": torch.nn.functional.softplus(after).tolist(), "learning_rate": rate,
            "logit_gradient_after_clip": None if parameter.grad is None else _local_cpu(parameter.grad).tolist(),
            "measurement": "actual optimizer update; gradient after outer clipping; rank-local for sharded tensors"}


class FirstUpdateProbe:
    def __init__(self, model):
        self.parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.before = {n: _local_cpu(p).clone() for n, p in self.parameters.items()}

    def report(self, optimizer):
        groups, missing, missing_backbone, components, origins = {}, [], [], {}, {}
        backbone = {"parameters": 0, "parameters_with_grad": 0, "updated_parameters": 0,
                    "changed_elements": 0, "optimizer_state_parameters": 0}
        core = {f"blocks.{i}.attn.{group}.weight" for i in ANCHORS for group in ("qkv", "proj", "output_gate")}
        state_parameters = {name for name in self.parameters if name.endswith(('.state_ttt.raw_basis', '.state_ttt.step_logits'))
                            or '.mlp.inplace_ttt.' in name}
        core.update(state_parameters)
        state_signals = {}
        for name, p in self.parameters.items():
            group = ".".join(name.split(".")[:4] if name.startswith("blocks.") else name.split(".")[:2])
            row = groups.setdefault(group, {"initial_squared": 0., "delta_squared": 0., "grad_squared": 0.,
                                           "changed_elements": 0, "missing_gradients": 0,
                                           "optimizer_state_parameters": 0})
            initial, current = self.before[name].double(), _local_cpu(p).double()
            delta = current - initial
            origin = "ttn_new" if is_ttn_new_parameter(name) else "sana_inherited"
            total = origins.setdefault(origin, {"grad_squared": 0., "delta_squared": 0., "parameters": 0})
            total["parameters"] += 1
            total["delta_squared"] += delta.square().sum().item()
            changed = int(delta.count_nonzero())
            if name in state_parameters:
                state_signals[name] = dict(delta_norm=float(delta.norm()), changed_elements=changed,
                    grad_norm=None, local_shape=list(current.shape), optimizer_state=bool(optimizer.state.get(p)))
            row["initial_squared"] += initial.square().sum().item()
            row["delta_squared"] += delta.square().sum().item()
            row["changed_elements"] += changed
            row["optimizer_state_parameters"] += bool(optimizer.state.get(p))
            if not (is_ttn_new_parameter(name) if getattr(optimizer, "ttn_optimizer_policy", "legacy") == "origin"
                    else is_ttn_parameter(name)):
                part = name.split(".")[2] if name.startswith("blocks.") else ""
                kind = {"attn": "gdn", "mlp": "ffn", "ffn": "ffn", "cross_attn": "text_cross_attention"}.get(part, "other_dit")
                if is_ttn_parameter(name):
                    kind = "anchor_camera" if is_ttn_camera_parameter(name) else "anchor_visual"
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
                total["grad_squared"] += grad.square().sum().item()
                if name in state_parameters: state_signals[name]['grad_norm'] = float(grad.norm())
        for total in origins.values():
            total["grad_norm"] = math.sqrt(total.pop("grad_squared"))
            total["delta_norm"] = math.sqrt(total.pop("delta_squared"))
        for row in groups.values():
            initial = math.sqrt(row.pop("initial_squared"))
            row["delta_norm"] = math.sqrt(row.pop("delta_squared"))
            row["grad_norm"] = math.sqrt(row.pop("grad_squared"))
            row["initial_norm"] = initial
            row["update_ratio"] = row["delta_norm"] / initial if initial else None
        self.before.clear()
        return {"scope": "actual optimizer update; gradients after clipping; rank-local shards for FSDP2",
                "groups": groups, "by_origin": origins, "missing_core_gradients": missing,
                "state_ttt_parameters": state_signals,
                "backbone": {**backbone, "by_component": components}, "missing_backbone_gradients": missing_backbone}
