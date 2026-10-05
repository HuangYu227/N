"""Read-only stage measurements and gradient probes; no optimizer dependency."""
import hashlib
from contextlib import contextmanager

import torch

from .core import ANCHORS, innovation_loss
from .evaluation import tensor_sha256
from .training import activation_storage


def comparison_metrics(teacher, student, mask=None):
    if teacher.shape != student.shape: raise ValueError("alignment output shapes disagree")
    # FP32 vector norms lose accuracy on full-width SANA chunks (even cos(x,x)>1).
    # This is diagnostic accumulation only; model tensors and compute stay unchanged.
    a, b = teacher.detach().double(), student.detach().double()
    if mask is not None:
        if mask.shape != a.shape[:-1]: raise ValueError("alignment mask must select tokens")
        a, b = a[mask.bool()], b[mask.bool()]
    if not a.numel() or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("alignment requires nonempty finite outputs")
    a, b = a.reshape(-1), b.reshape(-1)
    norm_a, norm_b = a.norm().item(), b.norm().item()
    mse, energy = (a - b).square().mean().item(), a.square().mean().item()
    ratio = norm_b / norm_a if norm_a > 0 else None
    return {"mse": mse, "relative_l2": (mse / energy)**.5 if energy > 0 else None,
            "cosine": max(-1., min(1., (a * b).sum().item() / (norm_a * norm_b))) if norm_a * norm_b > 0 else None,
            "teacher_rms": energy**.5, "student_rms": b.square().mean().item()**.5,
            "rms_ratio": ratio, "norm_ratio": ratio, "elements": a.numel()}


def tensor_stats(value):
    value = value.detach().float().reshape(-1)
    if not value.numel(): return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    if not torch.isfinite(value).all(): raise FloatingPointError("nonfinite diagnostic tensor")
    return {"count": value.numel(), "mean": value.mean().item(), "std": value.std(unbiased=False).item(),
            "min": value.min().item(), "max": value.max().item()}


def stage_metrics(teacher, student, context, cfg):
    required = {"input", "projected_qkv", "visual_features", "visual_raw", "fused_raw", "gated_raw"}
    for label, values in (("SANA", teacher), ("TTN", student)):
        if required - values.keys(): raise RuntimeError(f"missing {label} diagnostic stages: {required - values.keys()}; update the checkout")
    n = teacher["input"].shape[1]
    read, _ = context.token_masks(n)
    mask = read & (context.frame_ids != 0).repeat_interleave(n // context.frame_ids.shape[1], -1)
    stages = {"input": comparison_metrics(teacher["input"], student["input"], read)}
    stages["input"]["exact_equal"] = torch.equal(teacher["input"][read], student["input"][read])
    stages["projected_qkv"], stages["transformed_qkv"] = {}, {}
    for name, a, b, qa, qb in zip("qkv", teacher["projected_qkv"], student["projected_qkv"],
                                  teacher["visual_features"], student["visual_features"]):
        stages["projected_qkv"][name] = comparison_metrics(a, b, mask)
        qa, qb = qa.transpose(1, 2).flatten(-2), qb.transpose(1, 2).flatten(-2)
        stages["transformed_qkv"][name] = {**comparison_metrics(qa, qb, mask),
                                           "teacher_norm": qa[mask].float().norm().item(),
                                           "student_norm": qb[mask].float().norm().item()}
    stages["transformed_qkv_scope"] = "post Norm/RoPE; TTN also has designed SiLU, so difference is expected"
    pred, new, k, v, beta, w, write = student["memory"]
    with torch.autocast(device_type=k.device.type, enabled=False):
        residual, after = v - k @ pred, v - k @ new
        active = write[:, None, :].expand_as(w)
        selected, target = residual[active], v[active]
        before_loss, after_loss = innovation_loss(pred, k, v, w, write), innovation_loss(new, k, v, w, write)
        eta = cfg.alpha_s / (cfg.eps + (w * k.square().sum(-1)).sum(-1))
        change = new - pred
        stages["memory"] = {"S_pred_norm": pred.norm().item(), "S_new_norm": new.norm().item(),
                            "correction_norm": change.norm().item(),
                            "correction_ratio": (change.norm() / (pred.norm() + cfg.eps)).item(),
                            "S_pred_norm_head": pred.norm(dim=(-2, -1)).tolist(),
                            "correction_ratio_head": (change.norm(dim=(-2, -1)) / (pred.norm(dim=(-2, -1)) + cfg.eps)).tolist()}
        stages["innovation"] = {"scope": "write tokens only", "rms": selected.square().mean().sqrt().item() if selected.numel() else None,
                                "relative": (selected.norm() / (target.norm() + cfg.eps)).item() if selected.numel() else None,
                                "after_rms": after[active].square().mean().sqrt().item() if selected.numel() else None,
                                "weighted_loss_before_head": before_loss.tolist(), "weighted_loss_after_head": after_loss.tolist(),
                                "weighted_loss_increased": bool((after_loss > before_loss * (1 + 1e-5) + 1e-8).any()),
                                "descent_tolerance": "rtol=1e-5, atol=1e-8; numerical invariant, not a quality threshold"}
        stages["write"] = {"scope": "beta/w stats over write tokens and all heads", "beta": tensor_stats(beta[active]),
                           "w": tensor_stats(w[active]), "eta_s": tensor_stats(eta), "eta_s_head": eta.tolist(),
                           "effective_eta_s_head": torch.where(write.any(-1)[:, None], eta, 0.).tolist(),
                           "write_token_count": int(write.sum()), "read_token_count": int(read.sum())}
    for name in ("visual_raw", "camera_raw", "camera_contribution", "fused_raw", "gated_raw"):
        if name.startswith("camera") and name not in teacher and name not in student:
            stages[name] = None
        else:
            if name not in teacher or name not in student: raise RuntimeError(f"unpaired diagnostic stage {name}")
            stages[name] = comparison_metrics(teacher[name], student[name], mask)
    if stages["camera_contribution"] is not None:
        visual_norm = teacher["visual_raw"][mask].float().norm().item()
        camera_delta = (student["camera_contribution"][mask].float() - teacher["camera_contribution"][mask].float()).norm().item()
        stages["camera_contribution"]["delta_relative_to_teacher_visual_norm"] = camera_delta / visual_norm if visual_norm > 0 else None
    stages["camera_scope"] = "camera_raw after per-head UCPE output transform/merge; contribution after out_proj_cam"
    if "local_update" in student: stages["local"] = student["local_update"]
    return stages


@contextmanager
def block_drift_trace(teacher, student, context):
    """Compare native block entries/exits, not replayed student activations."""
    stored, rows, seen, handles = {}, [], set(), []
    def select(value):
        n = value.shape[1]
        read, _ = context.token_masks(n)
        mask = read & (context.frame_ids != 0).repeat_interleave(n // context.frame_ids.shape[1], -1)
        return value.detach()[mask].to("cpu", copy=True)
    def record(index, point, is_teacher):
        def observe(module, args, output=None):
            value = args[0] if point == "before" else output[0] if isinstance(output, tuple) else output
            key = index, point
            if is_teacher:
                if key in stored: raise RuntimeError("teacher block executed twice in drift probe")
                stored[key] = select(value)
            else:
                if key not in stored or key in seen: raise RuntimeError("unpaired/duplicate native block trace")
                seen.add(key)
                rows.append({"block": index, "point": point, **comparison_metrics(stored.pop(key), select(value))})
        return observe
    try:
        for model, is_teacher in ((teacher, True), (student, False)):
            for index in ANCHORS:
                handles.append(model.blocks[index].register_forward_pre_hook(record(index, "before", is_teacher)))
                handles.append(model.blocks[index].register_forward_hook(record(index, "after", is_teacher)))
        yield rows
        if stored or len(seen) != 10: raise RuntimeError("drift trace did not observe all five native blocks")
    finally:
        for handle in handles: handle.remove()


def parameter_fingerprint(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(tensor_sha256(parameter.reshape(-1)).encode())
    return digest.hexdigest()


def parameter_changes(teacher, student):
    """Reference is actual TTN initialization: base projections and zero beta."""
    rows = []
    for index in ANCHORS:
        groups = {}
        for name in ("qkv", "beta_proj", "proj", "output_gate"):
            current = dict(getattr(student.blocks[index].attn, name).named_parameters())
            reference = dict(getattr(teacher.blocks[index].attn, name).named_parameters()) if name != "beta_proj" else {}
            delta2, initial2, current2 = 0., 0., 0.
            for key, parameter in current.items():
                value = parameter.detach().float()
                initial = reference[key].detach().float() if reference else torch.zeros_like(value)
                delta2 += (value - initial).square().sum().item()
                initial2 += initial.square().sum().item()
                current2 += value.square().sum().item()
            delta, initial_norm = delta2**.5, initial2**.5
            groups[name] = {"delta_norm": delta, "initial_norm": initial_norm, "current_norm": current2**.5,
                            "update_ratio": delta / (initial_norm + student.ttn_system.config.eps),
                            "relative_update_ratio": delta / initial_norm if initial_norm > 0 else None,
                            "zero_initial_norm": initial_norm == 0,
                            "reference": "TTN beta initialized to zero by contract" if name == "beta_proj" else "same base SANA weights, cast to FP32 at TTN installation"}
        rows.append({"block": index, "groups": groups})
    return rows


def gradient_health(session, context, clean, y, mask, info, noise, flow_loss, loss_mask, timestep):
    """autograd.grad never writes .grad; frozen backbone still propagates gradients."""
    model = session.model
    flags = [(p, p.requires_grad, p.grad) for p in model.parameters()]
    parameters, groups = [], []
    for index in ANCHORS:
        for name in ("qkv", "beta_proj", "proj", "output_gate"):
            group = list(getattr(model.blocks[index].attn, name).parameters())
            groups.append((index, name, len(parameters), len(group)))
            parameters.extend(group)
    try:
        for index in ANCHORS: model.blocks[index].attn.requires_grad_(True)
        t = torch.full((1, 1, 4), timestep, device=clean.device, dtype=torch.long)
        t[:, :, 0] = 0
        with torch.enable_grad(), torch.autocast(device_type=clean.device.type, dtype=torch.bfloat16,
                                                enabled=clean.device.type == "cuda"), activation_storage("cpu"):
            loss = flow_loss(session, clean, t, noise, y, context, [[None] * 10 for _ in model.blocks],
                             0, 4, mask, info, loss_mask).mean()
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite gradient-probe loss")
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        rows = {index: {"block": index, "groups": {}} for index in ANCHORS}
        for index, name, start, count in groups:
            tensors = gradients[start:start + count]
            finite = all(g is None or bool(torch.isfinite(g).all()) for g in tensors)
            if not finite: raise FloatingPointError(f"nonfinite anchor {index}/{name} diagnostic gradient")
            rows[index]["groups"][name] = {"grad_norm": sum(g.float().square().sum().item() for g in tensors if g is not None)**.5,
                                           "unused_parameters": sum(g is None for g in tensors), "finite": finite}
        return {"timestep": timestep, "loss": loss.item(), "anchors": list(rows.values()),
                "scope": "unclipped fixed GT one-chunk gradient, not historical training-step gradients",
                "activation_offload": "cpu", "optimizer_step": False, "param_grad_written": False}
    finally:
        for parameter, requires_grad, previous_grad in flags:
            parameter.requires_grad_(requires_grad)
            if parameter.grad is not previous_grad: raise AssertionError("gradient probe changed parameter.grad")


def diagnostic_summary(rows):
    """Locate measured departures; do not classify an architecture as failed."""
    probe = next((p for p in rows if p["timestep"] == 500), rows[len(rows) // 2])
    issues, candidates = [], []
    for p in rows:
        for anchor in p["anchors"]:
            block, stages = anchor["block"], anchor["stages"]
            if not stages["input"]["exact_equal"]: issues.append((0, block, p["timestep"], "input_mismatch"))
            if stages["innovation"]["weighted_loss_increased"]: issues.append((2, block, p["timestep"], "Correct_weighted_loss_increased"))
    priority = {"QKV_projection": 1, "visual_read": 3, "camera_fusion": 4, "final_output": 5}
    for anchor in probe["anchors"]:
        block, stages = anchor["block"], anchor["stages"]
        options = {"QKV_projection": max(v["relative_l2"] or 0 for v in stages["projected_qkv"].values()),
                   "visual_read": stages["visual_raw"]["relative_l2"],
                   "camera_fusion": (stages["camera_contribution"] or {}).get("delta_relative_to_teacher_visual_norm"),
                   "final_output": anchor["relative_l2"]}
        for stage, value in options.items():
            if value is not None: candidates.append((value, -priority[stage], -block, stage))
    score, _, negative_block, stage = max(candidates)
    drift = probe["representation_drift"]
    growth = [(drift[i + 1]["relative_l2"] - drift[i]["relative_l2"], drift[i]["block"])
              for i in range(0, len(drift), 2) if drift[i]["relative_l2"] is not None and drift[i + 1]["relative_l2"] is not None]
    largest_growth = max(growth) if growth else None
    first = min(issues) if issues else None
    text = (f"首个数值契约异常={'无' if first is None else f'anchor {first[1]}/{first[3]} (t={first[2]})'}；"
            f"最大可比局部差异=anchor {-negative_block}/{stage} (rel_l2={score:.4g})；"
            f"最大 block 漂移增量={'未定义' if largest_growth is None else f'block {largest_growth[1]} ({largest_growth[0]:+.4g})'}。不判定结构失败。")
    return {"timestep": probe["timestep"], "first_contract_violation": None if first is None else {"block": first[1], "stage": first[3], "timestep": first[2]},
            "largest_local_departure": {"block": -negative_block, "stage": stage, "relative_l2": score},
            "largest_block_drift_growth": None if largest_growth is None else {"block": largest_growth[1], "delta_relative_l2": largest_growth[0]},
            "selection": "exact input equality and Correct descent first; otherwise largest comparable rel_l2, priority breaks ties; no quality threshold",
            "camera_score": "post out_proj_cam delta / teacher visual norm; raw camera mismatch alone may be suppressed by zero output projection",
            "text": text}


def print_concise_report(result):
    probe = next((p for p in result["probes"] if p["timestep"] == result["diagnostic_summary"]["timestep"]), result["probes"][0])
    fmt = lambda x: "null" if x is None else f"{x:.4g}"
    print(f"[TTN stages] timestep={probe['timestep']}；全量指标保存在 summary.json", flush=True)
    for row in probe["anchors"]:
        s = row["stages"]
        print(f"anchor {row['block']}: input={fmt(s['input']['relative_l2'])}; correction={fmt(s['memory']['correction_ratio'])}; "
              f"innovation={fmt(s['innovation']['relative'])}; visual={fmt(s['visual_raw']['relative_l2'])}; "
              f"camera={fmt((s['camera_raw'] or {}).get('relative_l2'))}; final={fmt(row['relative_l2'])}", flush=True)
    for row in probe["representation_drift"]:
        print(f"block {row['block']} {row['point']}: rel_l2={fmt(row['relative_l2'])}, cos={fmt(row['cosine'])}, norm_ratio={fmt(row['norm_ratio'])}", flush=True)
    if result["gradient_health"] is not None:
        changes = {row["block"]: row["groups"] for row in result["parameter_changes"]}
        for row in result["gradient_health"]["anchors"]:
            g, delta = row["groups"], changes[row["block"]]
            print(f"anchor {row['block']} grad: " + ", ".join(f"{name}_grad_norm={fmt(g[name]['grad_norm'])}" for name in g) +
                  f"; qkv_update_ratio={fmt(delta['qkv']['update_ratio'])}; beta_update_ratio_eps={fmt(delta['beta_proj']['update_ratio'])}; "
                  f"beta_delta_norm={fmt(delta['beta_proj']['delta_norm'])} (beta initial=0, eps ratio is not a percent)", flush=True)
    print("[TTN diagnosis] " + result["diagnostic_summary"]["text"], flush=True)
