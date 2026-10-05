"""Detached clean-commit measurements; never part of the training objective."""
import torch


@torch.no_grad()
def state_dynamics(previous, predicted, candidate, eps):
    """Evaluation-only, per [CFG branch, head]. Spectral concentration is not failure."""
    previous, predicted, candidate = (t.detach().float() for t in (previous, predicted, candidate))
    def ratio(a, b):
        valid = b > eps
        values = (a / b.clamp_min(eps)).cpu().tolist()
        return [[x if valid[i, j] else None for j, x in enumerate(row)] for i, row in enumerate(values)]
    def norm(t):
        return t.flatten(-2).norm(dim=-1)
    def cosine(a, b):
        return ratio((a * b).sum((-1, -2)), norm(a) * norm(b))
    spectra = {}
    for name, state in (("previous", previous), ("predicted", predicted), ("committed", candidate),
                        ("correction", candidate - predicted)):
        singular = torch.linalg.svdvals(state)
        spectra[name] = {"sigma_max": singular[..., 0].cpu().tolist(),
                         "top4": singular[..., :4].cpu().tolist(),
                         "top1_energy_fraction": ratio(singular[..., 0].square(), state.square().sum((-1, -2))),
                         "stable_rank": ratio(state.square().sum((-1, -2)), singular[..., 0].square())}
    correction = candidate - predicted
    return {"state_spectrum": spectra, "state_dynamics": {
        "transport_norm": norm(predicted - previous).cpu().tolist(),
        "transport_relative": ratio(norm(predicted - previous), norm(previous)),
        "previous_prediction_cosine": cosine(previous, predicted),
        "correction_prediction_cosine": cosine(correction, predicted),
        "correction_relative": ratio(norm(correction), norm(predicted)),
        "measurement": "detached FP32; [CFG branch, head]; zero-reference ratios are null"}}


@torch.no_grad()
def write_direction_stats(correction, history, eps):
    """History pairs are (original write, write transported into current basis)."""
    correction = correction.detach().float()
    norm = correction.flatten(-2).norm(dim=-1)
    def cosine(old):
        old = old.detach().float()
        old_norm = old.flatten(-2).norm(dim=-1)
        values = ((correction * old).sum((-1, -2)) / (norm * old_norm).clamp_min(eps**2)).clamp(-1, 1).cpu().tolist()
        valid = ((norm > eps) & (old_norm > eps)).cpu().tolist()
        return [[x if valid[i][j] else None for j, x in enumerate(row)] for i, row in enumerate(values)]
    return {"write_direction": {"lag_" + str(lag): {
        "cosine_raw": cosine(history[lag - 1][0]),
        "cosine_transport_aligned": cosine(history[lag - 1][1])} for lag in (1, 2, 4) if lag <= len(history)},
        "write_direction_measurement": "detached FP32; [branch, head]; clean predicted chunks only; zero writes null; diagnostic only"}


def matrix_scale(value):
    value = value.detach().float()
    return {"norm": value.norm().item(), "rms": value.square().mean().sqrt().item(),
            "abs_max": value.abs().max().item()}


def distribution(value):
    value = value.detach().float().flatten()
    if not value.numel():
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {"count": value.numel(), "mean": value.mean().item(),
            "std": value.std(unbiased=False).item(), "min": value.min().item(), "max": value.max().item()}


@torch.no_grad()
def local_anchor_stats(previous, before, adapted, candidate, k, v, w, support, query,
                       gradient, clipped, scale, local, eta, eps):
    """Detached evidence: query after Correct also consumes its own observations."""
    predictions = [k.detach() @ s.detach() for s in (before, adapted, candidate)]
    def partition(mask):
        count = mask.sum(-1)[:, None]
        selected = mask[:, None, :, None]
        target = torch.where(selected, v.detach(), 0.)
        weights = torch.where(mask[:, None], w.detach(), 0.)
        energy = (weights * target.square().sum(-1)).sum(-1)
        result = {"tokens_per_branch": mask.sum(-1).cpu().tolist()}
        for label, prediction in zip(("before", "after_transport", "after_correct"), predictions):
            residual = torch.where(selected, prediction - v.detach(), 0.)
            loss = (weights * residual.square().sum(-1)).sum(-1)/(eps + energy)
            rms = (residual.square().sum((-1, -2))/(count * v.shape[-1]).clamp_min(1)).sqrt()
            result["loss_" + label] = [[x if count[b, 0] else None for x in row]
                                     for b, row in enumerate(loss.cpu().tolist())]
            result["residual_rms_" + label] = [[x if count[b, 0] else None for x in row]
                                             for b, row in enumerate(rms.cpu().tolist())]
        return result
    change = (adapted.detach() - before.detach()).flatten(-2).norm(dim=-1)
    reference = before.detach().flatten(-2).norm(dim=-1)
    relative = (change/reference.clamp_min(eps)).cpu().tolist()
    return {"objective": "weighted_v_energy_support", "eta": float(eta.detach()) if eta is not None else None,
            "support": partition(support), "query": partition(query),
            "state_old": matrix_scale(previous), "state_before_local": matrix_scale(before),
            "state_after_local": matrix_scale(adapted), "state_after_correct_temporary": matrix_scale(candidate),
            "per_head": {"raw_grad_norm": gradient.detach().norm(dim=-1).cpu().tolist(),
                         "clipped_grad_norm": clipped.detach().norm(dim=-1).cpu().tolist(),
                         "clip_scale": scale.detach().squeeze(-1).cpu().tolist(),
                         "psi_norm": local.detach().norm(dim=-1).cpu().tolist(),
                         "tanh_saturation_fraction": (local.detach().tanh().abs() > .99).float().mean(-1).cpu().tolist(),
                         "transport_change_norm": change.cpu().tolist(),
                         "transport_change_relative": [[x if reference[b, h] > eps else None for h, x in enumerate(row)]
                                                       for b, row in enumerate(relative)]},
            "clip_fraction": float((scale.detach() < 1).float().mean()),
            "measurement": "current noisy features; pre-Correct query held out of inner loss only; detached FP32"}


@torch.no_grad()
def clean_anchor_stats(previous, predicted, candidate, q, k, v, beta, w, read, write, gradient, cfg):
    """Token scales use explicit masks. Head arrays are [batch, head].

    Raw innovation is measured BEFORE Correct/Adapt; its normalized version
    divides by the same weighted V energy, so different layer scales are visible.
    Undefined ratios (zero reference/empty writes) are null, never huge eps ratios.
    """
    previous, predicted, candidate = (s.detach().float() for s in (previous, predicted, candidate))
    q, k, v, beta, w = (t.detach().float() for t in (q, k, v, beta, w))
    prediction = k @ predicted
    residual = v - prediction
    count = write.sum(-1)[:, None].expand(-1, k.shape[1])
    selected = write[:, None, :, None]
    energy = lambda t: (t.square() * selected).sum((-1, -2))
    residual2, v2, k2 = energy(residual), energy(v), energy(k)
    weighted_residual2 = (w * residual.square().sum(-1)).sum(-1)
    weighted_v2 = (w * v.square().sum(-1)).sum(-1)
    loss = weighted_residual2 / (2 * k.shape[-1] * count.clamp_min(1))
    correction = candidate - predicted
    pred_norm = predicted.flatten(-2).norm(dim=-1)
    head = {
        "innovation_loss": loss, "innovation_rms": (residual2 / (count * k.shape[-1]).clamp_min(1)).sqrt(),
        "k_rms": (k2 / (count * k.shape[-1]).clamp_min(1)).sqrt(),
        "v_rms": (v2 / (count * v.shape[-1]).clamp_min(1)).sqrt(),
        "kv_prediction_rms": (energy(prediction) / (count * v.shape[-1]).clamp_min(1)).sqrt(),
        "q_rms": ((q.square() * read[:, None, :, None]).sum((-1, -2)) /
                  (read.sum(-1)[:, None] * q.shape[-1]).clamp_min(1)).sqrt(),
        "state_pred_norm": pred_norm, "state_norm": candidate.flatten(-2).norm(dim=-1),
        "state_rms": candidate.square().mean((-1, -2)).sqrt(),
        "correction_norm": correction.flatten(-2).norm(dim=-1),
        "eta_s": torch.where(count > 0, cfg.alpha_s / (cfg.eps + (w * k.square().sum(-1)).sum(-1)), 0.),
        "inner_grad_norm": gradient.detach().norm(dim=-1),
    }
    def ratio(a, b, valid):
        out = (a / b.clamp_min(cfg.eps)).cpu().tolist()
        valid = valid.cpu().tolist()
        return [[x if valid[i][j] else None for j, x in enumerate(row)] for i, row in enumerate(out)]
    heads = {name: value.cpu().tolist() for name, value in head.items()}
    heads["innovation_relative"] = ratio(residual2.sqrt(), v2.sqrt(), (count > 0) & (v2 > cfg.eps**2))
    heads["innovation_relative_weighted"] = ratio(weighted_residual2.sqrt(), weighted_v2.sqrt(),
                                                   (count > 0) & (weighted_v2 > cfg.eps**2))
    heads["correction_ratio"] = ratio(head["correction_norm"], pred_norm, pred_norm > cfg.eps)
    selected_beta = beta.masked_select(write[:, None, :].expand_as(beta))
    selected_w = w.masked_select(write[:, None, :].expand_as(w))
    pred, old = matrix_scale(predicted), matrix_scale(previous)
    return {"write_tokens": int(write.sum()), "read_tokens": int(read.sum()),
            "write_tokens_per_branch": write.sum(-1).cpu().tolist(),
            "innovation_loss": loss.mean().item(), "inner_grad_norm": gradient.detach().norm().item(),
            "state_old": old, "state_pred": pred, "state": matrix_scale(candidate), "correction": matrix_scale(correction),
            "predict_norm_relative_change": (pred["norm"] - old["norm"]) / old["norm"] if old["norm"] > cfg.eps else None,
            "beta": distribution(selected_beta), "w": distribution(selected_w),
            "per_head": heads, "measurement": "clean/pre-Correct innovation; masked new write tokens; FP32; detached"}


@torch.no_grad()
def committed_anchor_stats(stats, old_psi, new_psi, grad, scale, cbase, block, prefill):
    out = dict(stats, block=block, prefill=prefill)
    out.update(psi=matrix_scale(new_psi), psi_update=matrix_scale(new_psi - old_psi),
               controller_coefficients=matrix_scale(cbase))
    out["per_head"] = dict(stats.get("per_head", {}),
                           inner_grad_norm=grad.norm(dim=-1).cpu().tolist(),
                           inner_grad_clipped_norm=(grad * scale).norm(dim=-1).cpu().tolist(),
                           inner_clip_scale=scale.squeeze(-1).cpu().tolist(),
                           psi_norm=new_psi.norm(dim=-1).cpu().tolist(),
                           psi_update_norm=(new_psi - old_psi).norm(dim=-1).cpu().tolist())
    out["inner_clipped_fraction"] = (scale < 1).float().mean().item()
    return out


def stability_rows(record):
    """Flatten rank-0's gathered step record without new distributed collectives."""
    for rank in record["ranks"]:
        chunks = ([rank["prefill"]] if rank.get("prefill") else []) + rank["chunks"]
        for chunk in chunks:
            for anchor in chunk["anchors"]:
                yield {"step": record["step"], "stage": record["stage"], "train_scope": record["train_scope"],
                       "implementation_id": record.get("implementation_id"),
                       "rank": rank["rank"], "chunk": chunk["chunk"], "start": chunk["start"], "end": chunk["end"],
                       "camera_attention": record["config"]["camera_attention"], **anchor}


def progress_line(record, target):
    # Prefill starts from S=0, so innovation/V=1 by construction; do not hide later chunks.
    rows = [row for row in stability_rows(record) if not row.get("prefill", False)]
    values = [(max(x for branch in r.get("per_head", {}).get("innovation_relative", [])
                   for x in branch if x is not None), r["block"]) for r in rows
              if any(x is not None for b in r.get("per_head", {}).get("innovation_relative", []) for x in b)]
    largest = max(values) if values else None
    peak = max(r.get("peak_allocated_bytes", 0) for r in record["ranks"]) / 2**30
    suffix = "" if largest is None else f" | max clean innovation/V={largest[0]:.3g}@{largest[1]}"
    return (f"[TTN progress] step {record['step']}/{target} | scope={record['train_scope']} | "
            f"loss={record['loss']:.6f} | grad={record['outer_grad_norm']:.4g} | "
            f"train={record['seconds']:.1f}s | peak={peak:.2f}GiB{suffix}")


@torch.no_grad()
def anchor_gradient_scales(model):
    """After outer clipping. FSDP values are LOCAL shards, explicitly labelled."""
    from .core import ANCHORS
    groups = {}
    for name, p in model.named_parameters():
        if not any(name.startswith(f"blocks.{i}.attn.") for i in ANCHORS) and not name.startswith("ttn_system."):
            continue
        group = name.rsplit(".", 1)[0]
        row = groups.setdefault(group, {"grad_squared": 0., "parameter_squared": 0., "missing_gradients": 0,
                                        "trainable_parameters": 0, "sharded": False})
        value = p.detach()
        if hasattr(value, "to_local"):
            value = value.to_local()
            row["sharded"] = True
        row["parameter_squared"] += value.float().square().sum().item()
        if p.requires_grad:
            row["trainable_parameters"] += 1
            if p.grad is None: row["missing_gradients"] += 1
            else:
                g = p.grad.detach()
                if hasattr(g, "to_local"): g = g.to_local()
                row["grad_squared"] += g.float().square().sum().item()
    for row in groups.values():
        row["grad_norm"] = row.pop("grad_squared")**.5
        row["parameter_norm"] = row.pop("parameter_squared")**.5
    return {"scope": "after outer clipping; local parameter/gradient shards for FSDP2; no extra collectives", "groups": groups}
