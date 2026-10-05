"""Actual Softmax replacement. Imports only torch until camera geometry is requested."""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from .core import (ANCHORS, correct, analytic_psi_gradient, correct_with_aux, CorrectionAux,
                   analytic_psi_gradient_dense_from_aux, analytic_psi_gradient_projected,
                   CayleyFactors, write_weights, innovation_objective, coefficient_gradient, clip_inner_gradient)
from .performance import DEFAULT_EXECUTION, annotation, audit_layout
from .stability import clean_anchor_stats, local_anchor_stats, matrix_scale
from .geometry import apply_complex_rope
from .runtime import TTNSystem


class TTNAnchor(nn.Module):

    def __init__(self, source, index, config):
        super().__init__()
        if source.heads != config.heads or source.dim != config.head_dim:
            raise ValueError("checkpoint projection dimensions disagree with TTN config")
        if source.output_gate is None:
            raise ValueError("TTN v0.1 requires SANA's shared SiLU output gate")
        self.index = index
        self.config = config
        self.heads = source.heads
        self.dim = source.dim
        self.cam_head_dim = source.cam_head_dim
        self.patch_size = source.patch_size
        for name in ("qkv", "q_norm", "k_norm", "proj", "output_gate", "q_proj_cam", "k_proj_cam", "v_proj_cam",
                     "out_proj_cam", "q_norm_cam", "k_norm_cam"):
            setattr(self, name, getattr(source, name))
        self.beta_proj = nn.Linear(config.heads * config.head_dim,
                                   config.heads,
                                   device=source.qkv.weight.device,
                                   dtype=torch.float32)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        self.qkv_store_buffer = None
        native_camera = getattr(source, "_cached_cam_branch_softmax", None)
        self._sana_camera_forward = native_camera.__func__ if native_camera is not None else None
        if native_camera is not None:
            for name in ("cam_heads", "cam_dim", "conv_q_cam", "conv_k_cam", "conv_v_cam"):
                setattr(self, name, getattr(source, name))
        if config.camera_attention == "sana" and self._sana_camera_forward is None:
            raise ValueError("sana camera requires the original cached SANA camera implementation")

    def visual_features(self, x, rotary_emb):
        b, n, c = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = F.silu(self.q_norm(q)).reshape(b, n, self.heads, self.dim).transpose(1, 2)
        k = F.silu(self.k_norm(k)).reshape(b, n, self.heads, self.dim).transpose(1, 2)
        v = F.silu(v).reshape(b, n, self.heads, self.dim).transpose(1, 2)
        if rotary_emb is not None:
            rotary_emb = rotary_emb[..., -n:, :]
            q = apply_complex_rope(q, rotary_emb)
            k = apply_complex_rope(k, rotary_emb)
        return q.float(), k.float(), v.float()

    def camera_features(self, x, hw, camera_conditions, rotary_emb, prope_fns):
        b, n, c = x.shape
        q = F.silu(self.q_norm_cam(self.q_proj_cam(x))).reshape(b, n, self.heads, self.cam_head_dim).transpose(1, 2)
        k = F.silu(self.k_norm_cam(self.k_proj_cam(x))).reshape(b, n, self.heads, self.cam_head_dim).transpose(1, 2)
        v = F.silu(self.v_proj_cam(x)).reshape(b, n, self.heads, self.cam_head_dim).transpose(1, 2)
        if prope_fns is None:
            from diffusion.model.nets.sana_camctrl_blocks import prepare_prope_fns
            prope_fns = prepare_prope_fns("UCPE", self.cam_head_dim, camera_conditions, hw, self.patch_size, rotary_emb)
        tq, tkv, to = prope_fns
        return tq(q).float(), tkv(k).float(), tkv(v).float(), to

    def forward(self,
                x,
                mask=None,
                HW=None,
                rotary_emb=None,
                block_mask=None,
                camera_conditions=None,
                chunk_size=None,
                ttn_chunk_context=None,
                **kwargs):
        if ttn_chunk_context is None:
            raise ValueError("TTN anchors require an explicit ttn_chunk_context")
        ctx = ttn_chunk_context
        incoming_cache = kwargs.get("kv_cache")
        cache = list(incoming_cache) if incoming_cache is not None else [None] * 10
        cache[0] = cache[1] = None  # Visual history lives in TTN S, never visual K/V.
        diagnostic = kwargs.get("ttn_diagnostic")
        cfg = self.config
        execution = getattr(self, "ttn_execution", DEFAULT_EXECUTION)
        b, n, c = x.shape
        if diagnostic is not None: diagnostic("input", x)
        read, write = ctx.token_masks(n)
        x = (torch.where(read[..., None], x, 0.) if cfg.local_update or cfg.persistent_meta
             else x * read[..., None].to(x.dtype))
        q, k, v = self.visual_features(x, rotary_emb)
        if diagnostic is not None: diagnostic("visual_features", (q, k, v))
        with torch.autocast(device_type=x.device.type, enabled=False):
            beta = self.beta_proj(x.float()).sigmoid().transpose(1, 2)
            predicted = ctx.predicted[:, self.index]
            local_enabled = cfg.local_update and not ctx.clean_mode and not ctx.prefill_mode and ctx.ablation in ("full", "no-persistent")
            if local_enabled:
                support, query = ctx.support_query_masks(n, HW)
                factors = ctx.live_factors
                if factors is None: raise RuntimeError("Local Meta-TTT requires live Cayley factors")
                w = write_weights(beta, write, cfg.eps)
                _, h = innovation_objective(predicted, k, v, w, support, normalized=True, eps=cfg.eps)
                gradient = coefficient_gradient(ctx.previous[:, self.index], h,
                                                factors.p[self.index], factors.l[:, self.index])
                clipped, scale = clip_inner_gradient(gradient, cfg.inner_clip, cfg.eps)
                eta = F.softplus(ctx.system.local_eta_logits[self.index])
                local = -eta * clipped
                if support.any():
                    persistent = cfg.delta_psi * ctx.psi[:, self.index].tanh() if ctx.ablation == "full" else 0
                    coeff = ctx.cbase[:, self.index] + persistent + local.tanh()
                    adapted = CayleyFactors(ctx.system.generators.u[self.index].float(),
                                           ctx.system.generators.v[self.index].float(), coeff).right(ctx.previous[:, self.index])
                    predicted = torch.where(support.any(-1)[:, None, None, None], adapted, predicted)
            audit_layout(self, "clean" if ctx.clean_mode else "diagnostic" if diagnostic is not None else "noisy",
                         q=q, k=k, v=v, beta=beta, predicted=predicted, write=write)
            with annotation(execution, "CorrectRead"):
                if execution.core_backend == "reference":
                    state, w = correct(predicted, k, v, beta, write, cfg.alpha_s, cfg.eps)
                    raw = q @ state
                elif execution.core_backend == "compiled":
                    from . import compiled
                    if ctx.clean_mode or diagnostic is not None:
                        raw, *values = compiled.with_aux(q, k, v, beta, predicted, write, cfg.alpha_s, cfg.eps)
                        aux = CorrectionAux(*values)
                        state, w = aux.candidate, aux.w
                    else:
                        raw = compiled.read_only(q, k, v, beta, predicted, write, cfg.alpha_s, cfg.eps)
                else:
                    aux = correct_with_aux(predicted, k, v, beta, write, cfg.alpha_s, cfg.eps)
                    state, w = aux.candidate, aux.w
                    raw = q @ state
            if ((ctx.collect_local_stats or diagnostic is not None) and (cfg.local_update or cfg.persistent_meta)
                    and not ctx.clean_mode and not ctx.prefill_mode):
                if not local_enabled:
                    support, query = ctx.support_query_masks(n, HW)
                    gradient = clipped = local = torch.zeros_like(ctx.psi[:, self.index])
                    scale = torch.ones_like(gradient[..., :1])
                    eta = F.softplus(ctx.system.local_eta_logits[self.index]) if cfg.local_update else None
                with annotation(execution, "LocalStats"):
                    stats = local_anchor_stats(ctx.previous[:, self.index], ctx.predicted[:, self.index], predicted,
                                               state, k, v, w, support, query, gradient, clipped, scale, local, eta, cfg.eps)
                    stats.update(ctx.noise_info, enabled=local_enabled, recorded=True)
                    ctx.local_stats[self.index] = stats
                    if ctx.local_trajectory:
                        ctx.local_trajectory[-1]["anchors"][self.index] = {
                            "enabled": local_enabled, "eta": stats["eta"],
                            "per_head": {key: stats["per_head"][key] for key in
                                         ("raw_grad_norm", "clipped_grad_norm", "psi_norm", "transport_change_norm")},
                            **{part: {key: stats[part][key] for key in ("loss_before", "loss_after_transport")}
                               for part in ("support", "query")}}
                if diagnostic is not None: diagnostic("local_update", stats)
            if diagnostic is not None:
                diagnostic("memory", (predicted, state, k, v, beta, w, write))
            if ctx.clean_mode:
                gradient = torch.zeros_like(ctx.psi[:, self.index])
                if cfg.stage == "C" and not ctx.prefill_mode:
                    with annotation(execution, "Psi"):
                        if cfg.persistent_meta:
                            if ctx.ablation in ("full", "no-local"):
                                factors = ctx.live_factors
                                if factors is None: raise RuntimeError("Persistent Meta-TTT requires live Cayley factors")
                                _, h = innovation_objective(predicted, k, v, w, write, normalized=False, eps=cfg.eps)
                                gc = coefficient_gradient(ctx.previous[:, self.index], h,
                                                          factors.p[self.index], factors.l[:, self.index])
                                gradient = cfg.delta_psi * (1 - ctx.psi[:, self.index].tanh().square()) * gc
                        elif execution.core_backend == "reference":
                            gradient = analytic_psi_gradient(ctx.previous[:, self.index], k, v, w, write,
                                                             ctx.system.generators.u[self.index].float(),
                                                             ctx.system.generators.v[self.index].float(),
                                                             ctx.cbase[:, self.index], ctx.psi[:, self.index], cfg.delta_psi)
                        else:
                            if ctx.psi_snapshot is None: raise RuntimeError("missing detached Cayley snapshot")
                            ctx.psi_snapshot.validate(ctx.runtime_id, ctx.revision, ctx.begin_id)
                            fn = (analytic_psi_gradient_projected if execution.psi_backend == "projected"
                                  else analytic_psi_gradient_dense_from_aux)
                            gradient = fn(ctx.previous[:, self.index], aux.kt_weighted_residual,
                                          ctx.psi_snapshot.for_anchor(self.index), write.sum(-1), cfg.delta_psi)
                with annotation(execution, "Stats"):
                    stats = clean_anchor_stats(ctx.previous[:, self.index], ctx.predicted[:, self.index],
                                               state, q, k, v, beta, w, read, write, gradient, cfg)
                ctx.stage(self.index, state, gradient, stats)
        raw = raw.transpose(1, 2).reshape(b, n, c)
        if ctx.clean_mode:
            stats.update(camera_attention=cfg.camera_attention,
                         camera_cached_tokens=int(incoming_cache[2].shape[2]) if cfg.camera_attention == "sana" and incoming_cache is not None and incoming_cache[2] is not None else 0,
                         camera_current_tokens=n if camera_conditions is not None else 0,
                         visual_read=matrix_scale(raw))
        if diagnostic is not None: diagnostic("visual_raw", raw)
        if camera_conditions is not None:
            if cfg.camera_attention == "sana":
                # Reuse SANA's projections, Norm, UCPE/RoPE, output transform,
                # SDPA and native current-chunk cache writes WITHOUT SiLU.
                # FLASH/MATH preserve SANA's attention/mask mathematics while
                # excluding the efficient-SDPA backward fault observed on LTU.
                from torch.nn.attention import sdpa_kernel, SDPBackend
                camera_kwargs = {key: value for key, value in kwargs.items()
                                 if key not in ("kv_cache", "save_kv_cache")}
                with annotation(execution, "Camera"), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
                    camera_out = self._sana_camera_forward(
                        self, x, HW, camera_conditions, rotary_emb, cache,
                        kwargs.get("save_kv_cache", False), chunk_size=chunk_size, **camera_kwargs)
            else:
                with annotation(execution, "Camera"):
                    cq, ck, cv, to = self.camera_features(x, HW, camera_conditions, rotary_emb, kwargs.get("prope_fns"))
                    with torch.autocast(device_type=x.device.type, enabled=False):
                        denom = read.sum(-1).clamp_min(1).float() * self.cam_head_dim**.5
                        camera_state = ck.transpose(-1, -2) @ (cv * read[:, None, :, None]) / denom[:, None, None, None]
                        camera_out = to(cq @ camera_state)  # output transform BEFORE MergeHeads
                        camera_out = camera_out.transpose(1, 2).reshape(b, n, c)
            camera_contribution = self.out_proj_cam(camera_out.to(x.dtype))
            if ctx.clean_mode:
                stats.update(camera_raw=matrix_scale(camera_out), camera_contribution=matrix_scale(camera_contribution))
            if diagnostic is not None:
                diagnostic("camera_raw", camera_out.to(x.dtype))
                diagnostic("camera_contribution", camera_contribution)
            raw = raw + camera_contribution
        if diagnostic is not None: diagnostic("fused_raw", raw)
        gate = F.silu(self.output_gate(x).float())
        gated = (raw * gate).to(x.dtype)
        if diagnostic is not None: diagnostic("gated_raw", gated)
        out = self.proj(gated) * read[..., None].to(x.dtype)
        if ctx.clean_mode: stats["anchor_output"] = matrix_scale(out)
        if cfg.camera_attention == "sana":
            cache[6] = x.new_tensor([0.])  # Native concat layout: camera K/V in slots 2/3.
            cache[4] = cache[5] = cache[7] = cache[8] = None
        else:
            cache = [None] * 9 + [cache[9]]
        return (out, cache) if incoming_cache is not None else out


def configure_camera_attention(model, mode):
    """Explicit evaluation ablation, applied after checkpoint validation/loading."""
    config = replace(model.ttn_system.config, camera_attention=mode)
    if mode == "sana" and any(model.blocks[i].attn._sana_camera_forward is None for i in ANCHORS):
        raise ValueError("original SANA camera implementation is unavailable")
    model.ttn_system.config = config
    for i in ANCHORS:
        model.blocks[i].attn.config = config


def install_ttn(model, config):
    """Call AFTER loading the original checkpoint. Remove unused legacy GDN gates."""
    if hasattr(model, "ttn_system") or len(model.blocks) != 20:
        raise ValueError("expected an unmodified twenty-block SANA model")
    # Validate everything before replacing any module.
    for i in ANCHORS:
        a = model.blocks[i].attn
        if isinstance(a, TTNAnchor) or a.heads != config.heads or a.dim != config.head_dim:
            raise ValueError(f"invalid anchor {i}")
    model.requires_grad_(False)
    for index, i in enumerate(ANCHORS):
        model.blocks[i].attn = TTNAnchor(model.blocks[i].attn, index, config).requires_grad_(True)
    model.ttn_system = TTNSystem(config).to(device=next(model.parameters()).device).float()
    model.ttn_reference = True
    model.eval()  # fixed clip conditioning; gradients still flow through the frozen backbone
    return model


def adapter_state_dict(model):
    return {
        name: tensor
        for name, tensor in model.state_dict().items()
        if name.startswith("ttn_system.") or any(name.startswith(f"blocks.{i}.attn.") for i in ANCHORS)
    }


def is_ttn_parameter(name):
    return name.startswith("ttn_system.") or any(name.startswith(f"blocks.{i}.attn.") for i in ANCHORS)


def is_ttn_new_parameter(name):
    """Parameter origin, independent of the replacement module's ownership."""
    return name == "ttn_system.local_eta_logits" or name.startswith(("ttn_system.controller.", "ttn_system.generators.")) or any(
        name.startswith(f"blocks.{i}.attn.beta_proj.") for i in ANCHORS)


def is_ttn_camera_parameter(name):
    return any(name.startswith(f"blocks.{i}.attn.{group}.") for i in ANCHORS
               for group in ("q_proj_cam", "k_proj_cam", "v_proj_cam", "out_proj_cam",
                             "q_norm_cam", "k_norm_cam", "conv_q_cam", "conv_k_cam", "conv_v_cam"))


def trainable_parameter_names(model, scope=None):
    scope = scope or getattr(model, "ttn_train_scope", "ttn")
    if scope not in ("ttn", "ttn-visual", "ttn-new", "dit"): raise ValueError("invalid train-scope")
    return {name for name, _ in model.named_parameters()
            if (scope == "dit" or is_ttn_parameter(name))
            and not (scope == "ttn-new" and not is_ttn_new_parameter(name))
            and not (scope == "ttn-visual" and is_ttn_camera_parameter(name))
            and not (model.ttn_system.config.stage == "A" and name.startswith("ttn_system."))}


def configure_train_scope(model, scope="ttn"):
    """Configure optimizer ownership before DDP/FSDP wrapping; compute still uses autocast."""
    expected = trainable_parameter_names(model, scope)
    if scope in ("ttn-visual", "ttn-new", "dit"): model.float()  # Preserve inherited FP32 masters.
    for name, p in model.named_parameters(): p.requires_grad_(name in expected)
    model.ttn_train_scope = scope
    # A previously fine-tuned, now frozen backbone must still be exported in full.
    if scope == "dit": model.ttn_weight_scope = "dit"
    model.eval()  # Preserve the reference fixed-conditioning/dropout policy.
    return model


def offline_state_dict(model, scope=None):
    scope = scope or getattr(model, "ttn_weight_scope", "ttn")
    if scope not in ("ttn", "dit"): raise ValueError("checkpoint weight scope must be ttn or dit")
    return model.state_dict() if scope == "dit" else adapter_state_dict(model)
