"""Actual Softmax replacement. Imports only torch until camera geometry is requested."""
import torch
from torch import nn
from torch.nn import functional as F
from .core import ANCHORS, correct, innovation_loss, analytic_psi_gradient
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
        diagnostic = kwargs.get("ttn_diagnostic")
        cfg = self.config
        b, n, c = x.shape
        if diagnostic is not None: diagnostic("input", x)
        read, write = ctx.token_masks(n)
        x = x * read[..., None].to(x.dtype)
        q, k, v = self.visual_features(x, rotary_emb)
        if diagnostic is not None: diagnostic("visual_features", (q, k, v))
        with torch.autocast(device_type=x.device.type, enabled=False):
            beta = self.beta_proj(x.float()).sigmoid().transpose(1, 2)
            state, w = correct(ctx.predicted[:, self.index], k, v, beta, write, cfg.alpha_s, cfg.eps)
            raw = q @ state
            if diagnostic is not None:
                diagnostic("memory", (ctx.predicted[:, self.index], state, k, v, beta, w, write))
            if ctx.clean_mode:
                gradient = torch.zeros_like(ctx.psi[:, self.index])
                if cfg.stage == "C" and not ctx.prefill_mode:
                    gradient = analytic_psi_gradient(ctx.previous[:, self.index], k, v, w, write,
                                                     ctx.system.generators.u[self.index].float(),
                                                     ctx.system.generators.v[self.index].float(),
                                                     ctx.cbase[:, self.index], ctx.psi[:, self.index], cfg.delta_psi)
                stats = {
                    "write_tokens": int(write.sum()),
                    "read_tokens": int(read.sum()),
                    "innovation_loss":
                    float(innovation_loss(ctx.predicted[:, self.index], k, v, w, write).detach().mean()),
                    "inner_grad_norm": float(gradient.norm())
                }
                ctx.stage(self.index, state, gradient, stats)
        raw = raw.transpose(1, 2).reshape(b, n, c)
        if diagnostic is not None: diagnostic("visual_raw", raw)
        if camera_conditions is not None:
            cq, ck, cv, to = self.camera_features(x, HW, camera_conditions, rotary_emb, kwargs.get("prope_fns"))
            with torch.autocast(device_type=x.device.type, enabled=False):
                denom = read.sum(-1).clamp_min(1).float() * self.cam_head_dim**.5
                camera_state = ck.transpose(-1, -2) @ (cv * read[:, None, :, None]) / denom[:, None, None, None]
                camera_out = to(cq @ camera_state)  # output transform BEFORE MergeHeads
                camera_out = camera_out.transpose(1, 2).reshape(b, n, c)
            camera_contribution = self.out_proj_cam(camera_out.to(x.dtype))
            if diagnostic is not None:
                diagnostic("camera_raw", camera_out.to(x.dtype))
                diagnostic("camera_contribution", camera_contribution)
            raw = raw + camera_contribution
        if diagnostic is not None: diagnostic("fused_raw", raw)
        gate = F.silu(self.output_gate(x).float())
        gated = (raw * gate).to(x.dtype)
        if diagnostic is not None: diagnostic("gated_raw", gated)
        out = self.proj(gated) * read[..., None].to(x.dtype)
        cache = kwargs.get("kv_cache")
        return (out, [None] * 9 + [cache[9]]) if cache is not None else out


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
