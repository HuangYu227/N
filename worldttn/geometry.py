"""Shared pure tensor UCPE primitives, extracted from SANA without changing math."""
from functools import partial
from typing import Callable, List, Optional, Tuple
import torch
from torch.nn import functional as F


def apply_ray_projmat(
        feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
        matrix: torch.Tensor,  # (batch, seqlen, 4, 4)
) -> torch.Tensor:
    """Apply a per-token 4x4 projection matrix to feature channels grouped by 4."""
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    D = matrix.shape[-1]
    return torch.einsum(
        "bnij,bhnkj->bhnki",
        matrix,
        feats.reshape(batch, num_heads, seqlen, -1, D),
    ).reshape(feats.shape)


def apply_complex_rope(
    hidden_states: torch.Tensor,
    freqs: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply complex RoPE; SANA optionally compiles this same pure Tensor function."""
    x_real = hidden_states.to(torch.float64)
    if x_real.stride(-1) != 1:
        x_real = x_real.contiguous()
    x_complex = torch.view_as_complex(x_real.unflatten(-1, (-1, 2)))
    if inverse:
        freqs = freqs.conj()
    x_out = torch.view_as_real(x_complex * freqs).flatten(-2, -1)
    return x_out.type_as(hidden_states)


def apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function: split features by sizes, transform each, concat."""
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


def invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of a 4x4 SE(3) batch."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


# ---------------------------------------------------------------------------
# UCPE apply-fn preparation
# ---------------------------------------------------------------------------


def prepare_ray_apply_fns(
    head_dim: int,
    P: torch.Tensor,  # (batch, seqlen, 4, 4) P = ray<-world
    P_T: torch.Tensor,  # (batch, seqlen, 4, 4) P_T = world<-ray
    P_inv: torch.Tensor,  # (batch, seqlen, 4, 4) P_inv = world<-ray
    rotary_emb: Optional[torch.Tensor] = None,
    apply_vo: bool = True,
    *,
    ray_apply=apply_ray_projmat,
    rope_apply=apply_complex_rope,
) -> Tuple[Callable, Callable, Callable]:
    """Build ``(apply_q, apply_kv, apply_o)`` block-diagonal callables for UCPE."""
    if rotary_emb is not None:
        rope_fn = partial(rope_apply, freqs=rotary_emb, inverse=False)
        rope_fn_inv = partial(rope_apply, freqs=rotary_emb, inverse=True)
    else:
        rope_fn = lambda x: x
        rope_fn_inv = lambda x: x

    transforms_q = [
        (partial(ray_apply, matrix=P_T), head_dim // 2),
        (rope_fn, head_dim // 2),
    ]
    transforms_kv = [
        (partial(ray_apply, matrix=P_inv), head_dim // 2),
        (rope_fn, head_dim // 2),
    ]
    if apply_vo:
        transforms_o = [
            (partial(ray_apply, matrix=P), head_dim // 2),
            (rope_fn_inv, head_dim // 2),
        ]
    else:
        transforms_o = lambda x: x

    apply_fn_q = partial(apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(apply_block_diagonal, func_size_pairs=transforms_o) if apply_vo else transforms_o

    return apply_fn_q, apply_fn_kv, apply_fn_o


def _world_to_ray_mats(
        d_cam: torch.Tensor,  # [H, W, 3], [B, H, W, 3], or [B, T, H, W, 3]
        c2w: torch.Tensor,  # [B, T, 4, 4]
) -> torch.Tensor:
    """Build per-pixel ``ray<-world`` transforms from camera unit rays + C2W poses."""
    if d_cam.ndim == 3:
        d_cam = d_cam.unsqueeze(0)
    if d_cam.ndim == 4:
        B, H, W, _ = d_cam.shape
        T = c2w.shape[1]
        d_cam = d_cam.unsqueeze(1).expand(-1, T, -1, -1, -1)
    elif d_cam.ndim == 5:
        B, T, H, W, _ = d_cam.shape
    else:
        raise ValueError(f"Unsupported d_cam shape: {d_cam.shape}")

    device = d_cam.device
    dtype = d_cam.dtype
    R_cam = c2w[..., :3, :3]
    t_cam = c2w[..., :3, 3]
    d_world = torch.einsum("btij,bthwj->bthwi", R_cam, d_cam)
    cam_y = R_cam[..., :, 1]
    cam_y = cam_y[:, :, None, None, :].expand(-1, -1, H, W, -1)
    z_ray = F.normalize(d_world, dim=-1, eps=1e-6)
    x_ray = torch.cross(cam_y, z_ray, dim=-1)
    x_ray = F.normalize(x_ray, dim=-1, eps=1e-6)
    y_ray = torch.cross(z_ray, x_ray, dim=-1)
    y_ray = F.normalize(y_ray, dim=-1, eps=1e-6)
    R_l2w = torch.stack([x_ray, y_ray, z_ray], dim=-1)
    R_w2l = R_l2w.transpose(-1, -2)
    t_world = t_cam[:, :, None, None, :].expand(-1, -1, H, W, -1)
    t_w2l = -torch.einsum("bthwij,bthwj->bthwi", R_w2l, t_world)
    raymats = torch.zeros(B, T, H, W, 4, 4, device=device, dtype=dtype)
    raymats[..., :3, :3] = R_w2l
    raymats[..., :3, 3] = t_w2l
    raymats[..., 3, 3] = 1.0
    mask = torch.isnan(d_world).any(-1)
    raymats[mask] = torch.eye(4, device=device, dtype=dtype)
    return raymats


def world_to_ray_mats(d_cam, c2w):
    """Preserve input precision; autocast must not round camera geometry."""
    with torch.autocast(device_type=d_cam.device.type, enabled=False):
        return _world_to_ray_mats(d_cam, c2w)
