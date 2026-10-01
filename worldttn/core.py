"""Persistent matrix correction and rank-2 Cayley transport (no token-pair matrix)."""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F

ANCHORS = (3, 7, 11, 15, 19)
BASE_ID = "hf://Efficient-Large-Model/SANA-WM_chunk_causal/dit/sana_wm_chunk_causal_1600m_720p.safetensors"
BASE_REVISION = "f9178744c096dcf2a2ea773da183e341bcbeb044"


@dataclass(frozen=True)
class TTNConfig:
    """A: identity Predict + Correct/Read; B: learned Predict + Correct/Read;
    C: B plus a detached, per-clean-chunk first-order update of psi.
    All stages train the five replacement anchors' projections/Norms/gates/beta.
    """
    heads: int = 20
    head_dim: int = 112
    generators: int = 16
    alpha_s: float = .5
    eps: float = 1e-6
    delta_psi: float = 1.
    eta_psi: float = .01
    inner_clip: float = 1.
    stage: str = "A"
    base_id: str = BASE_ID

    def __post_init__(self):
        if self.stage not in ("A", "B", "C"):
            raise ValueError("stage must be A, B or C")
        if min(self.heads, self.head_dim, self.generators) < 1 or self.head_dim % 8:
            raise ValueError("positive dimensions; UCPE half-head must be divisible by four")
        if not 0 < self.alpha_s < 2 or self.eps <= 0 or min(self.delta_psi, self.eta_psi, self.inner_clip) < 0:
            raise ValueError("invalid correction/adaptation hyperparameters")

    def to_dict(self):
        return asdict(self)


class Rank2Generators(nn.Module):

    def __init__(self, config):
        super().__init__()
        shape = (5, config.heads, config.generators, config.head_dim)
        self.u = nn.Parameter(F.normalize(torch.randn(shape), dim=-1))
        self.v = nn.Parameter(F.normalize(torch.randn(shape), dim=-1))


class CayleyFactors:
    """R=I+P L P^T. The only solve has size 2M; no inverse of a dxd matrix.

    Per head, factor construction costs O(d M^2 + M^3); right-multiplying
    a dxd state costs O(d^2 M + d M^2). Neither depends on token count.
    This excludes token projections, Correct/Read, and the local psi gradient.
    """

    def __init__(self, u, v, c):
        # Stack [..., M, 2, D], then interleave generators on the column axis.
        self.p = torch.stack((u, v), dim=-2).flatten(-3, -2).transpose(-1, -2)
        m = c.shape[-1]
        omega = c.new_zeros(*c.shape[:-1], 2 * m, 2 * m)
        idx = torch.arange(m, device=c.device) * 2
        omega[..., idx, idx + 1] = c
        omega[..., idx + 1, idx] = -c
        gram = self.p.transpose(-1, -2) @ self.p
        eye = torch.eye(2 * m, dtype=c.dtype, device=c.device)
        self.l = torch.linalg.solve(eye - .5 * omega @ gram, omega)

    def right(self, x, inverse_b=False, transpose=False):
        l = self.l.transpose(-1, -2) if transpose else self.l
        return x + (.5 if inverse_b else 1.) * ((x @ self.p) @ l) @ self.p.transpose(-1, -2)


def cayley_dense(u, v, c):
    """FP64-friendly oracle, deliberately separate from the low-rank solve."""
    g = u.unsqueeze(-1) * v.unsqueeze(-2) - v.unsqueeze(-1) * u.unsqueeze(-2)
    a = (c[..., None, None] * g).sum(-3)
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    return torch.linalg.solve(eye - a / 2, eye + a / 2)


def write_weights(beta, mask, eps):
    mask = mask.to(dtype=beta.dtype)
    while mask.ndim < beta.ndim:
        mask = mask.unsqueeze(-2)
    count = mask.sum(-1, keepdim=True)
    mean = (beta * mask).sum(-1, keepdim=True) / count.clamp_min(1)
    return beta * mask / (eps + mean)


def correct(pred, k, v, beta, mask, alpha_s=.5, eps=1e-6):
    """One normalized, weighted least-squares step from the predicted state.

    With zero prediction and uniform write weights this reduces to a scaled
    K^T V memory. Persistence alone does not distinguish it from recurrent
    linear attention; the camera-conditioned transition is a separate operation.
    """
    w = write_weights(beta, mask, eps)
    denom = eps + (w * k.square().sum(-1)).sum(-1)
    residual = v - k @ pred
    candidate = pred + (alpha_s / denom)[..., None, None] * (k.transpose(-1, -2) @ (w[..., None] * residual))
    return candidate, w


def innovation_loss(pred, k, v, w, mask):
    count = mask.sum(-1).clamp_min(1).to(k.dtype)
    while count.ndim < k.ndim - 2:
        count = count.unsqueeze(-1)
    return (w * (k @ pred - v).square().sum(-1)).sum(-1) / (2 * k.shape[-1] * count)


@torch.no_grad()
def analytic_psi_gradient(prev, k, value, w, mask, u, v, cbase, psi, delta_psi):
    # detach every local input even when called within an outer training graph
    prev, k, value, w, u, v, cbase, psi = [t.detach() for t in (prev, k, value, w, u, v, cbase, psi)]
    factors = CayleyFactors(u, v, cbase + delta_psi * psi.tanh())
    pred = factors.right(prev)
    count = mask.sum(-1).clamp_min(1).to(k.dtype)
    while count.ndim < k.ndim - 2:
        count = count.unsqueeze(-1)
    h = k.transpose(-1, -2) @ (w[..., None] * (k @ pred - value)) / (count[..., None, None] * k.shape[-1])
    # The reference local gradient includes this dense O(d^3) product per
    # head/clean chunk; the 2M solve is not its entire computational cost.
    j = prev.transpose(-1, -2) @ h
    f = factors.right(j.transpose(-1, -2), inverse_b=True).transpose(-1, -2)
    f = factors.right(f, inverse_b=True, transpose=True)
    gc = torch.einsum("...md,...de,...me->...m", u, f, v) - torch.einsum("...md,...de,...me->...m", v, f, u)
    return delta_psi * (1 - psi.tanh().square()) * gc
