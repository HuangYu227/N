import pytest
import torch
from torch import nn
from worldttn.core import TTNConfig, ANCHORS
from worldttn.runtime import TTNSystem, TTNRuntimeState
from worldttn.anchor import TTNAnchor, install_ttn


# Real projection contract; no mocked attention kernels.
class ProjectionContract(nn.Module):

    def __init__(self, h=2, d=8):
        super().__init__()
        self.heads = h
        self.dim = d
        self.cam_head_dim = d
        self.patch_size = (1, 1, 1)
        self.qkv = nn.Linear(h * d, 3 * h * d)
        self.q_norm = nn.LayerNorm(h * d)
        self.k_norm = nn.LayerNorm(h * d)
        self.q_norm_cam = nn.LayerNorm(h * d)
        self.k_norm_cam = nn.LayerNorm(h * d)
        for name in ("q_proj_cam", "k_proj_cam", "v_proj_cam", "out_proj_cam", "output_gate", "proj"):
            setattr(self, name, nn.Linear(h * d, h * d))
        self.beta_proj = nn.Linear(h * d, h)
        nn.init.constant_(self.beta_proj.weight, 17.)


def context(cfg, b=1, f=2):
    sys = TTNSystem(cfg)
    r = TTNRuntimeState.create(cfg, b, "cpu")
    p = torch.eye(4).expand(b, f, 4, 4)
    intr = torch.tensor([100., 100., 50., 50.]).expand(b, f, 4)
    return r, r.begin_chunk(sys, p, intr, torch.arange(f).expand(b, -1), torch.ones(b, f, dtype=torch.bool), 100, 100)


def test_stage_a_reuses_projection_and_shared_silu_gate_without_sdpa(monkeypatch):
    cfg = TTNConfig(heads=2, head_dim=8, generators=3)
    source = ProjectionContract()
    anchor = TTNAnchor(source, 0, cfg)
    assert anchor.qkv is source.qkv and anchor.output_gate is source.output_gate
    assert anchor.beta_proj.weight.count_nonzero() == 0
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention",
                        lambda *a, **k: pytest.fail("SDPA reached"))
    r, c = context(cfg)
    x = torch.randn(1, 6, 16)
    read, write = c.token_masks(6)
    q, k, v = anchor.visual_features(x, None)
    beta = anchor.beta_proj(x.float()).sigmoid().transpose(1, 2)
    from worldttn.core import correct
    s, _ = correct(c.predicted[:, 0], k, v, beta, write, cfg.alpha_s, cfg.eps)
    camera = torch.randn(1, 2, 6, 8)
    # Identity camera geometry still runs the real camera projections and linear mixer.
    fns = (lambda z: z, lambda z: z, lambda z: z)
    cq, ck, cv, co = anchor.camera_features(x, (2, 1, 3), None, None, fns)
    cs = ck.transpose(-1, -2) @ cv / (6 * 8**.5)
    raw = (q @ s).transpose(1, 2).reshape_as(x) + anchor.out_proj_cam((cq @ cs).transpose(1, 2).reshape_as(x))
    expected = anchor.proj((raw * torch.nn.functional.silu(anchor.output_gate(x).float())).to(x.dtype))
    out, cache = anchor(x,
                        HW=(2, 1, 3),
                        camera_conditions=torch.zeros(1, 2, 20),
                        prope_fns=fns,
                        ttn_chunk_context=c,
                        kv_cache=[None] * 9 + ["ffn"])
    assert torch.allclose(out, expected, atol=1e-6)
    assert cache[:9] == [None] * 9 and cache[9] == "ffn"
    assert not c.candidates and r.world_state.count_nonzero() == 0


def test_five_replacements_and_frozen_backbone_gradient_path():
    cfg = TTNConfig(heads=2, head_dim=8, generators=3, stage="B")
    model = nn.Module()
    model.blocks = nn.ModuleList()
    for i in range(20):
        block = nn.Module()
        block.attn = ProjectionContract()
        block.ffn = nn.Linear(16, 16)
        model.blocks.append(block)
    install_ttn(model, cfg)
    assert [i for i, b in enumerate(model.blocks) if isinstance(b.attn, TTNAnchor)] == list(ANCHORS)
    assert model.ttn_system.generators.u.requires_grad
    assert all(not p.requires_grad for b in model.blocks for p in b.ffn.parameters())
    assert all(not p.requires_grad for i, b in enumerate(model.blocks) if i not in ANCHORS for p in b.attn.parameters())
    r, c = context(cfg)
    x = torch.randn(1, 6, 16, requires_grad=True)
    out = model.blocks[3].attn(x, HW=(2, 1, 3), ttn_chunk_context=c)
    out = model.blocks[4].ffn(out)
    out.square().mean().backward()
    assert model.blocks[3].attn.qkv.weight.grad.norm() > 0
    assert x.grad.norm() > 0


def test_clean_anchor_keeps_outer_graph_but_detaches_inner():
    cfg = TTNConfig(heads=2, head_dim=8, generators=3, stage="C")
    r, c = context(cfg)
    c = c.for_clean()
    a = TTNAnchor(ProjectionContract(), 0, cfg)
    a(torch.randn(1, 6, 16), HW=(2, 1, 3), ttn_chunk_context=c)
    s, g, stats = c.candidates[0]
    assert s.requires_grad and not g.requires_grad and stats["write_tokens"] == 6
    s.square().sum().backward()
    assert a.qkv.weight.grad.norm() > 0
