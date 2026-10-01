import copy
import torch
import pytest
from torch import nn
from test_anchor import ProjectionContract
from worldttn.anchor import install_ttn, TTNAnchor
from worldttn.core import TTNConfig
from worldttn.training import train_clip, linear_flow_loss
from worldttn.checkpoint import save_checkpoint, load_checkpoint, make_optimizer


class TinyWorldModel(nn.Module):
    """Small functional model exercising the real adapter and session contracts on CPU."""

    def __init__(self, stage="C"):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(20):
            block = nn.Module()
            block.attn = ProjectionContract(2, 8)
            block.ffn = nn.Linear(16, 16)
            self.blocks.append(block)
        install_ttn(self, TTNConfig(heads=2, head_dim=8, generators=3, stage=stage))
        self.saved_features = []

    def forward(self, x, t, y, kv_cache, ttn_chunk_context, save_kv_cache=False, **kw):
        b, c, f, h, w = x.shape
        z = x.permute(0, 2, 3, 4, 1).reshape(b, -1, c)
        caches = []
        for block, cache in zip(self.blocks, kv_cache):
            if isinstance(block.attn, TTNAnchor):
                result, cache = block.attn(z, HW=(f, h, w), ttn_chunk_context=ttn_chunk_context, kv_cache=cache)
                z = z + .02 * result
            z = z + .01 * block.ffn(z)
            cache = list(cache)
            if save_kv_cache: cache[9] = z[:, -1:].detach()
            caches.append(cache)
        if ttn_chunk_context.clean_mode:
            self.saved_features.append(ttn_chunk_context.candidates[0][0])
        return z.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3), caches


def inputs():
    torch.manual_seed(10)
    clean = torch.randn(1, 16, 13, 1, 1)
    noise = torch.randn_like(clean)
    t = torch.ones(1, 1, 13) * 500
    t[:, :, 0] = 0
    cam = torch.eye(4).flatten().expand(1, 13, 16)
    cam = torch.cat((cam, torch.tensor([100., 100., 50., 50.]).expand(1, 13, 4)), -1)
    return clean, noise, t, cam


@pytest.mark.parametrize("k", [1, 2, 4])
def test_real_train_clip_tbptt_and_single_optimizer_step(k):
    torch.manual_seed(1)
    m = TinyWorldModel()
    opt = make_optimizer(m)
    clean, noise, t, cam = inputs()
    # Retain clean candidates to observe whether future flow losses reach them.
    original = m.forward

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs["ttn_chunk_context"].clean_mode:
            for s, _, _ in kwargs["ttn_chunk_context"].candidates.values():
                if s.requires_grad: s.retain_grad()
        return result

    m.forward = record
    result = train_clip(m,
                        clean,
                        torch.zeros(1, 1, 2, 8),
                        cam,
                        opt,
                        linear_flow_loss,
                        t,
                        noise,
                        width=100,
                        height=100,
                        tbptt=k)
    assert result["runtime"].commit_count == 5 and result["runtime"].predict_count == 5
    assert result["runtime"].committed_frame_ids == [set(range(13))]
    # Prefill is outside TBPTT windows; current GT candidates influence the next chunk inside a window.
    s = m.saved_features[1]
    assert (s.grad is None) if k == 1 else (s.grad is not None and s.grad.norm() > 0)
    assert all(int(state["step"]) == 1 for state in opt.state.values())
    assert m.blocks[3].attn.qkv.weight.grad.norm() > 0
    assert torch.isfinite(torch.tensor(result["loss"]))


def test_future_chunk_does_not_change_earlier_predictions():
    torch.manual_seed(1)
    m = TinyWorldModel("B")
    state = copy.deepcopy(m.state_dict())
    clean, noise, t, cam = inputs()
    seen = []
    train_clip(m,
               clean,
               torch.zeros(1, 1, 2, 8),
               cam,
               make_optimizer(m),
               linear_flow_loss,
               t,
               noise,
               width=100,
               height=100,
               tbptt=2,
               on_prediction=lambda i, x: seen.append(x.detach().clone()))
    changed = clean.clone()
    changed[:, :, 7:] += 100
    m2 = TinyWorldModel("B")
    m2.load_state_dict(state)
    again = []
    train_clip(m2,
               changed,
               torch.zeros(1, 1, 2, 8),
               cam,
               make_optimizer(m2),
               linear_flow_loss,
               t,
               noise,
               width=100,
               height=100,
               tbptt=2,
               on_prediction=lambda i, x: again.append(x.detach().clone()))
    assert torch.equal(seen[0], again[0]) and torch.equal(seen[1], again[1])


def test_checkpoint_roundtrip_resume_and_stage_transition(tmp_path):
    m = TinyWorldModel("A")
    opt = make_optimizer(m)
    clean, noise, t, cam = inputs()
    train_clip(m, clean, torch.zeros(1, 1, 2, 8), cam, opt, linear_flow_loss, t, noise, width=100, height=100, tbptt=1)
    path = tmp_path / "last.pt"
    save_checkpoint(path, m, opt, 7)
    payload = torch.load(path, weights_only=False)
    assert "runtime" not in payload and all("world_state" not in name for name in payload["adapter"])
    expected_rng = torch.rand(5)
    n = TinyWorldModel("A")
    newopt = make_optimizer(n)
    assert load_checkpoint(path, n, newopt, resume=True) == 7
    assert torch.equal(torch.rand(5), expected_rng)
    assert torch.equal(n.blocks[3].attn.qkv.weight, m.blocks[3].attn.qkv.weight)
    assert newopt.state
    n = TinyWorldModel("B")
    newopt = make_optimizer(n)
    assert load_checkpoint(path, n, newopt, resume=False) == 0 and not newopt.state
    with pytest.raises(ValueError):
        load_checkpoint(path, n, newopt, resume=True)
    bad = copy.deepcopy(payload)
    bad["config"]["base_id"] = "wrong-base"
    torch.save(bad, path)
    with pytest.raises(ValueError):
        load_checkpoint(path, n)
