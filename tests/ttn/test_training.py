import copy
import torch
import pytest
from torch import nn
from test_anchor import ProjectionContract
from worldttn.anchor import install_ttn, TTNAnchor
from worldttn.core import TTNConfig
from worldttn.training import train_clip, linear_flow_loss
from worldttn.checkpoint import save_checkpoint, load_checkpoint, make_optimizer


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = ProjectionContract(2, 8)
        self.ffn = nn.Linear(16, 16)

    def forward(self, z, cache, context, hw, save):
        if isinstance(self.attn, TTNAnchor):
            result, cache = self.attn(z, HW=hw, ttn_chunk_context=context, kv_cache=cache)
            z = z + .02 * result
        z = z + .01 * self.ffn(z)
        cache = list(cache)
        if save: cache[9] = z[:, -1:].detach()
        return z, cache


class TinyWorldModel(nn.Module):
    """Small functional model exercising the real adapter and session contracts on CPU."""

    def __init__(self, stage="C"):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(20):
            self.blocks.append(TinyBlock())
        install_ttn(self, TTNConfig(heads=2, head_dim=8, generators=3, stage=stage))
        self.saved_features = []

    def forward(self, x, t, y, kv_cache, ttn_chunk_context, save_kv_cache=False, **kw):
        b, c, f, h, w = x.shape
        z = x.permute(0, 2, 3, 4, 1).reshape(b, -1, c)
        caches = []
        for block, cache in zip(self.blocks, kv_cache):
            z, cache = block(z, cache, ttn_chunk_context, (f, h, w), save_kv_cache)
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


@pytest.mark.parametrize("activation_offload", ["none", "cpu"])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_real_train_clip_tbptt_and_single_optimizer_step(k, activation_offload):
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
                        tbptt=k,
                        activation_offload=activation_offload)
    assert result["runtime"].commit_count == 5 and result["runtime"].predict_count == 5
    assert result["runtime"].committed_frame_ids == [set(range(13))]
    # Prefill is outside TBPTT windows; current GT candidates influence the next chunk inside a window.
    s = m.saved_features[1]
    assert (s.grad is None) if k == 1 else (s.grad is not None and s.grad.norm() > 0)
    assert all(int(state["step"]) == 1 for state in opt.state.values())
    assert m.blocks[3].attn.qkv.weight.grad.norm() > 0
    assert torch.isfinite(torch.tensor(result["loss"]))


@pytest.mark.parametrize("stage", ["A", "B", "C"])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_cpu_activation_offload_preserves_updates_states_and_transactions(stage, k):
    torch.manual_seed(17)
    baseline = TinyWorldModel(stage)
    offloaded = copy.deepcopy(baseline)
    clean, noise, t, cam = inputs()
    phases = []
    def run(model, offload, callback=None):
        return train_clip(model, clean, torch.zeros(1, 1, 2, 8), cam, make_optimizer(model),
                          linear_flow_loss, t, noise, width=100, height=100, tbptt=k,
                          activation_offload=offload, memory_callback=callback)
    expected = run(baseline, "none")
    actual = run(offloaded, "cpu", lambda phase, **info: phases.append((phase, info)))
    assert actual["loss"] == expected["loss"]
    assert actual["outer_grad_norm"] == expected["outer_grad_norm"]
    for left, right in zip(baseline.parameters(), offloaded.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        if left.grad is None: assert right.grad is None
        else: torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    for name in ("world_state", "transition_fast", "previous_committed_pose"):
        torch.testing.assert_close(getattr(actual["runtime"], name), getattr(expected["runtime"], name), rtol=0, atol=0)
    assert actual["runtime"].commit_count == actual["runtime"].predict_count == 5
    assert actual["runtime"].committed_frame_ids == expected["runtime"].committed_frame_ids
    names = [name for name, info in phases]
    assert names[:2] == ["prefill_begin", "prefill_end"]
    assert names.count("noisy_begin") == names.count("clean_end") == 4
    assert names.count("backward_end") == (4 + k - 1) // k
    assert names[-2:] == ["optimizer_begin", "optimizer_end"]
    assert [(name, info["chunk"]) for name, info in phases if name in ("noisy_begin", "clean_begin")] == [
        (name, chunk) for chunk in range(4) for name in ("noisy_begin", "clean_begin")]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA activation-memory measurement requires a GPU")
def test_save_on_cpu_reduces_live_cuda_saved_activations_without_changing_gradients():
    from worldttn.training import activation_storage
    results = []
    for mode in ("none", "cpu"):
        torch.manual_seed(7)
        x = torch.randn(2048, 2048, device="cuda", requires_grad=True)
        with activation_storage(mode):
            z = x
            for _ in range(8): z = torch.sin(z)
            loss = z.square().mean()
        torch.cuda.synchronize()
        live = torch.cuda.memory_allocated()
        loss.backward()
        results.append((live, x.grad.cpu().clone(), loss.item()))
        del x, z, loss
        torch.cuda.empty_cache()
    assert results[1][0] < results[0][0] * .75
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
    assert results[0][2] == results[1][2]


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
