import json

import pytest
import torch

from worldttn.core import TTNConfig, correct, innovation_loss
from worldttn.stability import clean_anchor_stats, committed_anchor_stats, stability_rows, progress_line


def features(scale=1., empty=False):
    cfg = TTNConfig(heads=2, head_dim=8, generators=3, stage="C")
    g = torch.Generator().manual_seed(9)
    previous = torch.randn(2, 2, 8, 8, generator=g).requires_grad_() * scale
    q, k = [torch.randn(2, 2, 4, 8, generator=g) for _ in range(2)]
    v = torch.randn(2, 2, 4, 8, generator=g) * scale
    beta = torch.rand(2, 2, 4, generator=g)
    write = torch.tensor([[False, True, True, False], [True, False, True, True]])
    if empty: write.zero_()
    candidate, w = correct(previous, k, v, beta, write, cfg.alpha_s, cfg.eps)
    read = torch.ones_like(write)
    grad = torch.zeros(2, 2, 3)
    stats = clean_anchor_stats(previous, previous, candidate, q, k, v, beta, w, read, write, grad, cfg)
    return cfg, previous, candidate, k, v, w, write, stats


def test_clean_metrics_use_masks_and_preserve_graph_values():
    cfg, old, new, k, v, w, write, stats = features()
    assert stats["write_tokens_per_branch"] == [2, 3]
    assert stats["beta"]["count"] == 10
    assert stats["innovation_loss"] == pytest.approx(innovation_loss(old, k, v, w, write).mean().item())
    assert stats["predict_norm_relative_change"] == 0
    assert len(stats["per_head"]["state_rms"]) == 2
    assert all(len(row) == 2 for row in stats["per_head"]["state_rms"])
    json.dumps(stats, allow_nan=False)  # no tensors/graphs retained, JSON finite
    assert new.requires_grad
    assert torch.autograd.grad(new.sum(), old)[0].abs().sum() > 0


def test_relative_innovation_separates_feature_scale_from_prediction_error():
    a, b = features()[-1], features(scale=7.)[-1]
    assert b["innovation_loss"] == pytest.approx(49 * a["innovation_loss"], rel=2e-6)
    torch.testing.assert_close(torch.tensor(a["per_head"]["innovation_relative"]),
                               torch.tensor(b["per_head"]["innovation_relative"]), rtol=2e-6, atol=1e-6)


def test_progress_summary_excludes_prefill_baseline_but_preserves_full_log_rows():
    def chunk(index, value, prefill):
        return {"chunk": index, "start": 0, "end": 1, "anchors": [
            {"block": 19, "prefill": prefill, "per_head": {"innovation_relative": [[value]]}}]}
    record = {"step": 1, "stage": "C", "train_scope": "dit", "config": {"camera_attention": "sana"},
              "loss": .4, "outer_grad_norm": .5, "seconds": 1., "ranks": [
                  {"rank": 0, "prefill": chunk(-1, 1., True), "chunks": [chunk(0, .5, False)]}]}
    assert len(list(stability_rows(record))) == 2
    assert "max clean innovation/V=0.5@19" in progress_line(record, 500)


def test_empty_writes_have_zero_active_eta_and_undefined_relative_error():
    _, old, new, *_, stats = features(empty=True)
    torch.testing.assert_close(old, new, rtol=0, atol=0)
    assert stats["innovation_loss"] == 0 and stats["beta"]["mean"] is None
    assert stats["per_head"]["eta_s"] == [[0., 0.], [0., 0.]]
    assert stats["per_head"]["innovation_relative"] == [[None, None], [None, None]]
    json.dumps(stats, allow_nan=False)


def test_commit_metrics_report_actual_head_clipping_and_independent_branches():
    stats = features()[-1]
    grad = torch.tensor([[[3., 4., 0.], [0., 0., 0.]], [[.3, .4, 0.], [0., 0., 0.]]])
    scale = (1 / grad.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp_max(1)
    old = torch.zeros_like(grad)
    new = old - .01 * grad * scale
    record = committed_anchor_stats(stats, old, new, grad, scale, old, 19, False)
    assert record["inner_clipped_fraction"] == .25
    assert record["per_head"]["inner_grad_norm"][0][0] == 5
    assert record["per_head"]["inner_grad_clipped_norm"][0][0] == 1
    assert record["per_head"]["psi_update_norm"][0][0] == pytest.approx(.01)
    assert record["per_head"]["psi_update_norm"][1][0] == pytest.approx(.005)
    record = {"step": 51, "stage": "C", "train_scope": "dit", "config": {"camera_attention": "sana"},
              "loss": .4, "outer_grad_norm": .5, "seconds": 150.,
              "ranks": [{"rank": 0, "peak_allocated_bytes": 2**30, "chunks": [
                  {"chunk": 0, "start": 0, "end": 4, "anchors": [record]}]}]}
    row = next(stability_rows(record))
    assert (row["step"], row["rank"], row["chunk"], row["block"]) == (51, 0, 0, 19)
    assert "step 51/500" in progress_line(record, 500)


def test_telemetry_preserves_full_stage_c_tbptt_gradients_updates_and_states(monkeypatch):
    import copy
    from test_training import TinyWorldModel, inputs
    from worldttn import anchor
    from worldttn.checkpoint import make_optimizer
    from worldttn.training import train_clip, linear_flow_loss
    torch.manual_seed(17)
    baseline = TinyWorldModel("C")
    measured = copy.deepcopy(baseline)
    clean, noise, timesteps, camera = inputs()
    def update(model):
        return train_clip(model, clean, torch.zeros(1, 1, 2, 8), camera, make_optimizer(model),
                          linear_flow_loss, timesteps, noise, width=100, height=100, tbptt=2, activation_offload="cpu")
    a = update(measured)
    monkeypatch.setattr(anchor, "clean_anchor_stats", lambda *args: {})
    b = update(baseline)
    assert a["loss"] == b["loss"] and a["outer_grad_norm"] == b["outer_grad_norm"]
    for left, right in zip(measured.parameters(), baseline.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        if left.grad is not None: torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    torch.testing.assert_close(a["runtime"].world_state, b["runtime"].world_state, rtol=0, atol=0)
    torch.testing.assert_close(a["runtime"].transition_fast, b["runtime"].transition_fast, rtol=0, atol=0)
    assert a["prefill"]["prefill"] and [c["chunk"] for c in a["chunks"]] == list(range(4))


def test_write_direction_distinguishes_raw_from_transported_basis_and_zero_writes():
    from worldttn.stability import write_direction_stats, state_dynamics
    previous = torch.tensor([[[[1., 0.], [0., 0.]]]])
    rotation = torch.tensor([[0., -1.], [1., 0.]])
    current = (previous @ rotation).requires_grad_()
    stats = write_direction_stats(current, ((previous, previous @ rotation),), 1e-6)
    assert stats["write_direction"]["lag_1"] == {"cosine_raw": [[0.]], "cosine_transport_aligned": [[1.]]}
    empty = write_direction_stats(current * 0, ((previous, previous),), 1e-6)
    assert empty["write_direction"]["lag_1"]["cosine_raw"] == [[None]]
    spectrum = state_dynamics(previous * 0, previous * 0, previous, 1e-6)
    assert spectrum["state_spectrum"]["correction"]["stable_rank"] == [[1.]]
    assert spectrum["state_spectrum"]["correction"]["top1_energy_fraction"] == [[1.]]
    json.dumps(stats, allow_nan=False)


def test_runtime_write_history_is_detached_bounded_transport_aligned_and_reset(monkeypatch):
    from test_runtime import config, begin
    from worldttn import runtime as module
    cfg = config("B")
    system = module.TTNSystem(cfg)
    runtime = module.TTNRuntimeState.create(cfg, 1, "cpu")
    rotation = torch.eye(8)
    rotation[:2, :2] = torch.tensor([[0., -1.], [1., 0.]])
    class Factors:
        def __init__(self, *args): pass
        def right(self, value): return value @ rotation
    monkeypatch.setattr(module, "CayleyFactors", Factors)
    write = torch.zeros_like(runtime.world_state)
    write[..., 0, 0] = 1.
    for step in range(6):
        context = begin(runtime, system, [step], clean=True)
        delta = write @ torch.linalg.matrix_power(rotation, step)
        for i in range(5): context.stage(i, context.predicted[:, i] + delta[:, i], context.psi[:, i], {})
        runtime.commit_chunk(context)
        assert len(runtime.write_history) == min(step + 1, 4)
        assert all(not value.requires_grad for pair in runtime.write_history for value in pair)
        if step:
            stats = runtime.last_stats["anchors"][0]["write_direction"]["lag_1"]
            assert stats["cosine_transport_aligned"] == [[1., 1.]]
            assert stats["cosine_raw"] == [[0., 0.]]
    runtime.reset()
    assert runtime.write_history == ()
