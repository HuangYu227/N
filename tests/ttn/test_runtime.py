import pytest
import torch
from worldttn.core import TTNConfig
from worldttn.controller import TransitionController, se3_log
from worldttn.runtime import TTNSystem, TTNRuntimeState


def config(stage="C"):
    return TTNConfig(heads=2, head_dim=8, generators=3, stage=stage)


def poses(b, f):
    p = torch.eye(4).expand(b, f, 4, 4).clone()
    p[..., 0, 3] = torch.arange(f)
    return p


def begin(runtime, system, ids, clean=False):
    ids = torch.tensor(ids).expand(runtime.world_state.shape[0], -1)
    p = poses(ids.shape[0], ids.shape[1])
    intr = torch.tensor([100., 100., 50., 50.]).expand(ids.shape[0], ids.shape[1], 4)
    c = runtime.begin_chunk(system, p, intr, ids, torch.ones_like(ids, dtype=torch.bool), 100, 100)
    return c.for_clean() if clean else c


def stage_all(c):
    for i in range(5):
        c.stage(i, c.predicted[:, i] + 1, torch.ones_like(c.psi[:, i]) * .01, {"write_tokens": int(c.write_mask.sum())})


def test_clean_commit_is_atomic_and_temporary_does_not_mutate():
    system = TTNSystem(config())
    runtime = TTNRuntimeState.create(config(), 2, "cpu")
    context = begin(runtime, system, [0, 1])
    before = runtime.world_state.clone()
    with pytest.raises(RuntimeError):
        context.stage(0, context.predicted[:, 0], context.psi[:, 0], {})
    clean = context.for_clean()
    clean.stage(0, clean.predicted[:, 0] + 1, clean.psi[:, 0], {})
    with pytest.raises(RuntimeError):
        runtime.commit_chunk(clean)
    assert torch.equal(runtime.world_state, before) and runtime.commit_count == 0
    clean = context.for_clean()
    stage_all(clean)
    runtime.commit_chunk(clean)
    assert runtime.commit_count == 1 and runtime.world_state[0, 0, 0, 0, 0] == 1
    assert runtime.transition_fast.norm() > 0 and not runtime.transition_fast.requires_grad
    with pytest.raises(RuntimeError):
        runtime.commit_chunk(clean)


def test_prefill_and_overlap_only_write_once_reset_isolated_branches():
    cfg = config()
    system = TTNSystem(cfg)
    runtime = TTNRuntimeState.create(cfg, 2, "cpu")
    clean = begin(runtime, system, [0], True)
    stage_all(clean)
    runtime.prefill(clean)
    assert runtime.prefilled
    with pytest.raises(RuntimeError):
        runtime.prefill(clean)
    c = begin(runtime, system, [0, 1, 2])
    assert c.write_mask.tolist() == [[False, True, True], [False, True, True]]
    assert c.read_mask.all()
    clean = c.for_clean()
    stage_all(clean)
    runtime.commit_chunk(clean)
    assert runtime.committed_frame_ids == [{0, 1, 2}, {0, 1, 2}]
    runtime.world_state[0].fill_(7)
    assert runtime.world_state[1].max() != 7
    runtime.reset()
    assert runtime.world_state.count_nonzero() == 0 and not runtime.prefilled


def test_invalid_candidate_never_partially_commits():
    cfg = config()
    system = TTNSystem(cfg)
    runtime = TTNRuntimeState.create(cfg, 1, "cpu")
    c = begin(runtime, system, [0], True)
    stage_all(c)
    c.candidates[4] = (torch.full_like(c.predicted[:, 4], float("nan")), c.psi[:, 4], {})
    with pytest.raises(ValueError):
        runtime.commit_chunk(c)
    assert runtime.world_state.count_nonzero() == 0


def test_controller_keeps_boundary_motion_and_masks_invalid_frames():
    torch.manual_seed(9)
    ctrl = TransitionController(config("B"))
    previous = torch.eye(4).expand(1, 4, 4).clone()
    p = poses(1, 3)
    p[:, 0, 0, 3] = 2
    p[:, 1, 0, 3] = 3
    p[:, 2, 0, 3] = 1e6
    intr = torch.tensor([100., 80., 50., 40.]).expand(1, 3, 4)
    mask = torch.tensor([[True, True, False]])
    features = ctrl.features(p, intr, previous, mask, 100, 80)
    assert features.shape == (1, 260)
    assert torch.allclose(features[0, -4:], torch.tensor([1., 1., .5, .5]))
    with torch.no_grad():
        ctrl.heads[0].weight.normal_()
    a = ctrl(p, intr, previous, mask, 100, 80)
    p[:, 2, 0, 3] = -1e6
    assert torch.equal(a, ctrl(p, intr, previous, mask, 100, 80))
    previous[:, 0, 3] = 1
    assert not torch.allclose(a, ctrl(p, intr, previous, mask, 100, 80))
    assert ctrl(p, intr, previous, mask * False, 100, 80).count_nonzero() == 0


@pytest.mark.parametrize("angle", [0., 1e-8, .7, 3.141592653589793])
def test_se3_log_rotation_and_translation(angle):
    t = torch.eye(4, dtype=torch.float64)
    c = torch.tensor(angle, dtype=t.dtype).cos()
    s = torch.tensor(angle, dtype=t.dtype).sin()
    t[0, 0] = c
    t[1, 1] = c
    t[0, 1] = -s
    t[1, 0] = s
    out = se3_log(t)
    assert torch.isfinite(out).all()
    assert torch.allclose(out[3:], torch.tensor([0., 0., angle], dtype=t.dtype), atol=1e-10, rtol=0)


def test_runtime_detach_cuts_only_window_boundary():
    cfg = config("B")
    sys = TTNSystem(cfg)
    r = TTNRuntimeState.create(cfg, 1, "cpu")
    old = torch.randn_like(r.world_state, requires_grad=True)
    r.world_state = old
    c = begin(r, sys, [0], True)
    for i in range(5):
        c.stage(i, c.predicted[:, i] * 2, c.psi[:, i], {})
    r.commit_chunk(c)
    r.world_state.sum().backward()
    assert old.grad.abs().sum() > 0
    r.detach()
    assert not r.world_state.requires_grad
