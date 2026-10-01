import torch
import pytest
from test_training import TinyWorldModel, inputs
from worldttn.session import TTNSession
from worldttn.geometry import world_to_ray_mats
from worldttn.controller import TransitionController
from worldttn.core import TTNConfig, ANCHORS
from worldttn.checkpoint import make_optimizer
from worldttn.training import train_clip, linear_flow_loss
from worldttn.anchor import TTNAnchor
from worldttn.core import CayleyFactors, correct, innovation_loss
from worldttn.runtime import TTNSystem, TTNRuntimeState
from test_anchor import ProjectionContract


def test_ray_geometry_preserves_fp32_under_bf16_autocast():
    rays = torch.randn(1, 2, 3, 4, 3)
    pose = torch.eye(4).expand(1, 2, 4, 4)
    expected = world_to_ray_mats(rays, pose)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = world_to_ray_mats(rays, pose)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_camera_model_scaling_preserves_controller_input():
    model = TinyWorldModel()
    _, _, _, camera = inputs()
    camera = camera.clone()
    camera[..., 16:] = torch.tensor([800., 600., 500., 300.])
    s = TTNSession(model, camera, 1000, 600)
    original = camera.clone()
    projected = s.camera_for_model(0, 4, 10, 6, 2)
    assert projected.shape == (2, 4, 20)
    assert torch.equal(projected[0, :, 16:], torch.tensor([8., 6., 5., 3.]).expand(4, 4))
    assert torch.equal(camera, original)
    assert torch.equal(s.camera, original)


@pytest.mark.parametrize("prefill", [True, False])
def test_nonfinite_final_prediction_does_not_commit(prefill):
    model = TinyWorldModel()
    clean, noise, t, camera = inputs()
    session = TTNSession(model, camera, 100, 100)
    runtime = session.reset(1)
    if not prefill: session.prefill(clean[:, :, :1], torch.zeros(1, 1, 2, 8))
    before = runtime.world_state.clone()
    commits = runtime.commit_count
    original = model.forward

    def nonfinite(*args, **kwargs):
        output, cache = original(*args, **kwargs)
        return output * float("nan"), cache

    model.forward = nonfinite
    with pytest.raises(FloatingPointError):
        if prefill: session.prefill(clean[:, :, :1], torch.zeros(1, 1, 2, 8))
        else:
            c = session.begin_chunk(0, 4)
            session.clean_forward(clean[:, :, :4], torch.zeros(1, 1, 2, 8), c, [[None] * 10 for _ in model.blocks], 0,
                                  4)
    assert torch.equal(runtime.world_state, before) and runtime.commit_count == commits


def test_controller_ignores_nonfinite_padding_and_keeps_valid_frame_time():
    torch.manual_seed(21)
    controller = TransitionController(TTNConfig(heads=2, head_dim=8, generators=3, stage="B"))
    with torch.no_grad():
        for head in controller.heads:
            head.weight.normal_()
    poses = torch.eye(4).expand(1, 2, 4, 4).clone()
    poses[0, :, 0, 3] = torch.tensor([2., 5.])
    intrinsics = torch.tensor([100., 80., 50., 40.]).expand(1, 2, 4)
    previous = torch.eye(4)[None]
    expected = controller(poses, intrinsics, previous, torch.ones(1, 2, dtype=torch.bool), 100, 80)
    # Padding between the two new frames must not change either the boundary
    # motion or their normalized time coordinates (1/2, 1).
    padded_pose = torch.full((1, 4, 4, 4), float("nan"))
    padded_intrinsics = torch.full((1, 4, 4), float("nan"))
    padded_pose[:, [0, 2]] = poses
    padded_intrinsics[:, [0, 2]] = intrinsics
    valid = torch.tensor([[True, False, True, False]])
    actual = controller(padded_pose, padded_intrinsics, previous, valid, 100, 80)
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    actual.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in controller.parameters())
    empty = controller(padded_pose, padded_intrinsics, previous, valid & False, 100, 80)
    assert torch.isfinite(empty).all() and empty.count_nonzero() == 0


@pytest.mark.parametrize("bad_input", ["mask", "previous_pose"])
def test_controller_rejects_broadcast_camera_inputs(bad_input):
    controller = TransitionController(TTNConfig(heads=2, head_dim=8, generators=3, stage="B"))
    poses = torch.eye(4).expand(2, 3, 4, 4)
    intrinsics = torch.ones(2, 3, 4)
    previous = torch.eye(4).expand(1 if bad_input == "previous_pose" else 2, 4, 4)
    new_mask = torch.ones(2, 1 if bad_input == "mask" else 3, dtype=torch.bool)
    with pytest.raises(ValueError):
        controller(poses, intrinsics, previous, new_mask, 100, 80)


@pytest.mark.parametrize("stage", ["A", "B", "C"])
def test_stage_training_reaches_replacements_and_keeps_backbone_frozen(stage):
    model = TinyWorldModel(stage)
    clean, noise, t, camera = inputs()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    result = train_clip(model, clean, torch.zeros(1, 1, 2, 8), camera,
                        make_optimizer(model), linear_flow_loss, t, noise,
                        width=100, height=100, tbptt=2)
    for index in ANCHORS:
        anchor = model.blocks[index].attn
        assert anchor.qkv.weight.grad.norm() > 0
        assert anchor.beta_proj.weight.grad.norm() > 0
        assert anchor.output_gate.weight.grad.norm() > 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None and torch.equal(parameter, before[name])
    slow = model.ttn_system
    assert all(p.requires_grad == (stage != "A") for p in slow.parameters())
    if stage == "A":
        assert all(p.grad is None for p in slow.parameters())
    else:
        assert sum(float(head.weight.grad.norm()) for head in slow.controller.heads) > 0
    psi = result["runtime"].transition_fast
    assert not psi.requires_grad
    assert (psi.count_nonzero() > 0) if stage == "C" else (psi.count_nonzero() == 0)


def test_online_psi_step_learns_transition_under_no_grad_with_fixed_slow_weights():
    torch.manual_seed(25)
    config = TTNConfig(heads=2, head_dim=8, generators=3, stage="C")
    system = TTNSystem(config)
    runtime = TTNRuntimeState.create(config, 1, "cpu")
    runtime.world_state.normal_(std=.2)
    previous_state = runtime.world_state.clone()
    anchors = [TTNAnchor(ProjectionContract(), index, config) for index in range(5)]
    modules = [system, *anchors]
    before = [[p.detach().clone() for p in module.parameters()] for module in modules]
    poses = torch.eye(4).expand(1, 2, 4, 4)
    intrinsics = torch.tensor([100., 100., 50., 50.]).expand(1, 2, 4)
    with torch.no_grad():
        context = runtime.begin_chunk(system, poses, intrinsics, torch.tensor([[0, 1]]),
                                      torch.ones(1, 2, dtype=torch.bool), 100, 100).for_clean()
        fixed_features = []
        for anchor in anchors:
            x = torch.randn(1, 6, 16)
            anchor(x, HW=(2, 1, 3), ttn_chunk_context=context)
            _, k, v = anchor.visual_features(x, None)
            beta = anchor.beta_proj(x).sigmoid().transpose(1, 2)
            _, write = context.token_masks(6)
            _, weights = correct(context.predicted[:, anchor.index], k, v, beta, write)
            fixed_features.append((k, v, weights, write))
        frozen_prediction = context.predicted.clone()
        old_psi = context.psi.clone()
        runtime.commit_chunk(context)
        assert runtime.transition_fast.norm() > 0
        # Adapt is lagged: committing cannot alter this chunk's prediction.
        assert torch.equal(context.predicted, frozen_prediction)
        assert torch.equal(context.psi, old_psi)
        next_context = runtime.begin_chunk(system, poses, intrinsics, torch.tensor([[2, 3]]),
                                           torch.ones(1, 2, dtype=torch.bool), 100, 100)
        next_expected = CayleyFactors(system.generators.u, system.generators.v,
                                      next_context.cbase + config.delta_psi * runtime.transition_fast.tanh()).right(
                                          runtime.world_state)
        next_unadapted = CayleyFactors(system.generators.u, system.generators.v,
                                       next_context.cbase).right(runtime.world_state)
        assert torch.allclose(next_context.predicted, next_expected, atol=1e-7, rtol=1e-6)
        assert (next_context.predicted - next_unadapted).norm() > 0
        adapted = CayleyFactors(system.generators.u, system.generators.v,
                                context.cbase + config.delta_psi * runtime.transition_fast.tanh()).right(previous_state)
        for index, (k, v, weights, write) in enumerate(fixed_features):
            original_loss = innovation_loss(context.predicted[:, index], k, v, weights, write)
            adapted_loss = innovation_loss(adapted[:, index], k, v, weights, write)
            assert (adapted_loss < original_loss).all()
    for module, saved in zip(modules, before):
        for parameter, expected in zip(module.parameters(), saved):
            assert parameter.grad is None and torch.equal(parameter, expected)
