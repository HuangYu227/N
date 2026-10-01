import torch
import pytest
from test_training import TinyWorldModel, inputs
from worldttn.session import TTNSession
from worldttn.geometry import world_to_ray_mats


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
