"""BF16 prefill must not poison trainable projections through the AMP cache."""
import pytest
import torch
from test_training import TinyWorldModel, inputs
from worldttn.checkpoint import make_optimizer
from worldttn.core import ANCHORS
from worldttn.training import train_clip, linear_flow_loss
from worldttn.training_health import FirstUpdateProbe


def camera_branch(module, args, kwargs):
    # Identity geometry isolates real camera projections/mixing from CUDA helpers.
    kwargs["camera_conditions"] = args[0].new_zeros(1, 1, 20)
    kwargs["prope_fns"] = (lambda x: x,) * 3
    return args, kwargs


@pytest.mark.parametrize("stage", ["A", "B", "C"])
@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("activation_offload", ["none", "cpu"])
def test_bf16_prefill_then_training_updates_visual_and_camera_projections(stage, k, activation_offload):
    torch.manual_seed(23)
    model = TinyWorldModel(stage)
    for i in ANCHORS:
        model.blocks[i].attn.register_forward_pre_hook(camera_branch, with_kwargs=True)
    optimizer = make_optimizer(model)
    clean, noise, t, camera = inputs()
    probe = FirstUpdateProbe(model)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = train_clip(model, clean, torch.zeros(1, 1, 2, 8), camera, optimizer,
                            linear_flow_loss, t, noise, width=100, height=100, tbptt=k,
                            activation_offload=activation_offload)
    health = probe.report(optimizer)
    assert not health["missing_core_gradients"]
    for i in ANCHORS:
        for name in ("qkv", "proj", "output_gate", "beta_proj", "q_proj_cam",
                     "k_proj_cam", "v_proj_cam", "out_proj_cam"):
            group = health["groups"][f"blocks.{i}.attn.{name}"]
            assert group["grad_norm"] > 0, (stage, k, i, name)
            assert group["delta_norm"] > 0 and group["changed_elements"] > 0
            assert group["optimizer_state_parameters"] > 0
    assert result["runtime"].commit_count == result["runtime"].predict_count == 5
    assert result["runtime"].committed_frame_ids == [set(range(13))]
    assert not torch.is_autocast_enabled("cpu")
    assert torch.is_autocast_cache_enabled()


def test_prefill_preserves_enclosing_amp_settings_and_no_grad_boundary():
    model = TinyWorldModel("C")
    clean, noise, t, camera = inputs()
    settings = []

    def capture(module, args, kwargs):
        settings.append((kwargs["ttn_chunk_context"].prefill_mode,
                         torch.is_grad_enabled(), torch.is_autocast_enabled("cpu"),
                         torch.get_autocast_dtype("cpu"), torch.is_autocast_cache_enabled()))

    model.blocks[3].attn.register_forward_pre_hook(capture, with_kwargs=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        train_clip(model, clean, torch.zeros(1, 1, 2, 8), camera, make_optimizer(model),
                   linear_flow_loss, t, noise, width=100, height=100, tbptt=2)
        assert torch.is_autocast_enabled("cpu") and torch.is_autocast_cache_enabled()
    assert settings[0] == (True, False, True, torch.bfloat16, False)
    assert all(row == (False, True, True, torch.bfloat16, True) for row in settings[1:])
