import copy
import subprocess
import sys
import torch
import pytest
from test_anchor import ProjectionContract
from test_training import TinyWorldModel, inputs
from worldttn.anchor import TTNAnchor
from worldttn.core import TTNConfig, CayleyFactors, cayley_dense
from worldttn.runtime import TTNSystem, TTNRuntimeState
from worldttn.session import TTNSession, carry_cache
from worldttn.controller import se3_log


def test_real_head_dimensions_lowrank_dense_fp64():
    torch.manual_seed(17)
    u = torch.nn.functional.normalize(torch.randn(20, 16, 112, dtype=torch.float64), dim=-1)
    v = torch.nn.functional.normalize(torch.randn_like(u), dim=-1)
    c = torch.randn(20, 16, dtype=u.dtype) * .05
    x = torch.randn(20, 112, 112, dtype=u.dtype)
    low = CayleyFactors(u, v, c).right(x)
    assert torch.allclose(low, x @ cayley_dense(u, v, c), atol=1e-10, rtol=0)


def test_empty_write_skips_adapt_and_preserves_prediction():
    cfg = TTNConfig(heads=2, head_dim=8, generators=3, stage="C")
    sys = TTNSystem(cfg)
    r = TTNRuntimeState.create(cfg, 1, "cpu")
    r.world_state = torch.randn_like(r.world_state)
    r.transition_fast = torch.randn_like(r.transition_fast)
    r.committed_frame_ids = [{0, 1}]
    cam = torch.eye(4).expand(1, 2, 4, 4)
    intr = torch.tensor([100., 100., 50., 50.]).expand(1, 2, 4)
    c = r.begin_chunk(sys, cam, intr, torch.arange(2)[None], torch.ones(1, 2, dtype=torch.bool), 100, 100).for_clean()
    a = TTNAnchor(ProjectionContract(), 0, cfg)
    out = a(torch.randn(1, 6, 16), HW=(2, 1, 3), ttn_chunk_context=c)
    s, g, stats = c.candidates[0]
    assert torch.equal(s, c.predicted[:, 0]) and g.count_nonzero() == 0
    assert torch.isfinite(out).all() and stats["write_tokens"] == 0


def test_clean_forward_failure_after_anchor_leaves_entire_runtime_unchanged():
    model = TinyWorldModel("C")
    clean, noise, t, camera = inputs()
    session = TTNSession(model, camera, 100, 100)
    session.reset(1)
    session.prefill(clean[:, :, :1], torch.zeros(1, 1, 2, 8))
    context = session.begin_chunk(0, 4)
    before = session.runtime.world_state.clone()
    psi = session.runtime.transition_fast.clone()

    def fail(*args):
        raise RuntimeError("downstream block failed")

    hook = model.blocks[15].attn.register_forward_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="downstream"):
            session.clean_forward(clean[:, :, :4], torch.zeros(1, 1, 2, 8), context,
                                  [[None] * 10 for _ in model.blocks], 0, 4)
    finally:
        hook.remove()
    assert torch.equal(session.runtime.world_state, before)
    assert torch.equal(session.runtime.transition_fast, psi)
    assert session.runtime.committed_frame_ids == [{0}]
    assert session.runtime.commit_count == 1


def test_prefill_discards_scratch_and_anchor_ffn_cache_is_carried():
    model = TinyWorldModel("A")
    clean, noise, t, camera = inputs()
    session = TTNSession(model, camera, 100, 100)
    session.reset(1)
    session.prefill(clean[:, :, :1], torch.zeros(1, 1, 2, 8))
    cache = [[None] * 10 for _ in model.blocks]
    c = session.begin_chunk(0, 4)
    _, cache = session.clean_forward(clean[:, :, :4], torch.zeros(1, 1, 2, 8), c, cache, 0, 4)
    assert cache[3][:9] == [None] * 9 and cache[3][9] is not None
    carried = carry_cache(cache)
    assert carried[3][9] is cache[3][9]
    assert carried[0][9] is cache[0][9]


def test_runtime_inner_clips_each_head_independently():
    cfg = TTNConfig(heads=2, head_dim=8, generators=3, stage="C", eta_psi=.01, inner_clip=1.)
    sys = TTNSystem(cfg)
    r = TTNRuntimeState.create(cfg, 1, "cpu")
    cam = torch.eye(4).expand(1, 1, 4, 4)
    intr = torch.tensor([100., 100., 50., 50.]).expand(1, 1, 4)
    c = r.begin_chunk(sys, cam, intr, torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.bool), 100,
                      100).for_clean()
    for i in range(5):
        grad = torch.tensor([[[3., 4., 0.], [.3, .4, 0.]]])
        c.stage(i, c.predicted[:, i], grad, {})
    r.commit_chunk(c)
    assert torch.allclose(r.transition_fast[0, 0, 0], torch.tensor([-.006, -.008, 0.]))
    assert torch.allclose(r.transition_fast[0, 0, 1], torch.tensor([-.003, -.004, 0.]))


@pytest.mark.parametrize("angle", [3.13, torch.pi - 1e-6, torch.pi])
def test_se3_log_near_pi_general_axis(angle):
    axis = torch.tensor([1., 2., 3.], dtype=torch.float64)
    axis = axis / axis.norm()
    a = torch.tensor([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]], dtype=axis.dtype)
    r = torch.matrix_exp(a * angle)
    t = torch.eye(4, dtype=axis.dtype)
    t[:3, :3] = r
    log = se3_log(t)
    assert torch.allclose(log[3:], axis * angle, atol=1e-9, rtol=0)


def test_cli_help_is_cpu_importable_without_sana_dependencies():
    result = subprocess.run([sys.executable, "-m", "worldttn.cli", "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--base-weights" in result.stdout


def test_checkpoint_rejects_changed_base_hash_or_generator_shape(tmp_path):
    from worldttn.checkpoint import save_checkpoint, load_checkpoint
    m = TinyWorldModel("A")
    m.base_load_report = {"sha256": "teacherA"}
    path = tmp_path / "adapter.pt"
    save_checkpoint(path, m, None, 0)
    n = TinyWorldModel("A")
    n.base_load_report = {"sha256": "teacherB"}
    with pytest.raises(ValueError, match="SHA256"):
        load_checkpoint(path, n)
    n.base_load_report = {"sha256": "teacherA"}
    payload = torch.load(path, weights_only=False)
    payload["adapter"]["ttn_system.generators.u"] = torch.zeros(1)
    torch.save(payload, path)
    old = n.blocks[3].attn.qkv.weight.detach().clone()
    with pytest.raises(ValueError, match="keys/shapes"):
        load_checkpoint(path, n)
    assert torch.equal(old, n.blocks[3].attn.qkv.weight)
