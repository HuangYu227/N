from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_alignment_detail import cached_sana
from test_frame_memory import config
from test_sink import actual_rope, begin
from test_sequence_training import sequence_model, batch
from test_activation_checkpoint import cached_ffn, text_attention
from worldttn.anchor import TTNAnchor
from worldttn.core import ANCHORS
from worldttn.runtime import TTNRuntimeState, TTNSystem
from worldttn.spectral_readout import (SpectralReadout, spatial_lowpass, spatial_stats,
                                      install_spectral_readout)


def test_filter_projection_identity_energy_and_frame_isolation():
    torch.manual_seed(91)
    value = torch.randn(2, 3*8*10, 16, dtype=torch.float64)
    weight = torch.randn(12, 16, dtype=torch.float64)
    filtered = spatial_lowpass(value, (3, 8, 10))
    torch.testing.assert_close(filtered @ weight.T,
        spatial_lowpass(value @ weight.T, (3, 8, 10)), rtol=1e-12, atol=1e-12)
    changed = value.clone(); changed[:, -80:] += 100
    torch.testing.assert_close(spatial_lowpass(changed, (3, 8, 10))[:, :160], filtered[:, :160], rtol=0, atol=0)
    constant = torch.ones_like(value)*3
    torch.testing.assert_close(spatial_lowpass(constant, (3, 8, 10)), constant)
    stats = spatial_stats(value, (3, 8, 10))
    energy = sum(torch.tensor(stats[k]) for k in ("dc_energy", "low_non_dc_energy", "higher_energy"))
    torch.testing.assert_close(energy, value.float().reshape(2, 3, 8, 10, 16).square().mean((2, 3, 4)))


def test_reference_isolation_failed_capture_reset_and_zero_bypass():
    state = torch.randn(2, 5, 2, 8, 8)
    runtime = SimpleNamespace(world_state=state, prefilled=True, commit_count=1)
    control = SpectralReadout("low-reference")
    with torch.no_grad():
        control.capture(runtime)
        assert control.reference.data_ptr() != state.data_ptr()
        saved = control.reference.clone()
        state.fill_(99)
        torch.testing.assert_close(saved, control.reference, rtol=0, atol=0)
        with pytest.raises(ValueError, match="exactly one"): control.capture(runtime)
        control.reset()
        assert control.reference is None and control.calls == []
        runtime.world_state.fill_(float("nan"))
        with pytest.raises(ValueError, match="invalid observed"): control.capture(runtime)
        assert control.reference is None and control.reference_hash is None
        model = nn.Module(); model.ttn_system = SimpleNamespace(config=config())
        model.blocks = nn.ModuleList([nn.Module() for _ in range(20)])
        for block in model.blocks: block.attn = nn.Identity()
        model.eval()
        zero = SpectralReadout("low-reference", gain=0)
        install_spectral_readout(model, zero)
        assert model._ttn_spectral_readout is None
        assert all(model.blocks[i].attn._ttn_spectral_readout is None for i in ANCHORS)


def test_isolated_first_chunk_uses_outer_noisy_context_but_keeps_observed_frame(sequence_model, tmp_path):
    from worldttn.session import TTNSession
    model = sequence_model.eval()
    model.rope = actual_rope()
    clean, _, times, camera = batch()
    clean, times, camera = clean[:, :, :4], times[:, :, :4], camera[:, :4]
    y = torch.randn(1, 1, 2, 16)
    def generate(control):
        if control is not None: install_spectral_readout(model, control)
        session = TTNSession(model, camera, 100, 100)
        session.reset(1)
        session.prefill(clean[:, :, :1], y)
        context = session.begin_chunk(0, 4)
        if control is not None: control.begin_call(context, torch.tensor([[[0., 1., 1., 1.]]]), 0, 4)
        return session.forward(clean, times, y, context, [[None]*10 for _ in model.blocks], 0, 4)[0]
    with torch.no_grad():
        original = generate(None)
        baseline = SpectralReadout("baseline")
        torch.testing.assert_close(generate(baseline), original, rtol=0, atol=0)
        control = SpectralReadout("low-reference", diagnostics=tmp_path/"first.jsonl")
        changed = generate(control)
        torch.testing.assert_close(changed[:, :, :1], original[:, :, :1], rtol=0, atol=0)
        assert not torch.equal(changed[:, :, 1:], original[:, :, 1:])
        assert control.verify()["nonzero_quantized_anchors"] == list(ANCHORS)
        import json
        rows = [json.loads(line) for line in control.path.read_text().splitlines()]
        assert all(row["private_candidate"] and row["frame_ids"] == [[1, 2, 3]] for row in rows)


@pytest.mark.parametrize("mode", ["baseline", "low-zero", "low-reference", "full-reference"])
def test_actual_frame_anchor_gate_camera_masks_sigma_and_clean_candidate(cached_sana, mode, tmp_path):
    torch.manual_seed(33)
    cfg = config(memory_selection="none")
    anchor = TTNAnchor(cached_sana(), 0, cfg).eval()
    system = TTNSystem(cfg)
    with torch.no_grad():
        runtime = TTNRuntimeState.create(cfg, 2, "cpu")
        runtime.prefilled, runtime.commit_count = True, 1
        runtime.world_state.normal_()
        control = SpectralReadout(mode, diagnostics=tmp_path/"stats.jsonl")
        control.capture(runtime)
        ctx = begin(runtime, system, [0, 1, 2], rope=actual_rope())
        x = torch.randn(2, 12, 16)
        kwargs = dict(HW=(3, 2, 2), camera_conditions=torch.zeros(2, 3, 20),
                      prope_fns=(lambda x: x,)*3, ttn_chunk_context=ctx, kv_cache=[None]*10)
        baseline = anchor(x, **kwargs)[0]
        anchor._ttn_spectral_readout = control
        sigma = torch.tensor([[[0., .9, .9]], [[0., .7, .7]]])
        control.begin_call(ctx, sigma, 0, 20)
        corrected = anchor(x, **kwargs)[0]
        torch.testing.assert_close(corrected[1], baseline[1], rtol=0, atol=0)
        torch.testing.assert_close(corrected[:, :4], baseline[:, :4], rtol=0, atol=0)
        assert not ctx.candidates and runtime.commit_count == 1
        if mode == "baseline": torch.testing.assert_close(corrected, baseline, rtol=0, atol=0)
        else: assert not torch.equal(corrected[0, 4:], baseline[0, 4:])
        control.begin_call(ctx, torch.full_like(sigma, .79), 1, 20)
        torch.testing.assert_close(anchor(x, **kwargs)[0], baseline, rtol=0, atol=0)
        clean = ctx.for_clean()
        kwargs["ttn_chunk_context"] = clean
        out = anchor(x, **kwargs)[0]
        candidate = clean.candidates[0][0].clone()
        anchor._ttn_spectral_readout = None
        clean2 = ctx.for_clean(); kwargs["ttn_chunk_context"] = clean2
        torch.testing.assert_close(anchor(x, **kwargs)[0], out, rtol=0, atol=0)
        torch.testing.assert_close(clean2.candidates[0][0], candidate, rtol=0, atol=0)
        assert control.reference_hash is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires local CUDA")
def test_real_dimensions_cfg_bf16_quantized_output_and_temporal_ridge_covariance():
    device = "cuda"
    torch.manual_seed(41)
    # Real per-anchor dimensions and spatial token grid; two CFG branches.
    b, heads, dim, hw = 2, 20, 112, (3, 22, 40)
    q = torch.randn(b, heads, 2640, dim, device=device)*.1
    reference = torch.randn(b, 5, heads, dim, dim, device=device)*.01
    ctx = SimpleNamespace(clean_mode=False, prefill_mode=False,
        frame_ids=torch.tensor([[4, 5, 6]]*2, device=device), read_mask=torch.ones(b, 3, dtype=torch.bool, device=device),
        memory_rope=actual_rope(dim))
    control = SpectralReadout("low-reference")
    proj = nn.Linear(2240, 2240, device=device).eval()
    fast = torch.randn(b, 2640, 2240, device=device)
    gate = torch.randn_like(fast).sigmoid()
    camera = torch.randn_like(fast)
    gated = (fast+camera)*gate
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        control.capture(SimpleNamespace(world_state=reference, prefilled=True, commit_count=1))
        control.begin_call(ctx, torch.ones(b, 1, 3, device=device), 0, 20)
        corrected = control.apply(0, ctx, q, fast, gate, gated, hw)
        output = proj(corrected)
        control.finish(0, ctx, output, proj, torch.float32, torch.ones(b, 2640, device=device, dtype=torch.bool))
        assert 3 in control.nonzero_anchors and torch.isfinite(output).all()
        assert control.shift.tolist() == [3, 3]
    # Check the regularized first-frame state's actual time-RoPE covariance.
    from worldttn.geometry import apply_complex_rope
    from worldttn.sink import realign_reference
    key = torch.randn(2, 2, 20, dim, device=device, dtype=torch.float64)
    value = torch.randn_like(key)
    rope = actual_rope(dim)
    solve = lambda k: torch.linalg.solve(k.transpose(-1, -2)@k + .3*torch.eye(dim, device=device), k.transpose(-1, -2)@value)
    rotated = apply_complex_rope(key, rope(((3, 4), 1, 1), torch.device(device)))
    torch.testing.assert_close(realign_reference(solve(key), rope, torch.tensor([3, 3], device=device)),
                               solve(rotated), rtol=1e-10, atol=1e-10)
