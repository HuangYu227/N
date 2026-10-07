"""Readonly reference compression, position covariance and episode isolation."""
import ast
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from test_alignment_detail import cached_sana
from worldttn.anchor import TTNAnchor
from worldttn.core import TTNConfig
from worldttn.geometry import apply_complex_rope
from worldttn.performance import ExecutionOptions
from worldttn.runtime import TTNRuntimeState, TTNSystem


def actual_rope(dim=8):
    """Execute SANA's actual causal forward without optional training imports."""
    path = Path(__file__).resolve().parents[2] / "diffusion/model/nets/sana_blocks.py"
    node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(n, ast.ClassDef) and n.name == "CausalWanRotaryPosEmbed")
    namespace = {"torch": torch, "WanRotaryPosEmbed": nn.Module}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    rope = namespace["CausalWanRotaryPosEmbed"]()
    rope.attention_head_dim = dim
    time = dim // 2 - 2 * (dim // 6)
    rates = torch.cat([torch.linspace(.07, .7, count, dtype=torch.float64)
                       for count in (time, dim // 6, dim // 6)])
    phase = torch.arange(64, dtype=torch.float64)[:, None] * rates
    rope.freqs = torch.polar(torch.ones_like(phase), phase)
    return rope


def config():
    return TTNConfig(heads=2, head_dim=8, generators=3, stage="C", camera_attention="sana",
                     local_update=True, persistent_meta=True)


def begin(runtime, system, ids, *, prefill=False, rope=None, valid=None):
    b = runtime.world_state.shape[0]
    frames = torch.tensor(ids).expand(b, -1)
    poses = torch.eye(4).expand(b, len(ids), 4, 4)
    intr = torch.tensor([100., 100., 50., 50.]).expand(b, len(ids), 4)
    return runtime.begin_chunk(system, poses, intr, frames,
        torch.ones_like(frames, dtype=torch.bool) if valid is None else valid,
        100, 100, prefill=prefill, rope=rope)


def stage_all(ctx, offset=1.):
    for i in range(5):
        state = ctx.predicted[:, i] + offset + torch.arange(ctx.predicted.shape[0])[:, None, None, None]
        ctx.stage(i, state, torch.zeros_like(ctx.psi[:, i]), {"write_tokens": int(ctx.write_mask.sum())})


@pytest.mark.parametrize("dim", [8, 112])
def test_temporal_realign_equals_recompressing_rotated_keys_and_preserves_spatial(dim):
    from worldttn.sink import realign_reference
    torch.manual_seed(17)
    rope = actual_rope(dim)
    k, v = torch.randn(2, 2, 6, dim, dtype=torch.float64), torch.randn(2, 2, 6, dim, dtype=torch.float64)
    reference = k.transpose(-1, -2) @ v
    phase = rope(((11, 12), 1, 1), torch.device("cpu"))
    rotated_k = apply_complex_rope(k, phase)
    aligned = realign_reference(reference, rope, torch.tensor([11, 11]))
    torch.testing.assert_close(aligned, rotated_k.transpose(-1, -2) @ v, rtol=1e-14, atol=1e-14)
    time = rope.attention_head_dim // 2 - 2 * (rope.attention_head_dim // 6)
    torch.testing.assert_close(aligned[..., 2*time:, :], reference[..., 2*time:, :], rtol=0, atol=0)
    assert torch.equal(realign_reference(reference, rope, torch.zeros(2, dtype=torch.long)), reference)
    with pytest.raises(ValueError, match="RoPE"):
        realign_reference(reference, None, torch.tensor([1, 1]))


@pytest.mark.parametrize("kwargs", [{"gain": float("nan")}, {"gain": -1}, {"gain": 1.1},
    {"gain": True}, {"mode": "typo"}, {"position": "guess"}, {"start_chunk": 0},
    {"start_chunk": 1.5}, {"start_chunk": True}])
def test_sink_options_reject_invalid_values(kwargs):
    from worldttn.sink import SinkOptions
    with pytest.raises(ValueError): SinkOptions(**kwargs)


def test_sink_cli_legacy_defaults_and_explicit_options():
    from argparse import ArgumentParser, Namespace
    from worldttn.sink import SinkOptions, add_sink_arguments, sink_options_from_args
    parser = ArgumentParser()
    add_sink_arguments(parser)
    assert sink_options_from_args(Namespace()) == SinkOptions()
    assert sink_options_from_args(parser.parse_args(["--tla-sink", "zero", "--sink-gain", ".25",
        "--sink-position", "absolute", "--sink-start-chunk", "2"])) == SinkOptions("zero", .25, "absolute", 2)


def test_reference_captured_once_cfg_isolated_masks_and_fifth_chunk_activation():
    from worldttn.sink import SinkOptions
    cfg = config()
    system = TTNSystem(cfg)
    with torch.no_grad():
        runtime = TTNRuntimeState.create(cfg, 2, "cpu", sink_options=SinkOptions("protected"))
        pre = begin(runtime, system, [0], prefill=True).for_clean()
        stage_all(pre)
        runtime.prefill(pre)
        initial = runtime.sink_reference.clone()
        assert runtime.sink_reference.data_ptr() != runtime.world_state.data_ptr()
        assert not torch.equal(initial[0], initial[1])
        for ordinal, ids in enumerate(([0, 1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]), 1):
            ctx = begin(runtime, system, ids, rope=actual_rope())
            assert (ctx.sink_reference is not None) == (ordinal >= 5)
            if ordinal == 1:
                assert ctx.write_mask[:, 0].count_nonzero() == 0
            if ordinal == 5:
                assert ctx.sink_shift.tolist() == [12, 12]
                assert ctx.for_clean().sink_reference is ctx.sink_reference
                assert not ctx.for_clean().candidates
            clean = ctx.for_clean()
            stage_all(clean, ordinal)
            runtime.commit_chunk(clean)
            assert torch.equal(runtime.sink_reference, initial)
        runtime.world_state[0].fill_(123.)
        assert torch.equal(runtime.sink_reference, initial)
        runtime.sink_reference_verified = True
        runtime.reset()
        assert runtime.sink_reference is None and runtime.commit_count == 0
        assert runtime.sink_reference_verified is None


def test_capture_failure_and_incomplete_prefill_leave_previous_state(monkeypatch):
    from worldttn.sink import SinkOptions
    import worldttn.sink as sink
    cfg = config()
    with torch.no_grad():
        runtime = TTNRuntimeState.create(cfg, 1, "cpu", sink_options=SinkOptions("protected"))
        clean = begin(runtime, TTNSystem(cfg), [0], prefill=True).for_clean()
        clean.stage(0, clean.predicted[:, 0]+1, clean.psi[:, 0], {})
        with pytest.raises(RuntimeError, match="five"):
            runtime.prefill(clean)
        stage_all(clean.for_clean())
        clean = clean.for_clean()
        stage_all(clean)
        def failed_capture(*args): raise RuntimeError("reference allocation failed")
        monkeypatch.setattr(sink, "freeze_reference", failed_capture)
        with pytest.raises(RuntimeError, match="allocation"):
            runtime.prefill(clean)
        assert runtime.world_state.count_nonzero() == 0
        assert runtime.commit_count == 0 and not runtime.prefilled and runtime.sink_reference is None


def test_sink_rejects_training_protocols_and_execution_before_forward():
    from worldttn.sink import SinkOptions
    options = SinkOptions("protected", start_chunk=1)
    with pytest.raises(ValueError, match="inference"):
        TTNRuntimeState.create(config(), 1, "cpu", sink_options=options)
    with torch.no_grad():
        for cfg in (replace(config(), camera_attention="linear"),
                    TTNConfig(heads=2, head_dim=8, generators=3, stage="A", camera_attention="sana")):
            with pytest.raises(ValueError, match="Stage C|camera"):
                TTNRuntimeState.create(cfg, 1, "cpu", sink_options=options)
        with pytest.raises(ValueError, match="full"):
            TTNRuntimeState.create(config(), 1, "cpu", sink_options=options, ablation="identity")
        cfg = replace(config(), local_update=False, persistent_meta=False)
        runtime = TTNRuntimeState.create(cfg, 1, "cpu", sink_options=options)
        system = TTNSystem(cfg)
        system.ttn_execution = ExecutionOptions("reuse", "projected")
        with pytest.raises(ValueError, match="reference/reference"):
            begin(runtime, system, [0], prefill=True)
        assert runtime.predict_count == 0


def test_anchor_sink_changes_read_only_masks_padding_and_never_mutates_reference(cached_sana):
    from worldttn.sink import SinkOptions
    cfg = config()
    source = cached_sana()
    anchor = TTNAnchor(source, 0, cfg)
    system = TTNSystem(cfg)
    with torch.no_grad():
        runtime = TTNRuntimeState.create(cfg, 2, "cpu", sink_options=SinkOptions("protected", .25, "absolute", 1))
        pre = begin(runtime, system, [0], prefill=True).for_clean()
        stage_all(pre)
        runtime.prefill(pre)
        ctx = begin(runtime, system, [1, 2], valid=torch.tensor([[True, False], [True, True]]))
        clean = ctx.for_clean()
        x = torch.randn(2, 4, 16)
        seen = {}
        out = anchor(x, HW=(2, 1, 2), ttn_chunk_context=clean,
            ttn_diagnostic=lambda key, value: seen.update({key: value}))
        baseline = clean.for_clean()
        baseline.sink_reference = None
        before = anchor(x, HW=(2, 1, 2), ttn_chunk_context=baseline)
        torch.testing.assert_close(clean.candidates[0][0], baseline.candidates[0][0], rtol=0, atol=0)
        q, _, _ = seen["visual_features"]
        state = clean.candidates[0][0]
        expected_raw = (q @ (.75*state + .25*ctx.sink_reference[:, 0])).transpose(1, 2).reshape_as(x)
        torch.testing.assert_close(seen["visual_raw"], expected_raw, rtol=0, atol=0)
        assert not torch.equal(out, before)
        assert out[0, 2:].count_nonzero() == 0
        baseline_raw = q @ state
        delta = expected_raw.reshape(2, 4, 2, 8).transpose(1, 2)-baseline_raw
        torch.testing.assert_close(torch.tensor(clean.sink_stats[0]["per_head"]["raw_delta_norm"][0]),
                                   delta[0, :, :2].flatten(-2).norm(dim=-1), rtol=1e-6, atol=1e-6)
        initial = runtime.sink_reference.clone()
        for _ in range(3):
            anchor(x, HW=(2, 1, 2), ttn_chunk_context=ctx)
        assert torch.equal(initial, runtime.sink_reference)
        assert runtime.commit_count == 1
        assert clean.sink_stats[0]["per_head"]["effective_delta_norm"]


@pytest.mark.parametrize("mode", ["off", "protected", "zero"])
def test_zero_gain_is_exact_original_path_without_capture_or_rope(mode, cached_sana):
    from worldttn.sink import SinkOptions
    cfg = config()
    with torch.no_grad():
        plain = TTNRuntimeState.create(cfg, 1, "cpu")
        zero = TTNRuntimeState.create(cfg, 1, "cpu", sink_options=SinkOptions(mode, 0.))
        system = TTNSystem(cfg)
        source = cached_sana()
        anchor = TTNAnchor(source, 0, cfg)
        for runtime in (plain, zero):
            pre = begin(runtime, system, [0], prefill=True).for_clean()
            stage_all(pre)
            runtime.prefill(pre)
        assert zero.sink_reference is None
        x = torch.randn(1, 4, 16)
        a, b = (begin(runtime, system, [1, 2]).for_clean() for runtime in (plain, zero))
        torch.testing.assert_close(anchor(x, HW=(2, 1, 2), ttn_chunk_context=a),
                                   anchor(x, HW=(2, 1, 2), ttn_chunk_context=b), rtol=0, atol=0)
        torch.testing.assert_close(a.candidates[0][0], b.candidates[0][0], rtol=0, atol=0)


def test_noise_sink_telemetry_samples_first_middle_last_only():
    from worldttn.sink import SinkOptions
    from worldttn.session import record_noise
    cfg = config()
    with torch.no_grad():
        runtime = TTNRuntimeState.create(cfg, 1, "cpu", sink_options=SinkOptions("zero", start_chunk=1))
        system = TTNSystem(cfg)
        pre = begin(runtime, system, [0], prefill=True).for_clean()
        stage_all(pre)
        runtime.prefill(pre)
        ctx = begin(runtime, system, [1, 2])
        for call in range(6):
            record_noise(ctx, torch.tensor([900.-call*100]), total_calls=6)
        assert [row["call"] for row in ctx.sink_trajectory] == [0, 3, 5]


def test_reference_audit_hash_is_fixed_and_detects_accidental_mutation():
    from worldttn.sink import SinkOptions
    cfg = config()
    with torch.no_grad():
        runtime = TTNRuntimeState.create(cfg, 1, "cpu", sink_options=SinkOptions("protected", position="absolute"))
        system = TTNSystem(cfg)
        pre = begin(runtime, system, [0], prefill=True).for_clean()
        stage_all(pre)
        runtime.prefill(pre)
        assert len(runtime.sink_reference_sha256) == 64
        assert runtime.verify_sink_reference() == runtime.sink_reference_sha256
        original = runtime.sink_reference.clone()
        runtime.sink_reference[0, 0, 0, 0, 0] += 1
        with pytest.raises(RuntimeError, match="reference.*changed"):
            runtime.verify_sink_reference()
        runtime.sink_reference = original
        assert runtime.verify_sink_reference() == runtime.sink_reference_sha256
        runtime.reset()
        assert runtime.sink_reference_sha256 is None and runtime.verify_sink_reference() is None
