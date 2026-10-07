"""Post-prediction history interventions must publish one isolated transaction."""
import importlib
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from worldttn.core import TTNConfig
from worldttn.performance import ExecutionOptions
from worldttn.runtime import TTNRuntimeState, TTNSystem
from worldttn.session import TTNSession


def history():
    assert importlib.util.find_spec("worldttn.history") is not None, "history transaction helper is missing"
    return importlib.import_module("worldttn.history")


def episode():
    cfg = TTNConfig(heads=2, head_dim=8, generators=3, stage="C", camera_attention="sana")
    system = TTNSystem(cfg)
    runtime = TTNRuntimeState.create(cfg, 2, "cpu", diagnostics=True)
    ids = torch.arange(3).expand(2, -1)
    poses = torch.eye(4).expand(2, 3, 4, 4).clone()
    intrinsics = torch.ones(2, 3, 4)
    context = runtime.begin_chunk(system, poses, intrinsics, ids, torch.ones_like(ids, dtype=torch.bool), 1, 1)
    context.local_stats = {0: {"noisy": [7]}}
    cache = [[torch.tensor([5.])] + [None] * 9]
    generated = torch.ones(2, 1, 3, 1, 1)
    reference = generated * 2
    return runtime, context, cache, generated, reference


@pytest.mark.parametrize("source,state_value,cache_value,passes", [
    ("generated", 1., 1., 1), ("gt", 2., 2., 1),
    ("ttn-gt", 2., 1., 2), ("native-gt", 1., 2., 2)])
@torch.no_grad()
def test_four_sources_select_state_and_native_cache_and_commit_once(source, state_value, cache_value, passes):
    runtime, context, cache, generated, reference = episode()
    incoming_state, incoming_cache = context.predicted.clone(), cache[0][0].clone()
    calls = []
    def forward(x, clean, native):
        calls.append((float(x.mean()), native[0][0].clone(), native is cache))
        clean.local_stats[0]["noisy"].append(float(x.mean()))
        native[0][0].fill_(float(x.mean()))
        for i in range(5):
            clean.stage(i, clean.predicted[:, i] + x.mean(), torch.ones_like(clean.psi[:, i]) * .1, {})
        return torch.zeros_like(x), native
    output, updated = history().clean_history_transaction(forward, generated, cache, source=source,
        reference=reference, context=context, runtime=runtime)
    assert output.shape == generated.shape and len(calls) == passes
    torch.testing.assert_close(runtime.world_state, incoming_state + state_value)
    assert updated[0][0].item() == cache_value
    assert runtime.commit_count == 1 and runtime.predict_count == 1
    assert runtime.committed_frame_ids == [set(range(3))] * 2 and context.candidates == {}
    assert runtime.last_stats["history_source"] == source
    assert runtime.last_stats["history_origins"] == {
        "ttn": "gt" if source in ("gt", "ttn-gt") else "generated",
        "native": "gt" if source in ("gt", "native-gt") else "generated"}
    if passes == 2:
        assert all(torch.equal(call[1], incoming_cache) and not call[2] for call in calls)
        assert torch.equal(cache[0][0], incoming_cache)
        assert context.local_stats == {0: {"noisy": [7]}}
    else:
        assert calls[0][2]  # Legacy one-pass cache identity is retained.


@pytest.mark.parametrize("failure", ["exception", "nonfinite-output", "bad-cache", "bad-candidate"])
@torch.no_grad()
def test_second_pass_failure_leaves_runtime_context_and_incoming_cache_unchanged(failure):
    runtime, context, cache, generated, reference = episode()
    before = (runtime.world_state.clone(), runtime.transition_fast.clone(), runtime.revision,
              context.predicted.clone(), cache[0][0].clone())
    calls = []
    def forward(x, clean, native):
        calls.append(float(x.mean()))
        native[0][0].add_(99)
        clean.predicted.add_(8)  # Even an in-place consumer must not corrupt the sibling pass.
        clean.local_stats[0]["noisy"].append(99)
        for i in range(5):
            clean.stage(i, clean.predicted[:, i], torch.zeros_like(clean.psi[:, i]), {})
        if len(calls) == 2:
            if failure == "exception": raise RuntimeError("second clean forward failed")
            if failure == "nonfinite-output": return x * float("nan"), native
            if failure == "bad-cache": return x, [[torch.tensor([float("nan")])] + [None]*9]
            if failure == "bad-candidate": clean.candidates[0] = (torch.zeros(1), clean.psi[:, 0], {})
        return x, native
    with pytest.raises((RuntimeError, ValueError, FloatingPointError)):
        history().clean_history_transaction(forward, generated, cache, source="ttn-gt",
            reference=reference, context=context, runtime=runtime)
    assert len(calls) == 2 and runtime.commit_count == 0
    assert runtime.revision == before[2] and runtime.committed_frame_ids == [set(), set()]
    assert torch.equal(runtime.world_state, before[0]) and torch.equal(runtime.transition_fast, before[1])
    assert torch.equal(context.predicted, before[3]) and torch.equal(cache[0][0], before[4])
    assert context.candidates == {} and context.local_stats == {0: {"noisy": [7]}}


@pytest.mark.parametrize("bad_reference", [None, "shape", "dtype", "nonfinite"])
@torch.no_grad()
def test_invalid_gt_rejected_before_forward(bad_reference):
    runtime, context, cache, generated, reference = episode()
    reference = {None: None, "shape": reference[:, :, :1], "dtype": reference.double(),
                 "nonfinite": reference * float("nan")}[bad_reference]
    def forward(*args): raise AssertionError("invalid GT reached model")
    with pytest.raises(ValueError, match="clean history"):
        history().clean_history_transaction(forward, generated, cache, source="ttn-gt",
            reference=reference, context=context, runtime=runtime)
    assert runtime.commit_count == 0


def test_source_resolution_keeps_provider_only_legacy_and_rejects_unsupported_mixed_modes():
    provider = lambda start, end: None
    helper = history()
    assert helper.resolve_clean_history_source(None, None, None) == "generated"
    assert helper.resolve_clean_history_source(None, provider, None) == "gt"
    with pytest.raises(ValueError, match="provider"):
        helper.resolve_clean_history_source("gt", None, None)
    with torch.no_grad():
        runtime, context, *_ = episode()
        model = SimpleNamespace(ttn_system=context.system)
        session = SimpleNamespace(model=model, runtime=runtime, ablation="full")
        assert helper.resolve_clean_history_source("native-gt", provider, session) == "native-gt"
        with pytest.raises(ValueError, match="TTN"):
            helper.resolve_clean_history_source("native-gt", provider, None)
        session.ablation = "identity"
        with pytest.raises(ValueError, match="Full Stage C"):
            helper.resolve_clean_history_source("native-gt", provider, session)
        session.ablation = "full"
        context.system.ttn_execution = ExecutionOptions("reuse", "reference")
        with pytest.raises(ValueError, match="reference/reference"):
            helper.resolve_clean_history_source("native-gt", provider, session)
    with pytest.raises(ValueError, match="inference-only"):
        helper.resolve_clean_history_source("ttn-gt", provider, session)


@pytest.mark.parametrize("flag", [0., 1.])
@torch.no_grad()
def test_native_fused_cache_python_type_flags_are_preserved(flag):
    runtime, context, cache, generated, reference = episode()
    cache[0][6] = flag  # fused_streaming._TYPE_STATE/_TYPE_CONCAT are Python floats.
    def forward(x, clean, native):
        for i in range(5):
            clean.stage(i, clean.predicted[:, i] + x.mean(), torch.zeros_like(clean.psi[:, i]), {})
        return x, native
    _, updated = history().clean_history_transaction(forward, generated, cache, source="ttn-gt",
        reference=reference, context=context, runtime=runtime)
    assert updated[0][6] == flag and runtime.commit_count == 1


@pytest.mark.parametrize("source", ["generated", "gt", "ttn-gt", "native-gt"])
@pytest.mark.parametrize("cfg_scale", [1., 4.5])
@torch.no_grad()
def test_actual_sampler_mixed_history_is_causal_and_preserves_one_commit_per_chunk(monkeypatch, source, cfg_scale):
    path = Path(__file__).resolve().parents[2]/"diffusion/scheduler/self_forcing_flow_euler_sampler.py"
    cls = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
               if isinstance(n, ast.ClassDef) and n.name == "SelfForcingFlowEulerCamCtrl")
    loop = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "sample_chunks")
    class Scheduler:
        def __init__(self, **kw): pass
        def set_timesteps(self, **kw):
            self.timesteps, self.sigmas = torch.tensor([900., 500.]), torch.tensor([.9, .5, 0.])
        def step(self, prediction, t, sample, **kw): return (sample + .01*prediction,)
    def retrieve(scheduler, *args):
        scheduler.set_timesteps()
        return scheduler.timesteps, 2
    namespace = dict(torch=torch, os=os, tqdm=lambda items, **kw: items,
                     FlowMatchEulerDiscreteScheduler=Scheduler, retrieve_timesteps=retrieve,
                     Transformer2DModelOutput=type("UnusedOutput", (), {}), _NUM_CACHE_SLOTS=10)
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(path), "exec"), namespace)
    monkeypatch.setenv("SANA_WM_STAGE1_KV_SAVE_STRIDE", "1")
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ttn_system = TTNSystem(TTNConfig(heads=2, head_dim=8, generators=3,
                                                 stage="C", camera_attention="sana"))
            self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(20)])
            self.events = []
        def forward(self, x, t, y, ttn_chunk_context, kv_cache, save_kv_cache=False, **kwargs):
            assert "clean_history_provider" not in kwargs and "clean_history_source" not in kwargs
            ctx = ttn_chunk_context
            native = kv_cache[0][0]
            native_value = 0. if native is None else float(native.mean())
            self.events.append(("prefill" if ctx.prefill_mode else "clean" if save_kv_cache else "noisy",
                                tuple(ctx.frame_ids[0].tolist()), x.clone(),
                                float(ctx.previous.mean()), native_value))
            pred = torch.ones_like(x)*(native_value + ctx.previous.mean())
            if not save_kv_cache: return pred, kv_cache
            payload = x.mean()
            for i in range(5):
                ctx.stage(i, ctx.predicted[:, i] + payload, torch.ones_like(ctx.psi[:, i])*.1, {})
            return pred, [[payload.reshape(1)] + [None]*9 for _ in self.blocks]
    camera = torch.cat((torch.eye(4).flatten().expand(1, 10, 16), torch.ones(1, 10, 4)), -1)
    reference = torch.arange(10.).reshape(1, 1, 10, 1, 1)
    def run(gt):
        model = Model().eval()
        session = TTNSession(model, camera, 1, 1)
        provider_calls = []
        def provider(start, end):
            # The two current solver calls precede even the first GT read.
            assert [row[0] for row in model.events[-2:]] == ["noisy", "noisy"]
            assert all(row[1] == tuple(range(start, end)) for row in model.events[-2:])
            provider_calls.append((start, end))
            return gt[:, :, start:end]
        sampler = SimpleNamespace(condition=torch.zeros(1, 1, 2, 8), uncondition=torch.zeros(1, 1, 2, 8),
            cfg_scale=cfg_scale, base_chunk_frames=3, num_cached_blocks=2, num_model_blocks=20,
            sink_token=False, flow_shift=1., model=model, mask=None, ttn_session=session,
            clean_history_source=source, clean_history_provider=provider,
            model_kwargs={"data_info": {"condition_frame_info": {0: 0.}}},
            create_autoregressive_segments=lambda frames: [0, 4, 7, 10],
            _initialize_kv_cache=lambda n: [[[None]*10 for _ in model.blocks] for _ in range(n)],
            accumulate_kv_cache=lambda caches, i: (caches[max(i-1, 0)], i, 0, 0))
        latents = torch.ones_like(reference)
        latents[:, :, :1] = reference[:, :, :1]
        assert len(list(namespace["sample_chunks"](sampler, latents, steps=2))) == 3
        assert session.runtime.commit_count == session.runtime.predict_count == 4
        assert session.runtime.committed_frame_ids == [set(range(10))]*(2 if cfg_scale > 1 else 1)
        assert len(provider_calls) == (0 if source == "generated" else 3)
        calls = [row for row in model.events if row[0] == "clean"]
        assert len(calls) == (6 if source in ("ttn-gt", "native-gt") else 3)
        if source in ("ttn-gt", "native-gt"):
            for generated_call, gt_call in zip(calls[::2], calls[1::2]):
                assert generated_call[1] == gt_call[1] and generated_call[3:] == gt_call[3:]
                start, end = gt_call[1][0], gt_call[1][-1]+1
                torch.testing.assert_close(gt_call[2], gt[:, :, start:end].expand_as(gt_call[2]), rtol=0, atol=0)
        return latents
    result = run(reference)
    changed = reference.clone()
    changed[:, :, 4:7] += 100
    alternative = run(changed)
    torch.testing.assert_close(result[:, :, :7], alternative[:, :, :7], atol=0, rtol=0)
    assert torch.equal(result, alternative) if source == "generated" else not torch.equal(result[:, :, 7:], alternative[:, :, 7:])
