"""Execute the real sampler loop without importing optional CUDA model packages."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from test_training import TinyWorldModel, inputs
from worldttn.session import TTNSession


@pytest.mark.parametrize("meta", [False, True])
@pytest.mark.parametrize("cfg_scale", [1., 4.5])
@pytest.mark.parametrize("telemetry", [False, True])
def test_rollout_records_exact_per_call_sigma_and_commits_once_per_clean_chunk(monkeypatch, meta, cfg_scale, telemetry):
    path = Path(__file__).resolve().parents[2]/"diffusion/scheduler/self_forcing_flow_euler_sampler.py"
    cls = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
               if isinstance(n, ast.ClassDef) and n.name == "SelfForcingFlowEulerCamCtrl")
    loop = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "sample_chunks")
    class Scheduler:
        def __init__(self, **kw): pass
        def set_timesteps(self, **kw):
            self.sigmas = torch.tensor([.73137, .23456, 0.])
            self.timesteps = torch.tensor([731., 234.])
        def step(self, prediction, t, sample, **kw): return (sample,)
    def retrieve(scheduler, *args):
        scheduler.set_timesteps()
        return scheduler.timesteps, 2
    ns = dict(torch=torch, os=os, tqdm=lambda items, **kw: items,
              FlowMatchEulerDiscreteScheduler=Scheduler, retrieve_timesteps=retrieve,
              Transformer2DModelOutput=type("UnusedOutput", (), {}), _NUM_CACHE_SLOTS=10)
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(path), "exec"), ns)
    monkeypatch.setenv("SANA_WM_STAGE1_KV_SAVE_STRIDE", "1")
    model = TinyWorldModel(local_update=meta, persistent_meta=meta).eval()
    _, latents, _, camera = inputs()
    session = TTNSession(model, camera[:, :7], 100, 100, collect_local_stats=telemetry)
    calls = []
    def call(x, t, y, **kw):
        ctx = kw["ttn_chunk_context"]
        before = (session.runtime.world_state.clone(), session.runtime.transition_fast.clone())
        result = model(x, t, y, **kw)
        if not ctx.clean_mode:
            calls.append((t.clone(), dict(ctx.noise_info), dict(ctx.local_stats)))
            assert torch.equal(before[0], session.runtime.world_state)
            assert torch.equal(before[1], session.runtime.transition_fast)
        return result
    sampler = SimpleNamespace(condition=torch.zeros(1, 1, 2, 8), uncondition=torch.zeros(1, 1, 2, 8),
        cfg_scale=cfg_scale, base_chunk_frames=3, num_cached_blocks=2, num_model_blocks=20,
        sink_token=False, flow_shift=9.8, model=call, mask=None, ttn_session=session,
        clean_history_provider=None, model_kwargs={"data_info": {"condition_frame_info": {0: 0.}}},
        create_autoregressive_segments=lambda frames: [0, 4, 7],
        _initialize_kv_cache=lambda n: [[[None]*10 for _ in model.blocks] for _ in range(n)],
        accumulate_kv_cache=lambda caches, i: (caches[i], i, 0, 0))
    chunks = list(ns["sample_chunks"](sampler, latents[:, :, :7].clone(), steps=2))
    assert len(chunks) == 2 and len(calls) == 4 and session.runtime.commit_count == 3
    for index, (t, info, local) in enumerate(calls):
        if not meta or not telemetry:
            assert info == {} and local == {}
            continue
        expected = torch.full_like(t, float(sampler.scheduler.sigmas[index%2]))
        if index < 2: expected[:, :, 0] = 0.
        torch.testing.assert_close(torch.tensor(info["noise_sigma"]), expected, atol=0, rtol=0)
        assert info["noise_timestep"] == t.tolist()
        assert set(local) == set(range(5))
        assert all(row["noise_sigma"] == info["noise_sigma"] for row in local.values())
    if meta and telemetry:
        for anchor in session.runtime.last_stats["anchors"]:
            assert len(anchor["local_trajectory"]) == 2
