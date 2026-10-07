"""Mechanism interventions must leave slow weights and training semantics alone."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
import os
import threading
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from test_runtime import begin, stage_all
from worldttn.core import TTNConfig
from worldttn.runtime import TTNSystem, TTNRuntimeState


def test_result_publication_has_private_temporary_files_and_rejects_nan(tmp_path, monkeypatch):
    from worldttn.mechanism_evaluation import atomic_json
    if os.name == "nt":
        # Windows rejects simultaneous replacement of one target. Keep writes
        # concurrent and serialize only that OS operation on the CPU test host.
        replace, lock = os.replace, threading.Lock()
        def windows_replace(source, target):
            with lock: replace(source, target)
        monkeypatch.setattr(os, "replace", windows_replace)
    path = tmp_path / "summary.json"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: atomic_json(path, {"writer": i}), range(20)))
    original = json.loads(path.read_text())
    assert original["writer"] in range(20)
    with pytest.raises(ValueError): atomic_json(path, {"metric": float("nan")})
    assert json.loads(path.read_text()) == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["summary.json"]

def test_runtime_abc_interventions_use_same_c_weights_and_next_chunk_psi():
    torch.manual_seed(7)
    cfg = TTNConfig(heads=2, head_dim=8, generators=3, stage="C")
    system = TTNSystem(cfg)
    with torch.no_grad():
        for head in system.controller.heads:
            head.bias.fill_(.2)
    weights = copy.deepcopy(system.state_dict())
    states = {}
    initial = torch.randn(2, 5, 2, 8, 8)
    with torch.no_grad():
        for mode in ("full", "no-ttt", "identity"):
            r = TTNRuntimeState.create(cfg, 2, "cpu", ablation=mode, diagnostics=True)
            r.world_state = initial.clone()
            c = begin(r, system, [0], True)
            if mode == "identity":
                assert torch.equal(c.predicted, initial)
            stage_all(c)
            r.commit_chunk(c)
            assert bool(r.transition_fast.count_nonzero()) == (mode == "full")
            assert r.last_stats["ablation"] == mode
            assert "state_spectrum" in r.last_stats["anchors"][0]
            states[mode] = (c.predicted.clone(), begin(r, system, [1]).predicted.clone())
            r.reset()
            assert r.ablation == mode and r.transition_fast.count_nonzero() == 0
    torch.testing.assert_close(states["full"][0], states["no-ttt"][0], rtol=0, atol=0)
    assert not torch.equal(states["full"][1], states["no-ttt"][1])
    for name, value in weights.items():
        assert torch.equal(value, system.state_dict()[name])
    assert system.config == cfg and system.config.stage == "C"


def test_spectrum_and_transport_have_exact_oracles_and_null_zero_ratios():
    from worldttn.stability import state_dynamics
    old = torch.diag(torch.tensor([3., 4.]))[None, None]
    predicted = old @ torch.tensor([[0., -1.], [1., 0.]])
    result = state_dynamics(old, predicted, predicted + 1, 1e-6)
    assert result["state_spectrum"]["previous"]["sigma_max"] == [[4.]]
    assert result["state_spectrum"]["previous"]["stable_rank"] == [[1.5625]]
    assert result["state_dynamics"]["transport_relative"][0][0] == pytest.approx(2**.5)
    zero = state_dynamics(torch.zeros_like(old), torch.zeros_like(old), old, 1e-6)
    assert zero["state_spectrum"]["previous"]["stable_rank"] == [[None]]
    assert zero["state_dynamics"]["transport_relative"] == [[None]]
    json.dumps(zero, allow_nan=False)


def test_late_metrics_fit_only_outside_training_horizon():
    from worldttn.evaluation import latent_metrics
    reference = torch.zeros(1, 1, 61, 1, 1)
    generated = torch.zeros_like(reference)
    generated[:, :, 1:13] = 10  # large prefix must not change tail slope
    generated[:, :, 13:] = torch.arange(48.).sqrt()[None, None, :, None, None]
    metrics = latent_metrics(generated, reference, [], training_frames=13)
    assert metrics["after_training_horizon_mean_latent_mse"] == pytest.approx(23.5)
    assert metrics["after_training_horizon_error_slope"] == pytest.approx(1.)
    assert metrics["after_training_horizon_frame_count"] == 48


@pytest.mark.parametrize("adapted", [False, True])
def test_actual_sampler_gt_history_is_post_prediction_causal_and_updates_all_caches(monkeypatch, adapted):
    """Run the production sampler with only its optional CUDA dependencies stubbed."""
    from worldttn.cli import rollout
    def inject(name, **symbols):
        module = ModuleType(name)
        for key, value in symbols.items(): setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    class Scheduler:
        def __init__(self, **kw): pass
        def set_timesteps(self, steps=None, device=None, **kw):
            self.timesteps = torch.linspace(900, 500, steps or 2, device=device)
        def step(self, model_output, t, sample, **kw): return (sample + .01 * model_output,)
    def retrieve(scheduler, steps, device, unused):
        scheduler.set_timesteps(steps, device)
        return scheduler.timesteps, steps
    inject("diffusers", FlowMatchEulerDiscreteScheduler=Scheduler)
    inject("diffusers.models.modeling_outputs", Transformer2DModelOutput=type("Output", (), {}))
    inject("diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3", retrieve_timesteps=retrieve)
    inject("diffusion.model.nets.basic_modules", CachedGLUMBConvTemp=type("Conv", (), {}))
    inject("diffusion.model.nets.sana_blocks", CachedCausalAttention=type("Attention", (), {}))
    root = Path(__file__).resolve().parents[2]
    name = "diffusion.scheduler.self_forcing_flow_euler_sampler"
    spec = importlib.util.spec_from_file_location(name, root / "diffusion/scheduler/self_forcing_flow_euler_sampler.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setenv("SANA_WM_STAGE1_KV_SAVE_STRIDE", "1")
    monkeypatch.setenv("DPM_TQDM", "True")
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(20)])
            self.weight = torch.nn.Parameter(torch.ones(1))
            if adapted: self.ttn_system = TTNSystem(TTNConfig(heads=2, head_dim=8, generators=3, stage="C"))
            self.calls = []
        def forward(self, *args, **kw): return self.forward_long(*args, **kw)
        def forward_long(self, x, t, y, kv_cache, save_kv_cache=False, start_f=0, end_f=1, **kw):
            assert "history_reference" not in kw and "clean_history_provider" not in kw
            self.calls.append((start_f, end_f, save_kv_cache, x.clone(), copy.deepcopy(kv_cache)))
            history = kv_cache[0][0]
            pred = torch.zeros_like(x) if history is None else torch.ones_like(x) * history.mean()
            if not save_kv_cache: return pred, kv_cache
            payload = x.mean((1, 2, 3, 4)).reshape(x.shape[0], 1, 1, 1)
            caches = [[payload, payload, payload, payload, payload, None, torch.tensor(1.), None, None, payload]
                      for _ in self.blocks]
            context = kw.get("ttn_chunk_context")
            if context is not None:
                for i in range(5):
                    context.stage(i, context.predicted[:, i] + payload,
                                  torch.ones_like(context.psi[:, i]) * .1, {})
            return pred, caches
    camera = torch.cat([torch.eye(4).flatten().expand(1, 10, 16), torch.ones(1, 10, 4)], -1)
    reference = torch.arange(10.).reshape(1, 1, 10, 1, 1)
    noise = torch.zeros_like(reference)
    batch = dict(initial_latent=reference[:, :, :1], camera_conditions=camera, width=1, height=1,
                 y=torch.zeros(1, 1, 2, 8), mask=torch.ones(1, 2), data_info={})
    config = SimpleNamespace(scheduler=SimpleNamespace(inference_flow_shift=1.))
    model = Model()
    weights = copy.deepcopy(model.state_dict())
    output, runtime, _ = rollout(model, config, batch, 2, 4.5, 2, initial_noise=noise,
                                 history_reference=reference, state_diagnostics=True)
    for start, end, save, x, cache in model.calls:
        if save:
            torch.testing.assert_close(x, reference[:, :, start:end].expand(2, -1, -1, -1, -1))
        if start >= 4:
            previous = reference[:, :, :4] if start == 4 else reference[:, :, 4:7]
            for slot in (0, 1, 2, 3, 4, 9):
                assert cache[0][slot].mean() == previous.mean()
    assert not torch.equal(output[:, :, 1:], reference[:, :, 1:])
    assert noise.count_nonzero() == 0 and torch.equal(reference.flatten(), torch.arange(10.))
    if adapted:
        assert runtime.commit_count == runtime.predict_count == 4
        assert runtime.world_state.shape[0] == 2 and runtime.committed_frame_ids == [set(range(10))] * 2
    changed = reference.clone()
    changed[:, :, 4:7] += 100
    again = Model()
    again.load_state_dict(weights)
    alternative = rollout(again, config, batch, 2, 4.5, 2, initial_noise=noise, history_reference=changed)[0]
    torch.testing.assert_close(output[:, :, :7], alternative[:, :, :7], rtol=0, atol=0)
    assert not torch.equal(output[:, :, 7:], alternative[:, :, 7:])
    for key, value in weights.items(): assert torch.equal(value, model.state_dict()[key])


def test_mechanism_options_reject_training_before_cuda_check():
    import subprocess
    result = subprocess.run([sys.executable, "-m", "worldttn.cli", "train", "--ttn-ablation", "no-ttt"],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "evaluate-only" in result.stderr


def mechanism_files(output, meta=False, histories=False):
    from worldttn.evaluation import latent_metrics
    from worldttn.mechanism_evaluation import VARIANTS, IDENTITY
    protocol = {key: "same" for key in IDENTITY}
    protocol.update(stage="C", step=100, frames=61, noise_frames=61, steps=20, cfg_scale=4.5,
                    cached_blocks=2, training_latent_frames=13, state_diagnostics=True, ttn_camera_attention="sana")
    gt = torch.zeros(1, 1, 61, 1, 1)
    def record(method, mse):
        generated = torch.ones_like(gt) * mse**.5
        return dict(method=method, case_id="case", seed=3407, input_sha256={"input": "same"},
                    initial_noise_sha256="noise", base_sha256="base",
                    metrics=latent_metrics(generated, gt, [], training_frames=13),
                    prefix_13_metrics=latent_metrics(generated[:, :, :13], gt[:, :, :13], [], training_frames=13))
    variants = dict(VARIANTS, **({"no-local": ("no-local", "generated", ["ttn"]),
                               "no-persistent": ("no-persistent", "generated", ["ttn"])} if meta else {}))
    if histories:
        variants = {"full": ("full", "generated", ["sana", "ttn"]),
                    "ttn-gt-history": ("full", "ttn-gt", ["ttn"]),
                    "native-gt-history": ("full", "native-gt", ["ttn"]),
                    "gt-history": ("full", "gt", ["sana", "ttn"])}
    protocol["meta_ttt"] = {"local_update": meta, "persistent_meta": meta}
    for variant, (ablation, history, methods) in variants.items():
        folder = output / variant
        folder.mkdir(parents=True)
        p = dict(protocol, ttn_ablation=ablation, history_source=history, eval_methods=methods)
        values = {"full": 4., "no-ttt": 3., "identity": 5., "gt-history": 2., "no-local": 6., "no-persistent": 7.,
                  "ttn-gt-history": 3., "native-gt-history": 2.5}
        records = [record(method, values[variant] if method == "ttn" else .5 if history == "gt" else 1.) for method in methods]
        (folder / "summary.json").write_text(json.dumps({"protocol": p, "episodes": records}))
    if histories:
        original = output.parent / f"{output.name}-source" / "long"
        original.mkdir(parents=True)
        original.joinpath("summary.json").write_text(output.joinpath("full/summary.json").read_text())
        output.joinpath("plan.json").write_text(json.dumps({"suite": "histories", "variants": list(variants),
                                                           "source_evaluation": str(original.parent)}))


def test_history_suite_collects_conditional_contrasts_and_null_safe_interaction(tmp_path):
    from worldttn.mechanism_evaluation import collect_mechanisms
    mechanism_files(tmp_path, meta=True, histories=True)
    result = collect_mechanisms(tmp_path)
    assert result["status"] == "completed" and result["suite"] == "histories"
    assert result["full_c_reproduction"]["cases"][0]["within_tolerance"]
    contrasts = result["contrasts"]
    assert contrasts["ttn_gt_state_minus_generated_state"]["long"]["mean_future_latent_mse"] == pytest.approx(-1.)
    assert contrasts["native_gt_cache_minus_generated_cache"]["long"]["mean_future_latent_mse"] == pytest.approx(-1.5)
    assert contrasts["history_factorial_interaction"]["long"]["mean_future_latent_mse"] == pytest.approx(.5)
    assert contrasts["history_factorial_interaction"]["long"]["revisit_return_gt_latent_mse"] is None
    assert "coupled" in result["interpretation"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("corrupt", ["variants", "active-sink", "noise", "history"])
def test_history_suite_rejects_wrong_suite_or_protocol(tmp_path, corrupt):
    from worldttn.mechanism_evaluation import collect_mechanisms
    mechanism_files(tmp_path, histories=True)
    path = tmp_path / ("plan.json" if corrupt == "variants" else "ttn-gt-history/summary.json")
    value = json.loads(path.read_text())
    if corrupt == "variants": value["variants"][1] = "no-ttt"
    elif corrupt == "active-sink": value["protocol"]["tla_sink"] = {"mode": "protected", "gain": .1}
    elif corrupt == "noise": value["episodes"][0]["initial_noise_sha256"] = "other"
    else: value["protocol"]["history_source"] = "native-gt"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): collect_mechanisms(tmp_path)


def test_history_suite_accepts_explicit_off_sink_and_preserves_failed_partial_status(tmp_path):
    from worldttn.mechanism_evaluation import collect_mechanisms
    mechanism_files(tmp_path, histories=True)
    path = tmp_path / "ttn-gt-history/summary.json"
    value = json.loads(path.read_text())
    value["protocol"]["tla_sink"] = {"mode": "off", "gain": .1, "position": "temporal-realign", "start_chunk": 5}
    path.write_text(json.dumps(value))
    assert collect_mechanisms(tmp_path)["status"] == "completed"
    path.unlink()
    (tmp_path / "ttn-gt-history-status.json").write_text('{"status":"failed"}')
    assert collect_mechanisms(tmp_path)["status"] == "failed"


def test_meta_collector_requires_both_contributions_and_reports_separate_contrasts(tmp_path):
    from worldttn.mechanism_evaluation import collect_mechanisms
    mechanism_files(tmp_path, meta=True)
    result = collect_mechanisms(tmp_path)
    assert result["status"] == "completed" and len(result["results"]) == 6
    assert result["contrasts"]["local_minus_no_local"]["long"]["mean_future_latent_mse"] == pytest.approx(-2.)
    (tmp_path / "no-persistent/summary.json").unlink()
    assert collect_mechanisms(tmp_path)["status"] == "running"


@pytest.mark.parametrize("corrupt", [None, "checkpoint", "noise", "history"])
def test_collector_pairs_same_case_checkpoint_and_distinguishes_gt_baseline(tmp_path, corrupt):
    from worldttn.mechanism_evaluation import collect_mechanisms
    mechanism_files(tmp_path)
    if corrupt:
        path = tmp_path / "no-ttt/summary.json"
        value = json.loads(path.read_text())
        if corrupt == "checkpoint": value["protocol"]["checkpoint_sha256"] = "other"
        elif corrupt == "noise": value["episodes"][0]["initial_noise_sha256"] = "other"
        else: value["protocol"]["history_source"] = "gt"
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError): collect_mechanisms(tmp_path)
        return
    result = collect_mechanisms(tmp_path)
    assert result["status"] == "completed"
    assert result["results"]["no-ttt"]["long"]["metrics"]["mean_future_latent_mse"]["sana"] == 1.
    assert result["results"]["gt-history"]["long"]["metrics"]["mean_future_latent_mse"]["sana"] == pytest.approx(.5)
    assert result["contrasts"]["online_ttt_minus_no_ttt"]["long"]["mean_future_latent_mse"] == pytest.approx(1.)
    assert result["contrasts"]["controller_no_ttt_minus_identity"]["long"]["mean_future_latent_mse"] == pytest.approx(-2.)
    json.dumps(result, allow_nan=False)


def test_collector_preserves_failed_partial_status(tmp_path):
    from worldttn.mechanism_evaluation import collect_mechanisms
    (tmp_path / "no-ttt-status.json").write_text('{"status":"failed"}')
    assert collect_mechanisms(tmp_path)["status"] == "failed"


def test_marking_running_keeps_variant_output_new_for_evaluator(tmp_path, monkeypatch):
    from worldttn.mechanism_evaluation import main
    output = tmp_path / "new-experiment"
    monkeypatch.setattr(sys, "argv", ["collect", str(output), "--variant", "full", "--status", "running"])
    main()
    assert not (output / "full").exists()
    assert json.loads((output / "summary.json").read_text())["variants"]["full"]["status"] == "running"


def test_full_c_reproduction_compares_actual_frame_errors_not_only_mean(tmp_path):
    from worldttn.mechanism_evaluation import collect_mechanisms
    output = tmp_path / "experiment"
    mechanism_files(output)
    original = tmp_path / "original/long"
    original.mkdir(parents=True)
    old = json.loads((output / "full/summary.json").read_text())
    (original / "summary.json").write_text(json.dumps(old))
    (output / "plan.json").write_text(json.dumps({"source_evaluation": str(original.parent)}))
    result = collect_mechanisms(output)
    assert result["full_c_reproduction"]["cases"][0]["within_tolerance"]
    old["episodes"][1]["metrics"]["per_frame_latent_mse"][30] += .1
    (original / "summary.json").write_text(json.dumps(old))
    result = collect_mechanisms(output)
    assert not result["full_c_reproduction"]["cases"][0]["within_tolerance"]
    assert result["full_c_reproduction"]["cases"][0]["max_abs_per_frame_mse_delta"] == pytest.approx(.1)


def test_submitter_reuses_verified_immutable_snapshot_and_fixed_cases(tmp_path):
    from tools.ttn_submit_mechanism import prepare, digest
    source = tmp_path / "joint"
    source.mkdir()
    fixed = source / "fixed-cases.pt"
    fixed.write_bytes(b"immutable cases")
    snapshot = tmp_path / "eval-snapshot-original"
    snapshot.mkdir()
    checkpoint = snapshot / "last.pt"
    checkpoint.write_bytes(b"immutable model")
    (snapshot / "snapshot.json").write_text(json.dumps({"source": str(source), "step": 100}))
    evaluation = source / "evaluations/step-000100"
    (evaluation / "long").mkdir(parents=True)
    (evaluation / "summary.json").write_text(json.dumps({"status": "completed", "results": {
        "long": {"metrics": {"mean_future_latent_mse": {"paired_count": 1}}}}}))
    protocol = dict(stage="C", frames=61, ttn_camera_attention="sana", training_run=str(snapshot), step=100,
                    checkpoint_sha256=digest(checkpoint), fixed_cases_sha256=digest(fixed), steps=20,
                    cfg_scale=4.5, cached_blocks=2)
    path = evaluation / "long/summary.json"
    path.write_text(json.dumps({"protocol": protocol}))
    (evaluation / "long/manifest.json").write_text('{"cases":[{"seed":3407}]}')
    plan = prepare(evaluation)
    assert plan["snapshot"] == str(snapshot) and plan["fixed_cases"] == str(fixed)
    assert not Path(plan["output"]).exists() and checkpoint.read_bytes() == b"immutable model"
    history_plan = prepare(evaluation, suite="histories")
    assert history_plan["suite"] == "histories" and history_plan["array"] == "0-3%1"
    assert history_plan["variants"] == ["full", "ttn-gt-history", "native-gt-history", "gt-history"]
    assert history_plan["interventions"]["ttn-gt-history"] == ["full", "ttn-gt", ["ttn"]]
    protocol["tla_sink"] = {"mode": "protected", "gain": .1}
    path.write_text(json.dumps({"protocol": protocol}))
    with pytest.raises(ValueError, match="sink off"): prepare(evaluation, suite="histories")
    protocol.pop("tla_sink")
    path.write_text(json.dumps({"protocol": protocol}))
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint"): prepare(evaluation)


def test_submitter_clears_launcher_and_camera_overrides_and_submits_only_array(tmp_path, monkeypatch, capsys):
    from tools import ttn_submit_mechanism as submit
    plan = dict(output=str(tmp_path / "results"), snapshot="pinned", fixed_cases="cases", step=100,
                steps=20, cfg_scale=4.5, cached_blocks=2, eval_cases=1, seed=3407,
                cross_attn_backend="math", compile={"GDN_DISABLE_COMPILE": "1"})
    monkeypatch.setattr(submit, "prepare", lambda *args: plan)
    python = tmp_path / "envs/worldttn/bin/python"
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setattr(sys, "argv", ["submit", "--evaluation", "completed-milestone"])
    monkeypatch.setenv("ROOT", str(tmp_path))
    for name in ("SLURM_JOB_ID", "RANK", "MASTER_ADDR", "CUDA_VISIBLE_DEVICES", "CAMERA_ATTENTION", "ADAPTER"):
        monkeypatch.setenv(name, "stale")
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        env = kwargs["env"]
        assert not any(k in env for k in ("SLURM_JOB_ID", "RANK", "MASTER_ADDR", "CUDA_VISIBLE_DEVICES", "CAMERA_ATTENTION", "ADAPTER"))
        assert env["TRAINING_RUN"] == "pinned" and env["PYTHON"] == str(python)
        assert env["GDN_DISABLE_COMPILE"] == "1"
        return SimpleNamespace(stdout="12345;cluster\n")
    monkeypatch.setattr(submit.subprocess, "run", run)
    submit.main()
    assert len(calls) == 1 and calls[0][0] == "sbatch" and calls[0][-1].endswith("ttn_slurm_mechanism.sbatch")
    assert json.loads((tmp_path / "results/job.json").read_text()) == {"job": "12345", "array": "0-3%1"}
    assert "MECHANISM_JOB=12345" in capsys.readouterr().out
