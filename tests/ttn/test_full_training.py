"""Trained DSR/Sink wiring, gradient boundaries and generated-history causality."""
import pytest
import torch
import os
from pathlib import Path
from test_training import TinyWorldModel, TinyBlock, inputs
from test_sink import actual_rope
from test_alignment_detail import cached_sana
from worldttn import training
from worldttn.anchor import configure_train_scope, install_ttn
from worldttn.checkpoint import make_optimizer, save_checkpoint, load_checkpoint
from worldttn.core import TTNConfig


class EulerOracle:
    """Constant-velocity oracle for the native per-token scheduler contract."""
    def __init__(self, steps, shift, device):
        self.sigmas = torch.linspace(1, 0, steps+1, device=device)
        self.timesteps = 1000*self.sigmas[:-1]
        self.steps = steps

    def step(self, prediction, timestep, sample, per_token_timesteps, return_dict):
        # Native per-token path takes negative velocity and a positive dt.
        return (sample + prediction/self.steps * (per_token_timesteps > 0)[..., None],)


def model(cached_sana, psi=True):
    torch.manual_seed(41)
    m = TinyWorldModel.__new__(TinyWorldModel)
    torch.nn.Module.__init__(m)
    m.blocks = torch.nn.ModuleList(TinyBlock() for _ in range(20))
    for block in m.blocks: block.attn = cached_sana()
    install_ttn(m, TTNConfig(heads=2, head_dim=8, generators=3, stage="C", camera_attention="sana",
                local_update=psi, persistent_meta=psi, persistent_update=psi,
                sink_gain=.25, replay_strength=.25, replay_budget=2))
    m.saved_features = []
    m.rope = actual_rope()
    return configure_train_scope(m, "dit")


def long_inputs():
    clean, noise, t, camera = inputs()
    return (torch.cat((clean, clean[:, :, 1:]), 2), torch.cat((noise, noise[:, :, 1:]), 2),
            torch.cat((t, t[:, :, 1:]), 2), torch.cat((camera, camera[:, 1:]), 1))


def update(m, optimizer=None, offload="none"):
    clean, noise, t, camera = long_inputs()
    return training.train_clip(m, clean, torch.zeros(1, 1, 2, 8), camera,
        optimizer or make_optimizer(m), training.linear_flow_loss, t, noise, width=100, height=100,
        tbptt=4, activation_offload=offload,
        history_training={"source": "generated", "steps": 2, "cached_chunks": 2})


@pytest.mark.parametrize("psi", [True, False])
def test_combined_training_gradients_boundaries_and_reference_integrity(monkeypatch, cached_sana, psi):
    monkeypatch.setattr(training, "_history_scheduler", EulerOracle)
    m = model(cached_sana, psi)
    original = m.forward
    def forward(*args, **kwargs):
        output = original(*args, **kwargs)
        context = kwargs["ttn_chunk_context"]
        for state, _, _ in context.candidates.values():
            if state.requires_grad: state.retain_grad()
        return output
    m.forward = forward
    result = update(m)
    runtime = result["runtime"]
    assert runtime.commit_count == runtime.predict_count == 9
    assert runtime.committed_frame_ids == [set(range(25))]
    assert runtime.verify_sink_reference() and runtime.verify_replay_reference()
    assert not runtime.world_state.requires_grad and not runtime.replay_values.requires_grad
    # Chunks 5/6/7 receive future flow credit inside the second window; 4/8 do not.
    assert m.saved_features[4].grad is None and m.saved_features[8].grad is None
    assert m.saved_features[5].grad.norm() > 0
    for i, row in enumerate(result["chunks"]):
        assert row["history_source"] == "generated" and row["history_generation_steps"] == 2
        for anchor in row["anchors"]:
            assert anchor["replay"]["active"] is (i >= 4)
            assert anchor["sink"]["active"] is (i >= 4)
    assert m.blocks[3].attn.beta_proj.weight.grad.norm() > 0
    assert m.ttn_system.controller.heads[0].weight.grad.norm() > 0
    if psi: assert m.ttn_system.local_eta_logits.grad.norm() > 0
    else: assert runtime.transition_fast.count_nonzero() == 0


def test_generated_history_is_independent_of_future_gt_and_does_not_commit(monkeypatch, cached_sana):
    monkeypatch.setattr(training, "_history_scheduler", EulerOracle)
    # A constant velocity has the known x_final = noise - v answer.
    class Session:
        def forward(self, x, t, y, context, cache, *args, **kwargs):
            return torch.ones_like(x)*.3, cache
    from worldttn.runtime import TTNRuntimeState
    from test_sink import begin
    m = model(cached_sana)
    runtime = TTNRuntimeState.create(m.ttn_system.config, 1, "cpu")
    context = begin(runtime, m.ttn_system, [0, 1, 2, 3])
    before = runtime.world_state.clone()
    x = torch.randn(1, 16, 4, 2, 2)
    observed = torch.ones_like(x[:, :, :1])*7
    actual = training.generate_history_chunk(Session(), x, observed, None, context, [],
        0, 4, None, None, training.history_settings({"source": "generated"}))
    torch.testing.assert_close(actual[:, :, 1:], x[:, :, 1:]-.3)
    torch.testing.assert_close(actual[:, :, :1], observed)
    assert runtime.commit_count == 0 and torch.equal(before, runtime.world_state)
    assert context.noise_info == {} and not context.candidates


def test_full_training_offload_and_checkpoint_retain_recipe(monkeypatch, tmp_path, cached_sana):
    monkeypatch.setattr(training, "_history_scheduler", EulerOracle)
    a, b = model(cached_sana), model(cached_sana)
    left, right = update(a), update(b, offload="cpu")
    assert left["loss"] == right["loss"]
    for p, q in zip(a.parameters(), b.parameters()): torch.testing.assert_close(p, q, rtol=0, atol=0)
    a.base_load_report = {"sha256": "test"}
    path = tmp_path/"last.pt"
    save_checkpoint(path, a, None, 1)
    restored = model(cached_sana); restored.base_load_report = a.base_load_report
    load_checkpoint(path, restored)
    for p, q in zip(a.parameters(), restored.parameters()): torch.testing.assert_close(p, q, rtol=0, atol=0)
    wrong = model(cached_sana, False); wrong.base_load_report = a.base_load_report
    with pytest.raises(ValueError, match="identity mismatch"): load_checkpoint(path, wrong)


def test_full_training_recipe_validation():
    with pytest.raises(ValueError, match="persistent_update"):
        TTNConfig(stage="C", persistent_meta=True, persistent_update=False)
    with pytest.raises(ValueError, match="native SANA"):
        TTNConfig(stage="C", replay_strength=.25)
    with pytest.raises(ValueError, match="CFG"):
        training.history_settings({"source": "generated", "cfg_scale": 4.5})
    for bad in (True, None, "0.25", float("nan")):
        with pytest.raises(ValueError): TTNConfig(stage="C", camera_attention="sana", sink_gain=bad)
        with pytest.raises(ValueError): training.history_settings({"flow_shift": bad})
    with pytest.raises(ValueError): training.history_settings({"cfg_scale": True})


@pytest.mark.parametrize("overrides,active", [([], True),
    (["--tla-sink", "off", "--tla-replay", "off"], False),
    (["--tla-sink", "protected", "--sink-gain", "0", "--tla-replay", "observed", "--replay-strength", "0"], False)])
def test_evaluation_explicit_off_overrides_trained_memory(cached_sana, overrides, active):
    import argparse
    from worldttn.sink import add_sink_arguments
    from worldttn.replay import add_replay_arguments, replay_options_from_args
    from worldttn.evaluation import validate_inference_interventions
    from worldttn.runtime import TTNRuntimeState
    parser = argparse.ArgumentParser()
    add_sink_arguments(parser); add_replay_arguments(parser)
    args = parser.parse_args(overrides)
    cfg = model(cached_sana).ttn_system.config
    sink = validate_inference_interventions(args, cfg)
    replay = replay_options_from_args(args, cfg)
    with torch.no_grad():
        runtime = TTNRuntimeState.create(cfg, 1, "cpu", sink_options=sink, replay_options=replay)
    assert runtime.sink_options.active is active and runtime.replay_options.active is active
    if active: assert sink.gain == replay.strength == .25


@pytest.mark.parametrize("command", ["train", "stage-evaluate"])
def test_default_cli_reaches_device_guard_without_memory_intervention_error(command, monkeypatch, capsys):
    from worldttn.cli import main
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    arguments = [command, "--device", "cpu"]
    if command == "stage-evaluate":
        arguments += ["--training-run", "unused", "--steps", "4", "--frames", "61", "--fixed-cases", "unused.pt"]
    with pytest.raises(SystemExit): main(arguments)
    error = capsys.readouterr().err
    assert "server entrypoints require CUDA" in error
    assert "mechanism CLI" not in error


def test_resume_identity_records_history_horizon_and_memory_math():
    from types import SimpleNamespace
    from worldttn.cli import _training_identity
    from test_parallel_cli import CPUFlowConfig
    import json
    settings = json.loads((Path(__file__).resolve().parents[2]/"configs/worldttn/full_memory.json").read_text())
    args = SimpleNamespace(seed=3407, batch_file="fixed-batch", train_scope="dit", optimizer_policy="origin")
    cfg = SimpleNamespace(scheduler=CPUFlowConfig())
    original = _training_identity(args, cfg, settings, 4)
    assert original["training_latent_frames"] == 121
    assert original["memory_training"]["replay_strength"] == .25
    assert original["history_training"]["source"] == "generated"
    for key, value in (("source", "gt"), ("steps", 20), ("cached_chunks", -1)):
        changed = dict(settings, history_training=dict(settings["history_training"], **{key: value}))
        assert original != _training_identity(args, cfg, changed, 4)
    assert original != _training_identity(args, cfg, dict(settings, training_latent_frames=25), 4)


def test_camera_window_uses_actual_token_counts_and_frees_evicted_storage(cached_sana):
    from worldttn.session import TTNSession
    from worldttn.core import ANCHORS
    m = model(cached_sana)
    _, _, _, camera = long_inputs()
    session = TTNSession(m, camera, 100, 100)
    session.reset(1)
    cache = [[None]*10 for _ in range(20)]
    # Token counts deliberately differ from image grid assumptions (packed/patchified input).
    for chunk, count in enumerate((8, 6, 6, 6)):
        current = [[None]*10 for _ in range(20)]
        for anchor in ANCHORS:
            current[anchor][2] = torch.full((1, 2, count, 8), float(chunk))
            current[anchor][3] = torch.full((1, 2, count, 8), float(chunk+10))
        marker = torch.tensor(float(chunk))
        current[0][0] = marker
        cache = session.carry_clean_cache(current, cache, cached_chunks=2)
        for anchor in ANCHORS:
            for slot in (2, 3):
                value = cache[anchor][slot]
                assert value.shape[2] == (8 if chunk == 0 else 14 if chunk == 1 else 12)
                assert value.untyped_storage().nbytes() == value.numel()*value.element_size()
        assert cache[0][0] is marker
    assert cache[3][2][0, 0, :, 0].tolist() == [2.]*6+[3.]*6


def test_direct_s_update_meta_derivatives_match_autograd_objective():
    from worldttn.core import write_weights
    from worldttn.replay import Observation, ReplayOptions, correct_with_replay
    torch.manual_seed(84)
    s = torch.randn(1, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 2, 5, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    beta = torch.randn(1, 2, 5, dtype=torch.float64, requires_grad=True)
    hk = torch.randn(1, 2, 3, 4, dtype=torch.float64)
    target = torch.randn_like(hk, requires_grad=True)
    hw = torch.rand(1, 2, 3, dtype=torch.float64)
    mask = torch.ones(1, 5, dtype=torch.bool)
    obs = Observation(hk, target.detach(), hw, torch.zeros(1, 3).long(), torch.zeros(1, 3).long())
    actual, w, _ = correct_with_replay(s, k, v, beta.sigmoid(), mask, ReplayOptions("observed"), obs, target)
    denom = 1e-6+(w*k.square().sum(-1)).sum(-1)
    hdenom = 1e-6+(hw*hk.square().sum(-1)).sum(-1)
    objective = ((w*(k@s-v).square().sum(-1)).sum(-1)/(2*denom)
               + .25*(hw*(hk@s-target).square().sum(-1)).sum(-1)/(2*hdenom))/1.25
    gradient, = torch.autograd.grad(objective.sum(), s, create_graph=True)
    expected = s-.5*gradient
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    args = (s, k, v, beta, target)
    # Subtract the identity path: test only future credit through the update.
    a = torch.autograd.grad((actual-s).square().sum(), args, retain_graph=True)
    e = torch.autograd.grad((expected-s).square().sum(), args)
    for left, right in zip(a, e):
        assert left.norm() > 0
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-12)


def test_installed_diffusers_per_token_schedule_matches_velocity_oracle(cached_sana):
    pytest.importorskip("diffusers")
    # Runs on LTU with its installed scheduler, also checking the native minus sign.
    class Session:
        def forward(self, x, *args, **kwargs): return torch.ones_like(x)*.3, []
    from worldttn.runtime import TTNRuntimeState
    from test_sink import begin
    m = model(cached_sana)
    runtime = TTNRuntimeState.create(m.ttn_system.config, 1, "cpu")
    context = begin(runtime, m.ttn_system, [0, 1, 2, 3])
    noise = torch.randn(1, 16, 4, 1, 1)
    observed = torch.ones_like(noise[:, :, :1])
    output = training.generate_history_chunk(Session(), noise, observed, None, context, [],
        0, 4, None, None, training.history_settings({"source": "generated"}))
    torch.testing.assert_close(output[:, :, 1:], noise[:, :, 1:]-.3, atol=2e-6, rtol=0)
    torch.testing.assert_close(output[:, :, :1], observed, atol=0, rtol=0)


def test_generated_clean_commits_do_not_depend_on_future_gt(monkeypatch, cached_sana):
    monkeypatch.setattr(training, "_history_scheduler", EulerOracle)
    a, b = model(cached_sana), model(cached_sana)
    clean, noise, t, camera = long_inputs()
    altered = clean.clone(); altered[:, :, 1:] += 3
    results = []
    for network, target in ((a, clean), (b, altered)):
        torch.manual_seed(71)
        results.append(training.train_clip(network, target, torch.zeros(1, 1, 2, 8), camera,
            make_optimizer(network), training.linear_flow_loss, t, noise, width=100, height=100,
            tbptt=4, history_training={"source": "generated", "steps": 2, "cached_chunks": 2}))
    assert results[0]["loss"] != results[1]["loss"]  # GT still supplies supervision.
    for name in ("world_state", "transition_fast", "replay_values"):
        torch.testing.assert_close(getattr(results[0]["runtime"], name),
                                   getattr(results[1]["runtime"], name), rtol=0, atol=0)
    for left, right in zip(a.saved_features, b.saved_features):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_generated_history_exact_cpu_resume_restores_sampling_rng(monkeypatch, tmp_path, cached_sana):
    monkeypatch.setattr(training, "_history_scheduler", EulerOracle)
    from worldttn.checkpoint import rng_state
    from tools.ttn_compare_resume import identical
    a = model(cached_sana); opt = make_optimizer(a)
    clean, noise, t, camera = long_inputs()
    def train(network, optimizer):
        return training.train_clip(network, clean, torch.zeros(1, 1, 2, 8), camera, optimizer,
            training.linear_flow_loss, t, noise, width=100, height=100, tbptt=4,
            history_training={"source": "generated", "steps": 2, "cached_chunks": 2})
    train(a, opt)
    path = tmp_path/"last.pt"
    save_checkpoint(path, a, opt, 1)
    expected = train(a, opt)
    expected_rng = rng_state()
    b = model(cached_sana); restored_opt = make_optimizer(b)
    assert load_checkpoint(path, b, restored_opt, resume=True) == 1
    actual = train(b, restored_opt)
    assert actual["loss"] == expected["loss"]
    identical(a.state_dict(), b.state_dict(), "model")
    identical(opt.state_dict(), restored_opt.state_dict(), "optimizer")
    identical(expected_rng, rng_state(), "sampling RNG")


def test_rotation_geometry_distinguishes_radial_and_tangent_gradients():
    from worldttn.stability import rotation_gradient_stats
    identity = torch.eye(2, dtype=torch.float64)[None, None]
    skew = torch.tensor([[0., 1.], [-1., 0.]], dtype=torch.float64)[None, None]
    coeff = torch.zeros(1, 1, 3, dtype=torch.float64)
    for scale in (1., 1e-12):
        radial = rotation_gradient_stats(identity, identity*scale, coeff)
        tangent = rotation_gradient_stats(identity, skew*scale, coeff)
        assert radial["rotation_sensitivity_defined"] == [[True]]
        assert radial["skew_fraction_of_TtH"] == [[0.]]
        assert tangent["skew_fraction_of_TtH"] == [[1.]]
    assert rotation_gradient_stats(identity, identity*0, coeff)["rotation_sensitivity_defined"] == [[False]]


@pytest.mark.skipif(not torch.cuda.is_available() or "META_TEST_OUTPUT" not in os.environ,
                   reason="requires the isolated four-node LTU allocation")
def test_full_memory_four_gpu_future_credit_and_resume(cached_sana):
    import copy
    import torch.distributed as dist
    from diffusers import FlowMatchEulerDiscreteScheduler  # required, never silently skip this gate
    from worldttn.distributed import (ParallelTraining, resolve_launch_environment,
                                      save_training_checkpoint, restore_training_checkpoint)
    from worldttn.training_health import FirstUpdateProbe
    launch = resolve_launch_environment()
    assert launch["world_size"] == 4 and torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method="env://", rank=launch["rank"], world_size=4)
    try:
        m = model(cached_sana).cuda()
        live = []
        def retain(module, args, out):
            context = args[2]
            if context.clean_mode and not context.prefill_mode:
                s, g = context.candidates[0][:2]
                s.retain_grad(); g.retain_grad(); live.append((s, g))
        m.blocks[3].register_forward_hook(retain)
        engine = ParallelTraining(m, training.linear_flow_loss, "fsdp2", activation_offload="cpu")
        opt = make_optimizer(m)
        probe = FirstUpdateProbe(m)
        clean, noise, t, camera = [x.cuda() for x in long_inputs()]
        clean = clean+.01*launch["rank"]
        protocol = {"source": "generated", "steps": 4, "cached_chunks": 2}
        kw = dict(width=100, height=100, tbptt=4, activation_offload="cpu", history_training=protocol)
        def train(network, optimizer, parallel):
            return training.train_clip(network, clean, torch.zeros(1, 1, 2, 8, device="cuda"), camera,
                optimizer, training.linear_flow_loss, t, noise, parallel=parallel, **kw)
        result = train(m, opt, engine)
        health = probe.report(opt)
        assert not health["missing_core_gradients"] and result["runtime"].commit_count == 9
        # Five eta scalars split over four ranks: one rank can own an empty shard.
        eta_delta = torch.tensor(health["groups"]["ttn_system.local_eta_logits"]["delta_norm"]**2, device="cuda")
        dist.all_reduce(eta_delta)
        assert eta_delta > 0
        assert len(live) == 8
        assert all(value.grad is not None and value.grad.norm() > 0 for value in live[4])
        assert all(value.grad is None for value in live[3]+live[7])
        live.clear()
        path = Path(os.environ["META_TEST_OUTPUT"])/"full-memory-resume"/"last.pt"
        identity = {"tbptt": 4, "history_training": training.history_settings(protocol)}
        cursor = {"epoch": 1, "batch_in_epoch": 2}
        save_training_checkpoint(path, engine, opt, 1, cursor, identity)
        expected = train(m, opt, engine)
        state = {name: (value.full_tensor() if hasattr(value, "full_tensor") else value).clone()
                 for name, value in m.state_dict().items()}
        expected_optimizer = copy.deepcopy(opt.state_dict())
        rng = torch.cuda.get_rng_state().clone()
        resumed = model(cached_sana).cuda()
        load_checkpoint(path, resumed)
        resumed_engine = ParallelTraining(resumed, training.linear_flow_loss, "fsdp2", activation_offload="cpu")
        resumed_opt = make_optimizer(resumed)
        step, data = restore_training_checkpoint(path, resumed_engine, resumed_opt, identity)
        assert step == 1 and data == cursor
        actual = train(resumed, resumed_opt, resumed_engine)
        assert actual["loss"] == expected["loss"] and torch.equal(torch.cuda.get_rng_state(), rng)
        for name, value in resumed.state_dict().items():
            torch.testing.assert_close(value.full_tensor() if hasattr(value, "full_tensor") else value,
                                       state[name], rtol=0, atol=0)
        from tools.ttn_compare_resume import identical
        identical(expected_optimizer, resumed_opt.state_dict(), "full-memory optimizer")
    finally:
        dist.destroy_process_group()
