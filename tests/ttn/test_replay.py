"""Observed-content vs write-dilution controls and clean-only cache lifecycle."""
from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch

from test_alignment_detail import cached_sana
from test_sink import begin, config
from worldttn.anchor import TTNAnchor
from worldttn.core import correct, cayley_dense, write_weights
from worldttn.replay import (ReplayOptions, Observation, capture_observation, observation_digest,
                            correct_with_replay, validate_replay)
from worldttn.runtime import TTNRuntimeState, TTNSystem
from worldttn.session import record_noise


def test_replay_fp64_oracle_zero_strength_and_matched_dilution():
    torch.manual_seed(72)
    pred = torch.randn(2, 2, 8, 8, dtype=torch.float64)
    k, v = (torch.randn(2, 2, 7, 8, dtype=torch.float64) for _ in range(2))
    beta, mask = torch.rand(2, 2, 7, dtype=torch.float64), torch.tensor([[1]*5+[0]*2, [1]*7]).bool()
    hk, hv = (torch.randn(2, 2, 4, 8, dtype=torch.float64) for _ in range(2))
    hw = torch.rand(2, 2, 4, dtype=torch.float64)
    obs = Observation(hk, hv, hw, torch.zeros(2, 4).long(), torch.zeros(2, 4).long())
    baseline, w = correct(pred, k, v, beta, mask)
    for mode in ("off", "shrink", "observed"):
        zero, _, _ = correct_with_replay(pred, k, v, beta, mask, ReplayOptions(mode, 0.))
        torch.testing.assert_close(zero, baseline, atol=0, rtol=0)
    live = pred.clone().requires_grad_()
    cd, hd = 1e-6+(w*k.square().sum(-1)).sum(-1), 1e-6+(hw*hk.square().sum(-1)).sum(-1)
    def objective(key, value, weight, denom):
        return (weight[..., None]*(key@live-value).square()).sum((-2, -1))/(2*denom)
    current = objective(k, v, w, cd)
    for mode in ("shrink", "observed"):
        loss = (current + (.25*objective(hk, hv, hw, hd) if mode == "observed" else 0))/1.25
        gradient, = torch.autograd.grad(loss.sum(), live, retain_graph=True)
        actual, _, stats = correct_with_replay(pred, k, v, beta, mask, ReplayOptions(mode), obs, hv,
                                              collect_stats=True)
        torch.testing.assert_close(actual, pred-.5*gradient, atol=2e-15, rtol=2e-15)
        assert stats[0]["current_scale"] == .8
    # History-free dilution and replay with exactly zero history gradient coincide.
    neutral = hk@pred
    a, _, _ = correct_with_replay(pred, k, v, beta, mask, ReplayOptions("observed"), obs, neutral)
    b, _, _ = correct_with_replay(pred, k, v, beta, mask, ReplayOptions("shrink"))
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_capture_cfg_masks_shared_head_indices_no_rng_or_aliasing():
    k = torch.arange(2*2*6*8).reshape(2, 2, 6, 8).float()
    v, w = k+1, torch.ones(2, 2, 6)
    write = torch.tensor([[1, 0, 1, 0, 1, 1], [0, 1, 1, 1, 0, 1]]).bool()
    rng = torch.get_rng_state().clone()
    obs = capture_observation(k, v, w, write, torch.zeros(2, 1).long(), 3)
    assert torch.equal(rng, torch.get_rng_state()) and obs.key.shape == (2, 2, 3, 8)
    assert obs.token_indices.tolist() == [[0, 4, 5], [1, 3, 5]]
    assert (obs.frame_ids == 0).all()
    digest = observation_digest((obs,))
    k.zero_(); v.zero_(); w.zero_()
    assert observation_digest((obs,)) == digest and not torch.equal(obs.key[0], obs.key[1])
    with pytest.raises(ValueError, match="valid prefill"):
        capture_observation(k, v, w, torch.zeros_like(write), torch.zeros(2, 1).long(), 3)


@torch.no_grad()
def test_real_anchors_prefix_activation_local_transport_and_atomic_clean_commit(cached_sana):
    torch.manual_seed(12)
    cfg = config()
    system = TTNSystem(cfg)
    for head in system.controller.heads:
        head.weight.normal_(std=.04)
    anchors = [TTNAnchor(cached_sana(), i, cfg) for i in range(5)]
    plain = TTNRuntimeState.create(cfg, 2, "cpu")
    runtime = TTNRuntimeState.create(cfg, 2, "cpu", replay_options=ReplayOptions("observed", budget=3))
    for state in (plain, runtime):
        pre = begin(state, system, [0], prefill=True).for_clean()
        x = torch.arange(2*4*16).reshape(2, 4, 16).float()/50
        for anchor in anchors: anchor(x, HW=(1, 2, 2), ttn_chunk_context=pre)
        state.prefill(pre)
    digest = runtime.verify_replay_reference()
    original = [(o.key.clone(), o.value.clone(), o.weight.clone()) for o in runtime.replay_observations]
    assert "replay" not in system.state_dict() and runtime.replay_values.shape == (2, 5, 2, 3, 8)
    for chunk, ids in enumerate(([0, 1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15])):
        x = torch.randn(2, len(ids)*2, 16)
        contexts = [begin(state, system, ids) for state in (plain, runtime)]
        ctx = contexts[1]
        assert ctx.replay_active is (chunk == 4)
        torch.testing.assert_close(ctx.replay_values, runtime.replay_values @ cayley_dense(
            system.generators.u, system.generators.v, ctx.cbase+ctx.psi.tanh()), atol=1e-6, rtol=1e-5)
        saved = runtime.replay_values.clone()
        for call in range(4):
            record_noise(ctx, torch.tensor([800.-100*call, 800.-100*call]), total_calls=4)
            for i, anchor in enumerate(anchors):
                seen = {}
                anchor(x, HW=(len(ids), 1, 2), ttn_chunk_context=ctx,
                       ttn_diagnostic=lambda key, value: seen.update({key: value}))
                if chunk == 4 and call == 0:
                    # Independent dense/autograd oracle for the Local target rotation.
                    q, k, v = seen["visual_features"]
                    write = ctx.token_masks(x.shape[1])[1]
                    support, _ = ctx.support_query_masks(x.shape[1], (len(ids), 1, 2))
                    w = write_weights(anchor.beta_proj(x).sigmoid().transpose(1, 2), write, cfg.eps)
                    with torch.enable_grad():
                        z = torch.zeros_like(ctx.psi[:, i], requires_grad=True)
                        coeff = ctx.cbase[:, i]+ctx.psi[:, i].tanh()
                        predicted = ctx.previous[:, i] @ cayley_dense(system.generators.u[i], system.generators.v[i], coeff+z.tanh())
                        sw = torch.where(support[:, None], w, 0.)
                        loss = (sw*(k@predicted-v).square().sum(-1)).sum(-1)/(cfg.eps+(sw*v.square().sum(-1)).sum(-1))
                        g, = torch.autograd.grad(loss.sum(), z)
                    clipped = g*(1/g.norm(dim=-1, keepdim=True).clamp_min(cfg.eps)).clamp_max(1)
                    eta = torch.nn.functional.softplus(system.local_eta_logits[i])
                    rotation = cayley_dense(system.generators.u[i], system.generators.v[i], coeff+(-eta*clipped).tanh())
                    target = runtime.replay_values[:, i]@rotation
                    obs = runtime.replay_observations[i]
                    effective = seen["memory"][0]
                    expected, _, _ = correct_with_replay(effective, k, v, anchor.beta_proj(x).sigmoid().transpose(1, 2),
                        write, runtime.replay_options, obs, target)
                    torch.testing.assert_close(seen["memory"][1], expected, atol=1e-6, rtol=1e-5)
            assert torch.equal(runtime.replay_values, saved) and runtime.commit_count == chunk+1
            assert runtime.verify_replay_reference() == digest and not ctx.candidates
        clean = [c.for_clean() for c in contexts]
        for i, anchor in enumerate(anchors):
            a = anchor(x, HW=(len(ids), 1, 2), ttn_chunk_context=clean[0])
            b = anchor(x, HW=(len(ids), 1, 2), ttn_chunk_context=clean[1])
            if chunk < 4: torch.testing.assert_close(a, b, atol=0, rtol=0)
        if chunk == 4:
            candidate = clean[1].candidates.pop(4)
            with pytest.raises(RuntimeError, match="five anchors"): runtime.commit_chunk(clean[1])
            assert torch.equal(runtime.replay_values, saved) and runtime.commit_count == 5
            clean[1].candidates[4] = candidate
        plain.commit_chunk(clean[0]); runtime.commit_chunk(clean[1])
        if chunk < 4:
            torch.testing.assert_close(plain.world_state, runtime.world_state, atol=0, rtol=0)
            torch.testing.assert_close(plain.transition_fast, runtime.transition_fast, atol=0, rtol=0)
        else:
            assert any(a["replay"]["effective_delta_norm"] > 0 for a in runtime.last_stats["anchors"])
            assert [c["call"] for c in runtime.last_stats["anchors"][0]["replay_trajectory"]] == [0, 2, 3]
        for old, obs in zip(original, runtime.replay_observations):
            assert all(torch.equal(a, b) for a, b in zip(old, (obs.key, obs.value, obs.weight)))
        assert runtime.verify_replay_reference() == digest
    assert runtime.commit_count == 6
    runtime.replay_observations[0].key[0, 0, 0, 0] += 1
    with pytest.raises(RuntimeError, match="changed"): runtime.verify_replay_reference()
    runtime.reset()
    assert not runtime.replay_observations and runtime.replay_values is None


def test_replay_guards_keep_training_history_sink_and_execution_separate(monkeypatch, capsys):
    from worldttn.cli import main
    from worldttn.evaluation import validate_inference_interventions
    from worldttn.performance import ExecutionOptions
    cfg = config()
    with pytest.raises(ValueError, match="inference-only"):
        TTNRuntimeState.create(cfg, 1, "cpu", replay_options=ReplayOptions("observed"))
    with torch.no_grad():
        with pytest.raises(ValueError, match="separate"):
            from worldttn.sink import SinkOptions
            TTNRuntimeState.create(cfg, 1, "cpu", replay_options=ReplayOptions("observed"),
                                   sink_options=SinkOptions("protected"))
        with pytest.raises(ValueError, match="reference/reference"):
            validate_replay(ReplayOptions("observed"), cfg, execution=ExecutionOptions("reuse", "projected"))
    for values in ({"history_source": "native-gt"}, {"tla_sink": "protected"}, {"eval_methods": ["sana"]},
                   {"camera_ablation": True}, {"ttn_ablation": "no-local"}):
        with pytest.raises(ValueError):
            validate_inference_interventions(Namespace(tla_replay="observed", **values), cfg)
    monkeypatch.setattr("sys.argv", ["worldttn", "train", "--tla-replay", "observed"])
    with pytest.raises(SystemExit) as error: main()
    assert error.value.code == 2 and "evaluate-only" in capsys.readouterr().err


@pytest.mark.parametrize("changes", [{"strength": float("nan")}, {"strength": -1}, {"strength": True},
                                      {"budget": 0}, {"budget": 1.5}, {"start_chunk": False}])
def test_replay_invalid_options(changes):
    with pytest.raises(ValueError): ReplayOptions(**changes)


@pytest.fixture
def replay_suite(tmp_path):
    """Small saved latents with known matched contrasts, not simulated GPU results."""
    from test_sink_suite import snapshot
    from worldttn.evaluation import latent_metrics, file_sha256
    from worldttn.training import chunk_ranges
    from tools import ttn_submit_sink as tool
    directory = snapshot(tmp_path)
    cases = tmp_path / "fixed-cases.pt"
    torch.save({"cases": []}, cases)
    evaluation = tmp_path / "evaluation"
    (evaluation / "long").mkdir(parents=True)
    metadata = json.loads((directory / "snapshot.json").read_text())
    metadata["source"] = str(tmp_path)
    (directory / "snapshot.json").write_text(json.dumps(metadata))
    protocol = dict(training_run=str(directory), checkpoint_sha256=file_sha256(directory / "last.pt"),
        fixed_cases_sha256=file_sha256(cases), step=100, stage="C", frames=61, noise_frames=61,
        steps=4, cfg_scale=4.5, cached_blocks=2, seed=3407, ttn_camera_attention="sana",
        cross_attn_backend="math", history_source="generated", ttn_ablation="full",
        state_diagnostics=True, flow_shift=9.8, tla_sink={"mode": "off"})
    (evaluation / "summary.json").write_text(json.dumps({"status": "completed", "results": {
        "long": {"metrics": {"mean_future_latent_mse": {"paired_count": 1}}}}}))
    (evaluation / "long/manifest.json").write_text('{"cases":[{"seed":3407}]}')
    (evaluation / "long/summary.json").write_text(json.dumps({"protocol": protocol}))
    plan = tool.prepare(evaluation=evaluation, output=tmp_path / "replay-suite", profile="observed-replay")
    root = Path(plan["output"])
    root.mkdir()
    (root / "plan.json").write_text(json.dumps(plan))
    identity = dict(case_id="fixture/scene", seed=3407, input_sha256={"latent": "same"},
                    initial_noise_sha256="noise", base_sha256="base")
    gt = torch.zeros(1, 1, 61, 1, 1)
    for name, tail in (("full", 2.), ("shrink-025", 1.8), ("observed-025", 1.6)):
        folder = root / name
        folder.mkdir()
        episodes = []
        for method in (("sana", "ttn") if name == "full" else ("ttn",)):
            latent = torch.ones_like(gt)
            latent[:, :, 0] = 0
            if method == "ttn": latent[:, :, 13:] = tail
            spec = plan["interventions"][name]
            chunks = []
            for index, (start, end) in enumerate(chunk_ranges(61)):
                anchors = []
                for block in (3, 7, 11, 15, 19):
                    measurement = dict(active=index >= 4 and name != "full", mode=spec["mode"])
                    if measurement["active"]:
                        measurement.update(strength=.25, current_scale=.8, key_position="original-absolute-rope",
                            value_transport="actual-current-cayley", source="observed-prefill-only",
                            samples=128 if name == "observed-025" else 0, effective_delta_norm=1.,
                            per_head={"history_gradient_norm": [[.5 if name == "observed-025" else 0.]]})
                    anchors.append(dict(block=block, replay=measurement,
                        replay_trajectory=[dict(measurement, call=call) for call in (0, 2, 3)]
                            if measurement["active"] else []))
                chunks.append(dict(chunk=index, start=start, end=end, anchors=anchors))
            row = dict(identity, method=method, chunks=chunks, commits=21, predictions=21, timing={"seconds": 1},
                metrics=latent_metrics(latent, gt, [], training_frames=13),
                prefix_13_metrics=latent_metrics(latent[:, :, :13], gt[:, :, :13], [], training_frames=13))
            if name == "observed-025":
                row.update(replay_reference_sha256="a"*64, replay_reference_verified=True, replay_storage_bytes=42)
            episodes.append(row)
            torch.save(dict(latents=latent, method=method, case_id=row["case_id"], seed=row["seed"]),
                       folder / f"case-000-{method}.pt")
        result = dict(protocol={**protocol, "tla_replay": spec}, episodes=episodes)
        (folder / "summary.json").write_text(json.dumps(result))
        (root / f"{name}-status.json").write_text('{"status":"completed"}')
        if name == "full":
            (evaluation / "long/summary.json").write_text(json.dumps(result))
    return root, plan


def test_replay_suite_matches_content_control_and_keeps_single_gpu_dependency(replay_suite, tmp_path, monkeypatch):
    from tools import ttn_submit_sink as tool
    root, plan = replay_suite
    assert plan["variants"] == ["full", "shrink-025", "observed-025"] and plan["array"] == "1-2%2"
    result = tool.collect(root)
    assert result["status"] == "completed"
    assert result["results"]["observed-025"]["replay_audit"][0]["history_gradient_anchors"] == [3, 7, 11, 15, 19]
    contrast = result["contrasts"]["observed-025_minus_shrink-025"]["long"]
    assert contrast["after_training_horizon_mean_latent_mse"] == pytest.approx(1.6**2-1.8**2)
    assert contrast["revisit_return_gt_latent_mse"] is None
    calls = []
    def sbatch(command, **kwargs):
        calls.append(command)
        return Namespace(stdout=f"{900+len(calls)}\n")
    monkeypatch.setattr(tool.subprocess, "run", sbatch)
    jobs = tool.submit(dict(plan, output=str(tmp_path / "submit")), partition="short", root=tmp_path)
    assert jobs == {"baseline": "901", "interventions": "902"}
    assert "--dependency=afterok:901" in calls[1] and "--array=1-2%2" in calls[1]
    assert all("--partition=short" in c and "--time=01:00:00" in c and "--job-name=ttn-replay" in c for c in calls)


@pytest.mark.parametrize("corrupt", ["early", "transport", "key", "dilution", "empty-output",
    "empty-history", "missing-call", "hash", "prefix", "sink"])
def test_replay_collector_rejects_confounded_or_disconnected_result(replay_suite, corrupt):
    from tools.ttn_submit_sink import collect
    root, _ = replay_suite
    path = root / "observed-025/summary.json"
    summary = json.loads(path.read_text())
    row = summary["episodes"][0]
    first = row["chunks"][4]["anchors"][0]
    if corrupt == "early": row["chunks"][0]["anchors"][0]["replay"]["active"] = True
    elif corrupt == "transport": first["replay"]["value_transport"] = "base-without-local"
    elif corrupt == "key": first["replay"]["key_position"] = "rewritten"
    elif corrupt == "dilution": first["replay"]["current_scale"] = 1.
    elif corrupt in ("empty-output", "empty-history"):
        for chunk in row["chunks"][4:]:
            for anchor in chunk["anchors"]:
                for call in anchor["replay_trajectory"]:
                    if corrupt == "empty-output": call["effective_delta_norm"] = 0.
                    else: call["per_head"]["history_gradient_norm"] = [[0.]]
    elif corrupt == "missing-call": first["replay_trajectory"].pop()
    elif corrupt == "hash": row["replay_reference_verified"] = False
    elif corrupt == "prefix":
        row["metrics"]["per_frame_latent_mse"][1] += .1
        row["prefix_13_metrics"]["per_frame_latent_mse"][1] += .1
    elif corrupt == "sink": summary["protocol"]["tla_sink"] = {"mode": "protected", "gain": .1}
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError): collect(root)


@pytest.mark.parametrize("changed_prefix", [False, True])
def test_replay_worker_forwards_options_and_checks_actual_prefix_before_success(replay_suite, monkeypatch, changed_prefix):
    from tools import ttn_submit_sink as tool
    from worldttn import cli
    root, plan = replay_suite
    (root / "observed-025-status.json").write_text('{"status":"pending"}')
    seen = []
    def infer(args):
        seen.extend(args)
        if changed_prefix:
            path = root / "observed-025/case-000-ttn.pt"
            payload = torch.load(path, weights_only=True)
            payload["latents"][:, :, 1] += .01
            torch.save(payload, path)
    monkeypatch.setattr(cli, "main", infer)
    if changed_prefix:
        with pytest.raises(AssertionError, match="first four chunks"): tool.run_variant(root, 2)
        status = json.loads((root / "observed-025-status.json").read_text())
        assert status["status"] == "failed" and "first four chunks" in status["first_exception"]
    else:
        assert tool.run_variant(root, 2)["status"] == "completed"
        status = json.loads((root / "observed-025-status.json").read_text())
        assert status["prefix_13_validation"]["cases"][0]["max_abs_difference"] == 0
    for flag, value in (("--tla-replay", "observed"), ("--replay-strength", "0.25"),
                        ("--replay-budget", "128"), ("--replay-start-chunk", "5"), ("--frames", "61")):
        assert seen[seen.index(flag)+1] == value
