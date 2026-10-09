from dataclasses import dataclass, replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from worldttn.evaluation import diagnostic_noise, fixed_data_identity, load_evaluation_cases
from worldttn.provenance import camera_contract
from worldttn.stage_evaluation import stage_evaluate_command


def test_default_cli_uses_restored_camera_but_legacy_config_is_still_explicitly_available():
    from worldttn.cli import DEFAULT_REFERENCE, ROOT, read_reference
    current, _ = read_reference(DEFAULT_REFERENCE)
    legacy, _ = read_reference(ROOT / "configs/worldttn/reference.json")
    assert current.stage == "C" and current.camera_attention == "sana"
    assert legacy.camera_attention == "linear"


def test_camera_contract_requires_labelled_ablation_and_retains_weight_source():
    assert camera_contract("sana")["camera_ablation"] is False
    with pytest.raises(ValueError, match="camera-ablation"): camera_contract("linear", "sana")
    policy = camera_contract("linear", "sana", ablation=True)
    assert policy["camera_ablation"] and "checkpoint" in policy["camera_weight_source"]


def test_short_long_noise_are_exact_prefixes_without_rng_side_effect():
    state = torch.get_rng_state().clone()
    long = diagnostic_noise((1, 4, 61, 2, 2), "cpu", 3407, 61)
    short = diagnostic_noise((1, 4, 13, 2, 2), "cpu", 3407, 61)
    torch.testing.assert_close(short, long[:, :, :13], rtol=0, atol=0)
    assert torch.equal(state, torch.get_rng_state())
    with pytest.raises(ValueError): diagnostic_noise((1, 4, 61, 2, 2), "cpu", 3407, 13)


def test_fixed_cases_are_prefixes_not_reselected_when_training_horizon_changes(tmp_path):
    @dataclass
    class Data:
        num_frames: int = 97
        vae_ratio: tuple = (8, 32)
    @dataclass
    class Text:
        name: str = "unchanged"
    config = SimpleNamespace(data=Data(), text_encoder=Text())
    camera = torch.cat([torch.eye(4).flatten().expand(1, 61, 16), torch.ones(1, 61, 4)], -1)
    case = {"case_id": "fixed", "reference": torch.randn(1, 4, 61, 2, 2), "camera": camera,
            "plucker": torch.randn(6, 61, 2, 2), "prompt": "fixed", "info": {}, "seed": 3407}
    fixed = tmp_path / "cases.pt"
    torch.save({"format": "TTN-fixed-cases-v1", "seed": 3407, "data_identity": fixed_data_identity(config),
                "cases": [case], "rejected": []}, fixed)
    args = SimpleNamespace(fixed_cases=str(fixed), seed=3407, eval_cases=1, frames=13,
        revisit_min_gap=30, revisit_distance_fraction=.02, revisit_angle_deg=5., revisit_max_pairs=5)
    config.data = replace(config.data, num_frames=481)
    result, _, _ = load_evaluation_cases(config, args)
    torch.testing.assert_close(result[0]["reference"], case["reference"][:, :, :13], rtol=0, atol=0)
    assert result[0]["plucker"].shape[1] == 13
    args.seed = 42
    with pytest.raises(ValueError, match="identity"): load_evaluation_cases(config, args)
    args.seed = 3407
    config.text_encoder = Text("changed")
    with pytest.raises(ValueError, match="configuration"): load_evaluation_cases(config, args)


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("meta", [False, True])
@pytest.mark.parametrize("frames", [61, 121])
def test_periodic_evaluation_uses_private_subprocesses_and_publishes_progress(tmp_path, monkeypatch, failure, meta, frames):
    snapshot = tmp_path / "immutable-snapshot"
    snapshot.mkdir()
    (snapshot / "run_config.json").write_text(json.dumps({"training": {"meta_ttt": {"local_update": meta, "persistent_meta": meta}}}))
    args = SimpleNamespace(output=str(tmp_path / "eval"), training_run=str(snapshot), frames=frames,
        fixed_cases=str(tmp_path / "cases.pt"), seed=3407, eval_cases=1, cross_attn_backend="math", device="cpu",
        steps=20, cfg_scale=4.5, cached_blocks=2)
    calls = []
    def run(command, **kw):
        calls.append(command)
        assert command[command.index("--training-run") + 1] == str(snapshot)
        assert command[command.index("--noise-frames") + 1] == str(frames)
        if len(calls) == 1: assert command[command.index("--frames") + 1] == str(frames)
        assert "--camera-attention" not in command and "--resume" not in command
        out = Path(command[command.index("--output") + 1])
        if failure and out.name == "short": raise subprocess.CalledProcessError(1, command)
        out.mkdir()
        row = {"diagnostic_summary": {"text": "original"}, "matched_backbone_softmax": {"diagnostic_summary": {"text": "matched"}}}
        (out / "summary.json").write_text(json.dumps({"protocol": {"step": 25, "stage": "C", "checkpoint_sha256": "c", "fixed_cases_sha256": "f"},
            "episodes": [row], "metrics": {}, "ttn_minus_sana": {}}))
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("worldttn.provenance.implementation_identity", lambda: {"git_commit": "test"})
    if failure:
        with pytest.raises(subprocess.CalledProcessError): stage_evaluate_command(args)
    else: stage_evaluate_command(args)
    result = json.loads((tmp_path / "eval/summary.json").read_text())
    assert result["status"] == ("failed" if failure else "completed")
    expected = {"long", "short", "align"} | ({"no-local", "no-persistent"} if meta else set())
    assert len(calls) == (2 if failure else len(expected))
    assert set(result["results"]) == ({"long"} if failure else expected)
    if meta and not failure:
        for command in calls[3:]:
            assert command[command.index("--frames") + 1] == "13"
            assert command[command.index("--ttn-ablation") + 1] in ("no-local", "no-persistent")


@pytest.mark.parametrize("frames,extra,expected", [
    (61, [], "server entrypoints require CUDA"),
    (121, [], "server entrypoints require CUDA"),
    (58, [], "stage-evaluate requires"),
    (120, [], "1+3n latent frames"),
    (121, ["--adapter", "unused.pt"], "stage-evaluate requires"),
    (121, ["--fixed-cases", ""], "stage-evaluate requires"),
])
def test_stage_cli_accepts_long_horizon_and_retains_protocol_guards(monkeypatch, capsys, frames, extra, expected):
    from worldttn.cli import main
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    arguments = ["stage-evaluate", "--device", "cpu", "--training-run", "unused", "--steps", "20",
                 "--frames", str(frames), "--fixed-cases", "unused.pt", *extra]
    with pytest.raises(SystemExit): main(arguments)
    assert expected in capsys.readouterr().err
