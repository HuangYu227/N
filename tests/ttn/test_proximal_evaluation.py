"""Proximal evaluations retain trained mathematics and publish memory evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from worldttn.core import BASE_REVISION, TTNConfig
from worldttn.evaluation import (load_evaluation_run, memory_evaluation_protocol,
                                 validate_inference_interventions)
from worldttn.stage_evaluation import stage_evaluate_command


def recipe():
    return TTNConfig(stage="C", camera_attention="sana", memory_update="proximal", persistent_update=False)


@pytest.mark.parametrize("ablation", ["no-local", "no-persistent", "no-ttt", "identity"])
def test_proximal_rejects_legacy_controls_before_model_loading(ablation):
    with pytest.raises(ValueError, match="retains checkpoint memory math"):
        validate_inference_interventions(SimpleNamespace(ttn_ablation=ablation), recipe())
    assert not validate_inference_interventions(SimpleNamespace(ttn_ablation="full"), recipe()).active


def test_memory_protocol_records_numerical_recipe_and_capacity_without_quality_claim():
    result = memory_evaluation_protocol(recipe())
    assert result["enabled"]
    assert result["settings"]["memory_kappa"] == 16.
    assert result["settings"]["memory_history_weight"] == .25
    assert result["settings"]["memory_capacity_frames"] == 16
    assert result["settings"]["memory_prefix_frames"] == 10
    assert result["settings"]["memory_recent_frames"] == 4
    assert result["settings"]["memory_position"] == "absolute"
    assert not memory_evaluation_protocol(TTNConfig())["enabled"]
    json.dumps(result, allow_nan=False)


def test_stage_proximal_keeps_paired_rollouts_and_teacher_without_psi_controls(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "run_config.json").write_text(json.dumps({"training": {"meta_ttt": recipe().to_dict()}}))
    args = SimpleNamespace(output=str(tmp_path / "evaluation"), training_run=str(snapshot), frames=121,
        fixed_cases=str(tmp_path / "cases.pt"), seed=3407, eval_cases=1, cross_attn_backend="math", device="cpu",
        steps=20, cfg_scale=4.5, cached_blocks=2)
    calls = []
    memory = memory_evaluation_protocol(recipe())
    def run(command, **kwargs):
        calls.append(command)
        assert "--ttn-ablation" not in command and "--eval-methods" not in command
        out = Path(command[command.index("--output") + 1])
        out.mkdir()
        row = {"case_id": "scene", "seed": 3407, "method": "ttn", "memory_prefix_sha256": ["hash"] * 5,
               "memory_prefix_verified": True, "memory_storage_bytes": 100,
               "diagnostic_summary": {}, "matched_backbone_softmax": {"diagnostic_summary": {}}}
        protocol = {"step": 25, "stage": "C", "checkpoint_sha256": "c", "fixed_cases_sha256": "f",
                    "tla_memory": memory, "rollout_sampler": {"steps": 20, "cfg_scale": 4.5, "cached_blocks": 2}}
        (out / "summary.json").write_text(json.dumps({"protocol": protocol, "episodes": [row], "metrics": {}}))
    monkeypatch.setattr("subprocess.run", run)
    monkeypatch.setattr("worldttn.provenance.implementation_identity", lambda: {"git_commit": "test"})
    monkeypatch.setattr("tools.ttn_eval_snapshot.release_evaluation_model", lambda *args: {"status": "retained"})
    stage_evaluate_command(args)
    summary = json.loads((tmp_path / "evaluation/summary.json").read_text())
    assert len(calls) == 3
    assert summary["expected_children"] == ["long", "short", "align"]
    assert summary["status"] == "completed"
    assert summary["results"]["long"]["tla_memory"] == memory
    assert summary["results"]["short"]["memory_audits"][0]["memory_prefix_verified"] is True
    assert summary["results"]["align"]["teacher_diagnoses"] == [{"original": {}, "matched": {}}]


@pytest.mark.parametrize("proximal", [False, True])
def test_checkpoint_controls_evaluation_math_even_when_cli_recipe_differs(tmp_path, monkeypatch, proximal):
    config = recipe() if proximal else TTNConfig(stage="C", camera_attention="sana")
    (tmp_path / "run_config.json").write_text(json.dumps({"arguments": {}, "training": {}, "base": {"sha256": "base"}}))
    (tmp_path / "train.jsonl").write_text(json.dumps({"stage": "C", "step": 25}))
    torch.save({"format": "TTN-SANA-WM-v0.1", "base_revision": BASE_REVISION, "base_sha256": "base",
                "stage": "C", "step": 25, "config": config.to_dict()}, tmp_path / "last.pt")
    other_recipe = TTNConfig() if proximal else recipe()
    monkeypatch.setattr("worldttn.cli.read_reference", lambda *args: (other_recipe, {"sana_config": "test.yaml"}))
    monkeypatch.setattr("worldttn.evaluation.evaluation_config", lambda *args: SimpleNamespace())
    args = SimpleNamespace(training_run=str(tmp_path), adapter=None, stage=None, config="other-recipe.json",
        sana_config=None, dataset_root=None, data_dir=None, vae_cache_dir=None)
    _, _, restored, *_ = load_evaluation_run(args)
    assert restored == config


def test_proximal_resume_identity_and_stage_children_use_full_recipe():
    from worldttn.cli import _training_identity
    from test_parallel_cli import CPUFlowConfig
    from tools.ttn_eval_snapshot import expected_evaluation_children
    root = Path(__file__).resolve().parents[2]
    settings = json.loads((root / "configs/worldttn/proximal_memory.json").read_text())
    acceptance = json.loads((root / "configs/worldttn/proximal_memory_acceptance.json").read_text())
    assert acceptance == dict(settings, training_latent_frames=25)
    args = SimpleNamespace(seed=3407, batch_file="fixed", train_scope="dit", optimizer_policy="origin")
    config = SimpleNamespace(scheduler=CPUFlowConfig())
    identity = _training_identity(args, config, settings, 4)
    assert identity["ttn"] == TTNConfig(**settings["ttn"]).to_dict()
    assert expected_evaluation_children({"training": identity}) == {"long", "short", "align"}
    for key, value in (("memory_kappa", 8), ("memory_history_weight", .5),
                       ("memory_position", "temporal-realign"), ("memory_selection", "fifo"),
                       ("memory_capacity_frames", 17), ("memory_start_chunk", 4)):
        changed = dict(settings, ttn=dict(settings["ttn"], **{key: value}))
        assert _training_identity(args, config, changed, 4) != identity
