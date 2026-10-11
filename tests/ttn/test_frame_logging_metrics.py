import json
from types import SimpleNamespace

import pytest
import torch

from test_sequence_training import batch, cached_ffn, cached_sana, sequence_model, text_attention
from test_training_metrics import record
from tools import ttn_training_metrics as metrics
from worldttn import cli
from worldttn.stability import progress_line, stability_rows
from worldttn.training import linear_flow_loss, train_clip


def test_full_sequence_result_reaches_cli_stability_and_progress(sequence_model, monkeypatch):
    clean, noise, timesteps, camera = batch()
    model = sequence_model
    model.base_load_report = {}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    result = train_clip(model, clean, torch.ones(1, 1, 2, 16), camera, optimizer,
        linear_flow_loss, timesteps, noise, width=100, height=100, tbptt=0,
        mask=torch.ones(1, 2, dtype=torch.bool))
    monkeypatch.setattr(cli, "execution_report", lambda model: {})
    parallel = SimpleNamespace(rank=0, world=1, mode="single")
    row = dict(step=1, stage="C", tbptt=0,
        **cli._training_record(model, result, {"seconds": 1., "peak_allocated_bytes": 0}, parallel))
    rows = list(stability_rows(row))
    assert len(rows) == 15  # Five observed anchors and two predicted groups.
    observed = [r for r in rows if r["prefill"]]
    assert [r["block"] for r in observed] == [3, 7, 11, 15, 19]
    assert all((r["chunk"], r["start"], r["end"]) == (-1, 0, 1) for r in observed)
    assert all(r["proximal"]["frames"][0]["frame_ids"] == [[0]] for r in observed)
    assert result["prefill"]["live_gradient"] and result["prefill"]["mode"] == "isolated_in_each_layer"
    assert "step 1/500" in progress_line(row, 500)
    json.dumps(row, allow_nan=False)


def frame_record():
    row = record(2)
    row.update(tbptt=0, config={"memory_granularity": "frame", "memory_frame_kappa": 49})
    rank = row["ranks"][0]
    rank["commits"] = 0
    anchor = rank["chunks"][0]["anchors"][0]
    rank["chunks"][0].update(start=1, end=4, history_source="noisy")
    anchor.pop("proximal_trajectory")
    anchor["proximal"] = {"implementation": "frame_proximal_s", "output_effect": {"relative_delta": .3},
        "frames": [{"frame_ids": [[i]], "history_active": [i >= 13], "per_head": {
            "inner_objective_before": [[value]], "inner_objective_after": [[value/2]],
            "current_residual_mse_before": [[value]], "current_residual_mse_after": [[value/4]]}}
            for i, value in ((1, 2.), (2, 4.), (3, 6.))]}
    return row


def test_foreground_metrics_read_run_config_without_chain_and_frame_effects(tmp_path, capsys):
    (tmp_path / "run_config.json").write_text(json.dumps({"training": {
        "tbptt": 0, "training_protocol": "frame-noisy-fullgrad-v1", "training_latent_frames": 121}}))
    (tmp_path / "train.jsonl").write_text(json.dumps(frame_record()) + "\n")
    metrics.report(tmp_path)
    out = capsys.readouterr().out
    assert "frame-noisy-fullgrad-v1" in out
    assert "TBPTT： 0" in out and "帧/head" in out
    assert "1 | 3 | 0.1 | 2 | 50 | -50 | -75 | null | null | null | null | 30" in out
    assert "noisy | 1 | 3" in out and "noisy | 3 | 3" in out
    assert "[(0, 1, 0)]" in out  # No inference commits are invented for sequence training.


def test_frame_summary_aggregates_every_frame_instead_of_last_frame():
    row = frame_record()
    frames = row["ranks"][0]["chunks"][0]["anchors"][0]["proximal"]["frames"]
    frames[-1]["per_head"]["inner_objective_after"] = [[6.]]
    summary = next(metrics.anchor_rows(row))
    assert summary[5] == pytest.approx(-25.)
    assert summary[-1] == 30.


def test_sequence_cache_payload_aggregates_anchors_and_preserves_missing():
    chunk = {"anchors": [{"memory_storage_bytes": 12}, {"memory_storage_bytes": 34}]}
    assert metrics.cache_bytes(chunk) == 46
    chunk["anchors"][1].pop("memory_storage_bytes")
    assert metrics.cache_bytes(chunk) is None
    chunk["memory_storage_bytes"] = 99
    assert metrics.cache_bytes(chunk) == 99
