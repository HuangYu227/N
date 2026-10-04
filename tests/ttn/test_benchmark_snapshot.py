import json
import os
import pytest
import torch


def test_benchmark_snapshot_pins_weights_optimizer_rng_and_cursor(tmp_path):
    from tools.ttn_benchmark_snapshot import snapshot_training_bundle
    source = tmp_path / "training"; source.mkdir()
    folder = source / "last-resume-00000025-abc"; folder.mkdir()
    payload = {"step": 25, "stage": "C", "base_sha256": "base", "train_scope": "ttn-visual",
               "distributed": {"resume_dir": folder.name, "world_size": 2, "checkpoint_id": "abc"}}
    torch.save(payload, source / "last.pt")
    for rank in range(2):
        torch.save({"rank": rank, "step": 25, "rng": torch.get_rng_state(), "data": {"cursor": 13},
                    "optimizer": {"value": torch.ones(3)}}, folder / f"rank-{rank:05d}.pt")
    (source / "run_config.json").write_text(json.dumps({"base": {"sha256": "base"}}))
    (source / "train.jsonl").write_text(json.dumps({"stage": "C", "step": 25}) + "\n" + json.dumps({"stage": "C", "step": 26}) + "\n")
    target = tmp_path / "pinned"
    snapshot_training_bundle(source, target)
    torch.save({"step": 26}, source / "next.pt")
    os.replace(source / "next.pt", source / "last.pt")
    assert torch.load(target / "last.pt", weights_only=False)["step"] == 25
    for rank in range(2):
        shard = torch.load(target / folder.name / f"rank-{rank:05d}.pt", weights_only=False)
        assert shard["data"] == {"cursor": 13} and torch.equal(shard["optimizer"]["value"], torch.ones(3))
    assert json.loads((target / "train.jsonl").read_text())["step"] == 25
    with pytest.raises(FileExistsError): snapshot_training_bundle(source, target)


def test_benchmark_snapshot_rejects_evaluation_only_and_missing_shards(tmp_path):
    from tools.ttn_benchmark_snapshot import snapshot_training_bundle
    source = tmp_path / "training"; source.mkdir()
    (source / "run_config.json").write_text("{}")
    (source / "train.jsonl").write_text('{"step": 1, "stage": "C"}')
    torch.save({"step": 1}, source / "last.pt")
    with pytest.raises(ValueError, match="resume"):
        snapshot_training_bundle(source, tmp_path / "evaluation-only")
    torch.save({"step": 1, "distributed": {"resume_dir": "shards", "world_size": 2}}, source / "last.pt")
    with pytest.raises(FileNotFoundError): snapshot_training_bundle(source, tmp_path / "missing")
