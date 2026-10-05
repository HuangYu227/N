"""Evaluation snapshots must stay consistent while active training advances."""
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch


@pytest.fixture
def snapshot():
    path = Path(__file__).resolve().parents[2] / "tools/ttn_eval_snapshot.py"
    spec = importlib.util.spec_from_file_location("ttn_eval_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_run(tmp_path):
    source = tmp_path / "formal"
    source.mkdir()
    (source / "run_config.json").write_text(json.dumps({"base": {"sha256": "base"},
                                                       "arguments": {"train_scope": "dit"}}))
    torch.save({"step": 156, "stage": "C", "base_sha256": "base",
                "adapter": {"weight": torch.arange(12).float()}}, source / "last.pt")
    (source / "train.jsonl").write_text('{"step": 156, "stage": "C", "loss": 0.4}\n'
                                        '{"step": 157, "stage": "C", "loss": 0.3}\n{"step":')
    return source


def test_snapshot_uses_saved_step_not_latest_unsaved_update(snapshot, tmp_path):
    source = make_run(tmp_path)
    frozen = snapshot.snapshot_training_run(source)
    assert json.loads((frozen / "train.jsonl").read_text())["step"] == 156
    assert json.loads((frozen / "snapshot.json").read_text())["source"] == str(source.resolve())
    assert os.path.samefile(source / "last.pt", frozen / "last.pt")
    future = source / "next.pt"
    torch.save({"step": 166, "stage": "C", "adapter": {"weight": torch.zeros(12)}}, future)
    os.replace(future, source / "last.pt")
    pinned = torch.load(frozen / "last.pt", weights_only=False, mmap=True)
    assert pinned["step"] == 156 and torch.equal(pinned["adapter"]["weight"], torch.arange(12).float())
    assert torch.load(source / "last.pt", weights_only=False)["step"] == 166


def test_publication_between_link_and_read_cannot_change_snapshot(snapshot, tmp_path, monkeypatch):
    source = make_run(tmp_path)
    original_link = os.link
    def link_then_publish(src, dest):
        original_link(src, dest)
        future = source / "next.pt"
        torch.save({"step": 166, "stage": "C", "base_sha256": "base"}, future)
        os.replace(future, source / "last.pt")
        with (source / "train.jsonl").open("a") as stream:
            stream.write('\n{"step": 166, "stage": "C"}\n')
    monkeypatch.setattr(snapshot.os, "link", link_then_publish)
    frozen = snapshot.snapshot_training_run(source)
    assert json.loads((frozen / "train.jsonl").read_text())["step"] == 156


@pytest.mark.parametrize("change", ["missing-record", "wrong-stage", "wrong-base"])
def test_snapshot_rejects_inconsistent_metadata_without_modifying_source(snapshot, tmp_path, change):
    source = make_run(tmp_path)
    if change == "missing-record":
        (source / "train.jsonl").write_text('{"step": 157, "stage": "C"}\n')
    elif change == "wrong-stage":
        (source / "train.jsonl").write_text('{"step": 156, "stage": "A"}\n')
    else:
        (source / "run_config.json").write_text(json.dumps({"base": {"sha256": "different"}}))
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    with pytest.raises(ValueError): snapshot.snapshot_training_run(source)
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    assert not list(tmp_path.glob("eval-snapshot-*"))


@pytest.mark.parametrize("retain", [True, False])
def test_completed_evaluation_releases_only_opted_in_model_alias(snapshot, tmp_path, retain):
    source = make_run(tmp_path)
    frozen = snapshot.snapshot_training_run(source, retain_model=retain)
    result = {"status": "completed", "training_run": str(frozen), "identity": {"step": 156},
              "results": {"long": {}, "short": {}, "align": {}}}
    report = snapshot.release_evaluation_model(frozen, result)
    assert report["status"] == ("retained" if retain else "released")
    assert (frozen / "last.pt").exists() == retain
    assert (source / "last.pt").is_file()
    assert (frozen / "train.jsonl").is_file() and (frozen / "run_config.json").is_file()
    if not retain: assert snapshot.release_evaluation_model(frozen, result)["status"] == "released"


def test_manual_keep_protects_non_key_model(snapshot, tmp_path):
    source = make_run(tmp_path)
    frozen = snapshot.snapshot_training_run(source, retain_model=False)
    (frozen / ".keep").touch()
    result = {"status": "completed", "training_run": str(frozen), "identity": {"step": 156},
              "results": {"long": {}, "short": {}, "align": {}}}
    assert snapshot.release_evaluation_model(frozen, result)["status"] == "retained"
    assert (frozen / "last.pt").is_file()


def test_meta_snapshot_waits_for_contribution_children_before_release(snapshot, tmp_path):
    source = make_run(tmp_path)
    config = json.loads((source / "run_config.json").read_text())
    config["training"] = {"meta_ttt": {"local_update": True, "persistent_meta": True}}
    (source / "run_config.json").write_text(json.dumps(config))
    frozen = snapshot.snapshot_training_run(source, retain_model=False)
    result = {"status": "completed", "training_run": str(frozen), "identity": {"step": 156},
              "results": {"long": {}, "short": {}, "align": {}}}
    with pytest.raises(ValueError): snapshot.release_evaluation_model(frozen, result)
    assert (frozen / "last.pt").is_file()
    result["results"].update({"no-local": {}, "no-persistent": {}})
    assert snapshot.release_evaluation_model(frozen, result)["status"] == "released"


@pytest.mark.parametrize("damage", ["running", "failed", "step", "source", "missing-child", "replacement", "symlink"])
def test_model_release_protects_pending_failed_different_or_replaced_snapshots(snapshot, tmp_path, damage):
    source = make_run(tmp_path)
    frozen = snapshot.snapshot_training_run(source, retain_model=False)
    result = {"status": "completed", "training_run": str(frozen), "identity": {"step": 156},
              "results": {"long": {}, "short": {}, "align": {}}}
    if damage in ("running", "failed"): result["status"] = damage
    elif damage == "step": result["identity"]["step"] = 157
    elif damage == "source": result["training_run"] = str(source)
    elif damage == "missing-child": result["results"].pop("align")
    elif damage == "replacement":
        replacement = frozen / "replacement.pt"
        torch.save({"step": 156}, replacement)
        os.replace(replacement, frozen / "last.pt")
    else:
        (frozen / "last.pt").unlink()
        try: (frozen / "last.pt").symlink_to(source / "last.pt")
        except OSError: pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError): snapshot.release_evaluation_model(frozen, result)
    assert (frozen / "last.pt").exists() and (source / "last.pt").exists()
