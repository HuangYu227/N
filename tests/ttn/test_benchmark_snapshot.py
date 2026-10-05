import json
import os
import pytest
import torch


def test_benchmark_snapshot_pins_weights_optimizer_rng_and_cursor(tmp_path):
    from tools.ttn_benchmark_snapshot import snapshot_training_bundle
    source = tmp_path / "training"; source.mkdir()
    folder = source / "last-resume-00000025-abc"; folder.mkdir()
    payload = {"format": "TTN-SANA-WM-v0.1", "step": 25, "stage": "C", "base_sha256": "base", "train_scope": "ttn-visual",
               "distributed": {"format": "TTN-parallel-resume-v1", "mode": "fsdp2", "resume_dir": folder.name, "world_size": 2, "checkpoint_id": "abc"}}
    torch.save(payload, source / "last.pt")
    for rank in range(2):
        torch.save({**payload["distributed"], "rank": rank, "step": 25, "rng": torch.get_rng_state(), "data": {"cursor": 13},
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


def test_new_bundle_snapshot_copies_manifest_and_survives_source_retention(tmp_path):
    from types import SimpleNamespace
    from test_training import TinyWorldModel
    from worldttn.checkpoint import make_optimizer
    from worldttn.parallel_checkpoint import save_training_checkpoint
    from worldttn.checkpoint_integrity import audit_checkpoint
    from tools.ttn_benchmark_snapshot import snapshot_training_bundle
    source = tmp_path / "source"; source.mkdir()
    model = TinyWorldModel("C")
    engine = SimpleNamespace(model=model, rank=0, world=1, mode="single", reshard=lambda: None)
    for step in (1,): save_training_checkpoint(source / "last.pt", engine, make_optimizer(model), step)
    (source / "run_config.json").write_text(json.dumps({"base": {"sha256": None}}))
    (source / "train.jsonl").write_text('{"stage":"C","step":1}\n')
    target = snapshot_training_bundle(source, tmp_path / "snapshot")
    for step in (2, 3, 4): save_training_checkpoint(source / "last.pt", engine, make_optimizer(model), step)
    assert audit_checkpoint(target / "last.pt")["step"] == 1
    assert len(list(source.glob("last-resume-*"))) == 2


def test_benchmark_profile_uses_saved_data_and_scope_without_shell_injection(tmp_path):
    import shlex
    from tools.ttn_benchmark_snapshot import benchmark_environment
    (tmp_path / "run_config.json").write_text(json.dumps({"arguments": {
        "seed": 123, "dataset_root": "shared path/$(do-not-run)", "config": "reference.json", "unknown": "unsafe",
        "ttn_core_backend": "reference", "ttn_psi_backend": "reference"}}))
    torch.save({"train_scope": "dit"}, tmp_path / "last.pt")
    env = benchmark_environment(tmp_path)
    assert env["TRAIN_SCOPE"] == "dit" and env["SEED"] == "123" and "unknown" not in env
    assert "TTN_CORE_BACKEND" not in env and "TTN_PSI_BACKEND" not in env
    assert shlex.split(shlex.quote(env["DATASET_ROOT"])) == ["shared path/$(do-not-run)"]


def test_snapshot_in_progress_does_not_fail_live_checkpoint_save(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Timer
    from types import SimpleNamespace
    from test_training import TinyWorldModel
    from worldttn.checkpoint import make_optimizer
    from worldttn.parallel_checkpoint import save_training_checkpoint
    from worldttn.checkpoint_integrity import audit_checkpoint
    from tools import ttn_benchmark_snapshot as snapshots
    source = tmp_path / "source"; source.mkdir()
    model = TinyWorldModel("C")
    engine = SimpleNamespace(model=model, rank=0, world=1, mode="single", reshard=lambda: None)
    save_training_checkpoint(source / "last.pt", engine, make_optimizer(model), 1)
    (source / "run_config.json").write_text(json.dumps({"base": {"sha256": None}}))
    (source / "train.jsonl").write_text('{"stage":"C","step":1}\n')
    entered, release = Event(), Event()
    original = snapshots._snapshot_training_bundle
    def pause(*args):
        entered.set()
        assert release.wait(5), "test snapshot release timed out"
        return original(*args)
    monkeypatch.setattr(snapshots, "_snapshot_training_bundle", pause)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(snapshots.snapshot_training_bundle, source, tmp_path / "snapshot")
        assert entered.wait(5)
        timer = Timer(.2, release.set); timer.start()
        try: save_training_checkpoint(source / "last.pt", engine, make_optimizer(model), 2)
        finally: release.set(); timer.join()
        target = future.result(timeout=5)
    assert audit_checkpoint(source / "last.pt")["step"] == 2
    assert audit_checkpoint(target / "last.pt")["step"] == 1


def test_snapshot_unlock_failure_preserves_original_read_error(tmp_path, monkeypatch):
    from tools import ttn_benchmark_snapshot as snapshots
    from worldttn import checkpoint_integrity as integrity
    source = tmp_path / "source"; source.mkdir()
    monkeypatch.setattr(snapshots, "_snapshot_training_bundle",
                        lambda *a: (_ for _ in ()).throw(ValueError("original missing shard")))
    monkeypatch.setattr(integrity, "release_checkpoint_lock",
                        lambda *a: (_ for _ in ()).throw(OSError("release I/O failure")))
    with pytest.raises(ValueError, match="original missing shard"):
        snapshots.snapshot_training_bundle(source, tmp_path / "snapshot")
