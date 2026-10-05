import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from test_training import TinyWorldModel
from worldttn.checkpoint import make_optimizer, load_checkpoint
from worldttn import parallel_checkpoint as saving
from worldttn import checkpoint_integrity as integrity


@pytest.fixture
def engine():
    model = TinyWorldModel("C")
    return SimpleNamespace(model=model, rank=0, world=1, mode="single", reshard=lambda: None)


def save(path, engine, step):
    return saving.save_training_checkpoint(path, engine, make_optimizer(engine.model), step, {"cursor": step}, {"tbptt": 2})


def test_complete_bundles_retain_two_recoverable_models_and_snapshots(tmp_path, engine):
    path = tmp_path / "last.pt"
    save(path, engine, 1)
    first = integrity.audit_checkpoint(path)
    snapshot = tmp_path / "eval-snapshot.pt"
    os.link(path, snapshot)
    (Path(first["bundle"]) / ".keep").touch()
    for step in (2, 3, 4): save(path, engine, step)
    bundles = sorted(tmp_path.glob("last-resume-*"))
    assert len(bundles) == 3  # pinned 1 and latest 3/4; all have model + optimizer
    assert integrity.audit_checkpoint(snapshot)["step"] == 1
    for bundle in bundles: assert integrity.audit_checkpoint(bundle / "model.pt")["status"] == "verified"
    previous = next(p for p in bundles if "00000003" in p.name)
    resumed = TinyWorldModel("C")
    load_checkpoint(previous / "model.pt", resumed)
    other = SimpleNamespace(model=resumed, rank=0, world=1, mode="single")
    assert saving.restore_training_checkpoint(previous / "model.pt", other, make_optimizer(resumed), {"tbptt": 2}) == (3, {"cursor": 3})


@pytest.mark.parametrize("damage", ["missing", "hash", "rank", "step", "id", "manifest", "model"])
def test_rejects_corrupt_or_incomplete_bundles(tmp_path, engine, damage):
    path = tmp_path / "last.pt"; save(path, engine, 1)
    folder = Path(integrity.audit_checkpoint(path)["bundle"])
    shard = folder / "rank-00000.pt"
    if damage == "missing": shard.unlink()
    elif damage == "hash":
        data = bytearray(shard.read_bytes()); data[100] ^= 1; shard.write_bytes(data)
    elif damage == "manifest":
        manifest = json.loads((folder / "manifest.json").read_text()); manifest["world_size"] = 2
        (folder / "manifest.json").write_text(json.dumps(manifest))
    elif damage == "model":
        torch.save({"format": "TTN-SANA-WM-v0.1", "step": 99}, path)
    else:
        value = torch.load(shard, weights_only=False)
        value[{"rank": "rank", "step": "step", "id": "checkpoint_id"}[damage]] = "wrong"
        torch.save(value, shard)
        manifest = json.loads((folder / "manifest.json").read_text())
        manifest["files"][1] = integrity.file_receipt(shard, rank=0)
        (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError): integrity.audit_checkpoint(path, 1)


@pytest.mark.parametrize("phase", ["shard", "model", "publish", "disk"])
def test_failed_save_preserves_last_verified_checkpoint(tmp_path, engine, monkeypatch, phase):
    path = tmp_path / "last.pt"; save(path, engine, 1)
    before = path.read_bytes()
    if phase == "disk":
        monkeypatch.setattr(integrity.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    elif phase == "publish":
        monkeypatch.setattr(saving, "publish_model", lambda *a: (_ for _ in ()).throw(OSError("quota")))
    else:
        original = saving.atomic_save
        def fail(payload, target):
            if (phase == "shard" and target.name.startswith("rank-")) or (phase == "model" and target.name == "model.pt"):
                raise OSError("quota")
            original(payload, target)
        monkeypatch.setattr(saving, "atomic_save", fail)
    with pytest.raises(RuntimeError, match="failed"): save(path, engine, 2)
    assert path.read_bytes() == before
    assert integrity.audit_checkpoint(path)["step"] == 1
    assert not (tmp_path / ".checkpoint-save.lock").exists()
    assert list(tmp_path.glob("failure-*.json"))


def test_retention_ignores_legacy_partial_foreign_and_unknown_files(tmp_path, engine):
    path = tmp_path / "last.pt"
    legacy = tmp_path / "last-resume-legacy"; legacy.mkdir(); (legacy / "rank-00000.pt").write_bytes(b"old")
    partial = tmp_path / "last-resume-partial"; partial.mkdir(); (partial / "model.pt.tmp").touch()
    save(path, engine, 1)
    first = Path(integrity.audit_checkpoint(path)["bundle"])
    (first / "user-note.txt").write_text("keep")
    for step in (2, 3, 4): save(path, engine, step)
    assert legacy.is_dir() and partial.is_dir() and first.is_dir()
    assert integrity.prune_bundles(path)["deleted"] == []
    with pytest.raises(ValueError): integrity.prune_bundles(path, keep=1)
    payload = torch.load(path, weights_only=False); payload["distributed"]["resume_dir"] = "../escape"
    torch.save(payload, tmp_path / "bad.pt")
    with pytest.raises(ValueError): integrity.audit_checkpoint(tmp_path / "bad.pt")


def test_publication_copy_fallback_and_failure_preserve_alias(tmp_path, monkeypatch):
    source = tmp_path / "model.pt"; source.write_bytes(b"new")
    dest = tmp_path / "last.pt"; dest.write_bytes(b"old")
    monkeypatch.setattr(integrity.os, "link", lambda *a: (_ for _ in ()).throw(OSError("no hardlinks")))
    integrity.publish_model(source, dest)
    assert source.read_bytes() == dest.read_bytes() == b"new"
    monkeypatch.setattr(integrity.shutil, "copyfile", lambda *a: (_ for _ in ()).throw(OSError("quota")))
    with pytest.raises(OSError): integrity.publish_model(source, dest)
    assert dest.read_bytes() == b"new" and not list(tmp_path.glob("*.publish-*"))


def test_legacy_checkpoint_validates_metadata_but_is_not_auto_pruned(tmp_path, engine):
    path = tmp_path / "last.pt"; save(path, engine, 1)
    value = torch.load(path, weights_only=False)
    value["distributed"].pop("manifest"); value["distributed"].pop("bundle_owner")
    torch.save(value, path)
    assert integrity.audit_checkpoint(path)["status"] == "legacy_unverified"
    assert not integrity.prune_bundles(path, apply=True)["deleted"]


def test_budget_accounts_for_two_model_copies_and_all_rank_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(integrity.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    r = integrity.storage_budget(tmp_path / "last.pt", 1000, [2000, 2000])
    assert r["estimated_new_bytes"] == 6000
    assert r["required_free_bytes"] == 6900 + integrity.MARGIN


def test_retention_does_not_count_same_size_corrupted_bundle_as_recoverable(tmp_path, engine):
    path = tmp_path / "last.pt"
    save(path, engine, 1)
    first = Path(integrity.audit_checkpoint(path)["bundle"])
    save(path, engine, 2)
    second = Path(integrity.audit_checkpoint(path)["bundle"])
    shard = torch.load(second / "rank-00000.pt", weights_only=False, mmap=True)
    # Alter tensor storage only: archive size and readable metadata stay identical.
    content = (second / "rank-00000.pt").read_bytes()
    data = shard["rng"]["torch"].numpy().tobytes()
    position = content.index(data)
    with (second / "rank-00000.pt").open("r+b") as stream:
        stream.seek(position); stream.write(bytes([content[position] ^ 1]))
    integrity.audit_checkpoint(second / "model.pt", verify_hashes=False)
    with pytest.raises(ValueError, match="SHA256"): integrity.audit_checkpoint(second / "model.pt")
    save(path, engine, 3)
    assert integrity.audit_checkpoint(first / "model.pt")["step"] == 1
    assert integrity.audit_checkpoint(path)["step"] == 3


def test_lock_timeout_never_steals_an_existing_writer_lock(tmp_path):
    lock = tmp_path / ".checkpoint-save.lock"
    integrity.acquire_checkpoint_lock(lock, "existing-writer")
    with pytest.raises(TimeoutError, match="inspect the owning job"):
        integrity.acquire_checkpoint_lock(lock, "other-writer", timeout=0)
    integrity.release_checkpoint_lock(lock, "other-writer")
    assert lock.read_text() == "existing-writer"
    integrity.release_checkpoint_lock(lock, "existing-writer")
    assert not lock.exists()


def test_lock_write_and_cleanup_errors_keep_the_original_write_error(tmp_path, monkeypatch):
    from contextlib import contextmanager
    original = integrity.os.fdopen
    @contextmanager
    def broken_handle(*args, **kwargs):
        with original(*args, **kwargs):
            yield SimpleNamespace(write=lambda *a: (_ for _ in ()).throw(OSError("original write failure")))
    monkeypatch.setattr(integrity.os, "fdopen", broken_handle)
    monkeypatch.setattr(Path, "unlink", lambda *a, **k: (_ for _ in ()).throw(OSError("cleanup failure")))
    with pytest.raises(OSError, match="original write failure"):
        integrity.acquire_checkpoint_lock(tmp_path / ".checkpoint-save.lock", "writer")
