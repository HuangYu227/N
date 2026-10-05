"""Integrity and bounded retention for complete, immutable training bundles."""
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import shutil
import tempfile
import time
import uuid
import torch

FORMAT = "TTN-checkpoint-bundle-v1"
MARGIN = 2 * 1024**3


def atomic_json(value, path):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def acquire_checkpoint_lock(path, token, timeout=600):
    """Serialize publication, retention and snapshot pinning; never steal a lock."""
    path = Path(path)
    deadline = time.monotonic() + timeout
    waiting = False
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if not waiting:
                print(f"[TTN checkpoint] waiting for save/snapshot lock: {path}", flush=True)
                waiting = True
            if time.monotonic() >= deadline:
                raise TimeoutError(f"checkpoint lock wait exceeded {timeout}s: {path}; inspect the owning job before removing it")
            time.sleep(min(.2, max(0, deadline - time.monotonic())))
            continue
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream: stream.write(token)
        except BaseException:
            try: path.unlink()  # This invocation exclusively created the file.
            except OSError as error: print(f"[TTN checkpoint] lock cleanup failed: {error}", flush=True)
            raise
        return


def release_checkpoint_lock(path, token):
    path = Path(path)
    if path.exists() and path.read_text(encoding="utf-8") == token: path.unlink()


def file_receipt(path, **metadata):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError(f"not a regular checkpoint file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest(), **metadata}


def _name(value):
    if not isinstance(value, str) or value in ("", ".", "..") or Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError("invalid checkpoint path component")
    return value


def bundle_directory(path, metadata):
    path = Path(path)
    name = _name(metadata["resume_dir"])
    folder = path.parent if path.name == "model.pt" and path.parent.name == name else path.parent / name
    if folder.is_symlink() or folder.resolve().parent != folder.parent.resolve():
        raise ValueError("checkpoint bundle must stay within its parent directory")
    return folder


def _load(path):
    # mmap keeps a full DiT validation from allocating another CPU model copy.
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def audit_checkpoint(path, expected_step=None, *, verify_hashes=True):
    """Validate model + EVERY rank before exact resume or chain advancement.

    Legacy bundles are metadata-validated and explicitly marked unverified;
    they remain loadable but are never eligible for automatic retention.
    """
    path = Path(path)
    if path.is_symlink(): raise ValueError("checkpoint cannot be a symbolic link")
    try:
        payload = _load(path)
        if payload.get("format") != "TTN-SANA-WM-v0.1": raise ValueError("invalid model checkpoint format")
        step = payload["step"]
        if expected_step is not None and step != expected_step:
            raise ValueError(f"checkpoint step {step} does not match expected step {expected_step}")
        meta = payload["distributed"]
        if meta.get("format") != "TTN-parallel-resume-v1" or meta.get("world_size", 0) < 1:
            raise ValueError("invalid distributed checkpoint metadata")
        folder = bundle_directory(path, meta)
        expected = {"checkpoint_id": meta["checkpoint_id"], "step": step,
                    "mode": meta["mode"], "world_size": meta["world_size"]}
        manifest = None
        if meta.get("manifest") is not None:
            manifest_path = folder / _name(meta["manifest"])
            if manifest_path.is_symlink(): raise ValueError("manifest cannot be a symbolic link")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format") != FORMAT or any(manifest.get(k) != v for k, v in expected.items()):
                raise ValueError("checkpoint manifest metadata mismatch")
            if manifest.get("owner") != meta.get("bundle_owner"):
                raise ValueError("checkpoint manifest owner mismatch")
            receipts = manifest["files"]
            names = [r["file"] for r in receipts]
            if len(names) != meta["world_size"] + 1 or set(names) != {
                    "model.pt", *(f"rank-{rank:05d}.pt" for rank in range(meta["world_size"]))}:
                raise ValueError("checkpoint manifest file/rank set mismatch")
            for receipt in receipts:
                file = folder / _name(receipt["file"])
                if file.is_symlink() or not file.is_file() or file.stat().st_size != receipt["bytes"]:
                    raise ValueError(f"checkpoint file missing or wrong size: {file}")
                if verify_hashes and file_receipt(file)["sha256"] != receipt["sha256"]:
                    raise ValueError(f"checkpoint SHA256 mismatch: {file}")
            model_receipt = next(r for r in receipts if r["file"] == "model.pt")
            if not os.path.samefile(path, folder / "model.pt"):
                actual = file_receipt(path)
                if any(actual[key] != model_receipt[key] for key in ("bytes", "sha256")):
                    raise ValueError("published model differs from bundle model")
        for rank in range(meta["world_size"]):
            shard_path = folder / f"rank-{rank:05d}.pt"
            if shard_path.is_symlink(): raise ValueError("optimizer shard cannot be a symbolic link")
            shard = _load(shard_path)
            if shard.get("format") != meta["format"] or any(shard.get(k) != v for k, v in {**expected, "rank": rank}.items()):
                raise ValueError(f"resume shard metadata mismatch: rank {rank}")
            if "rng" not in shard or "data" not in shard or (meta["mode"] == "fsdp2" or rank == 0) and shard.get("optimizer") is None:
                raise ValueError(f"incomplete resume shard: rank {rank}")
            if manifest:
                receipt = next(r for r in receipts if r["file"] == shard_path.name)
                if receipt.get("rank") != rank: raise ValueError("manifest rank metadata mismatch")
            del shard
        return {**expected, "path": str(path), "bundle": str(folder), "owner": meta.get("bundle_owner"),
                "stage": payload.get("stage"), "train_scope": payload.get("train_scope"),
                "status": "verified" if manifest and verify_hashes else "metadata_verified" if manifest else "legacy_unverified"}
    except (OSError, KeyError, TypeError, AttributeError, RuntimeError, EOFError, pickle.UnpicklingError, json.JSONDecodeError) as error:
        raise ValueError(f"checkpoint validation failed for {path}: {error}") from error


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        value = value.to_local() if hasattr(value, "to_local") else value
        return value.numel() * value.element_size()
    if isinstance(value, dict): return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)): return sum(tensor_bytes(v) for v in value)
    return 0


def storage_budget(path, model_bytes, shard_bytes):
    """Budget includes copy fallback for publication; old bundles stay live."""
    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    estimate = 2 * model_bytes + sum(shard_bytes)
    required = math.ceil(estimate * 1.15) + MARGIN
    free = shutil.disk_usage(parent).free
    report = {"estimated_new_bytes": estimate, "required_free_bytes": required, "available_bytes": free}
    if free < required:
        raise OSError(f"checkpoint disk budget insufficient: need {required / 1024**3:.2f} GiB, free {free / 1024**3:.2f} GiB")
    return report


def bundle_owner(path):
    marker = Path(path).parent / ".checkpoint-owner.json"
    if marker.is_symlink(): raise ValueError("checkpoint owner marker cannot be a symbolic link")
    if not marker.exists(): atomic_json({"format": FORMAT, "owner": uuid.uuid4().hex}, marker)
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value.get("format") != FORMAT or not value.get("owner"): raise ValueError("invalid checkpoint owner marker")
    return value["owner"]


def publish_model(source, destination):
    """Publish an alias of an immutable model; failure leaves old alias intact."""
    source, destination = Path(source), Path(destination)
    temporary = destination.parent / (destination.name + ".publish-" + uuid.uuid4().hex)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
            with temporary.open("r+b") as stream: os.fsync(stream.fileno())
            if file_receipt(temporary)["sha256"] != file_receipt(source)["sha256"]:
                raise ValueError("checkpoint publication copy hash mismatch")
        os.replace(temporary, destination)
    finally:
        if temporary.exists(): temporary.unlink()


def prune_bundles(path, *, keep=2, apply=False):
    """Only our complete bundles; unknown/legacy/partial artifacts stay put.

    .keep pins a resume bundle. Root-level .pt aliases also pin their referenced
    bundles. Evaluation snapshots outside this root retain model hardlinks only.
    """
    if keep < 2: raise ValueError("retain at least two complete resume bundles")
    path = Path(path).absolute()
    current = audit_checkpoint(path)
    root = path.parent.resolve()
    marker = root / ".checkpoint-owner.json"
    if not marker.exists() or marker.is_symlink(): return {"eligible": [], "deleted": [], "skipped": ["unmanaged run"]}
    owner = json.loads(marker.read_text(encoding="utf-8")).get("owner")
    if current["owner"] != owner: return {"eligible": [], "deleted": [], "skipped": ["legacy or foreign current checkpoint"]}
    protected = {Path(current["bundle"]).resolve()}
    for alias in root.glob("*.pt"):
        if alias == path or alias.is_symlink(): continue
        try:
            meta = _load(alias).get("distributed", {})
            if meta.get("resume_dir"): protected.add(bundle_directory(alias, meta).resolve())
        except (ValueError, OSError, AttributeError, RuntimeError, EOFError, pickle.UnpicklingError):
            # Unknown root checkpoints may contain references we cannot resolve.
            return {"eligible": [], "deleted": [], "skipped": [f"unreadable checkpoint alias: {alias}"]}
    bundles, skipped = [], []
    for folder in root.glob(path.stem + "-resume-*"):
        if folder.is_symlink() or folder.resolve().parent != root or not (folder / "manifest.json").is_file(): continue
        try:
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("owner") != owner or manifest.get("target") != path.name: continue
            # A readable, same-size shard may still have corrupted tensor storage.
            # Only SHA256-verified bundles can replace an older recovery point.
            audit_checkpoint(folder / "model.pt")
            allowed = {"manifest.json", *(r["file"] for r in manifest["files"])}
            if (folder / ".keep").exists(): protected.add(folder.resolve())
            elif {p.name for p in folder.iterdir()} != allowed:
                skipped.append(f"unknown files: {folder}")
                continue
            bundles.append((manifest["created_ns"], folder, manifest))
        except (ValueError, OSError, KeyError, TypeError):
            skipped.append(f"invalid/partial bundle: {folder}")
    bundles.sort(key=lambda item: item[0], reverse=True)
    protected.update(folder.resolve() for _, folder, _ in bundles[:keep])
    eligible = [folder for _, folder, _ in bundles if folder.resolve() not in protected]
    deleted = []
    for _, folder, manifest in bundles:
        if folder not in eligible or not apply: continue
        # Validate again immediately before removing regular, explicitly named files.
        audit_checkpoint(folder / "model.pt", verify_hashes=False)
        for name in [*(r["file"] for r in manifest["files"]), "manifest.json"]:
            file = folder / _name(name)
            if file.is_symlink() or file.resolve().parent != folder.resolve():
                raise ValueError("refusing to prune a file outside its checkpoint bundle")
        for name in [*(r["file"] for r in manifest["files"]), "manifest.json"]: (folder / name).unlink()
        folder.rmdir()
        deleted.append(str(folder))
    return {"eligible": [str(p) for p in eligible], "deleted": deleted, "skipped": skipped}
