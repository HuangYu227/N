"""Plain offline weight exports plus rank-local optimizer shards and RNG for exact resume."""
from pathlib import Path
from datetime import timedelta
import json
import time
import uuid
import torch
from torch import distributed as dist
from .anchor import offline_state_dict
from .checkpoint import atomic_save, checkpoint_payload, read_checkpoint, rng_state, restore_rng
from .training_health import optimizer_parameter_names
from .checkpoint_integrity import (FORMAT, acquire_checkpoint_lock, release_checkpoint_lock, atomic_json,
                                   audit_checkpoint, bundle_directory, bundle_owner, file_receipt,
                                   publish_model, prune_bundles, storage_budget, tensor_bytes)
from .failure import record_failure


def _pack(value):
    if isinstance(value, torch.Tensor):
        if hasattr(value, "to_local"):
            return {"ttn_dtensor": True, "local": value.to_local().detach().cpu(),
                    "shape": tuple(value.shape), "stride": tuple(value.stride()),
                    "placements": value.placements}
        return value.detach().cpu()
    if isinstance(value, dict): return {k: _pack(v) for k, v in value.items()}
    if isinstance(value, list): return [_pack(v) for v in value]
    if isinstance(value, tuple): return tuple(_pack(v) for v in value)
    return value


def _unpack(value, parameter):
    if isinstance(value, dict) and value.get("ttn_dtensor") is True:
        from torch.distributed.tensor import DTensor
        if not hasattr(parameter, "device_mesh"):
            raise ValueError("optimizer shard requires a DTensor parameter")
        return DTensor.from_local(value["local"].to(parameter.device), device_mesh=parameter.device_mesh,
                                 placements=value["placements"], shape=torch.Size(value["shape"]),
                                 stride=value["stride"], run_check=False)
    if isinstance(value, dict): return {k: _unpack(v, parameter) for k, v in value.items()}
    return value


def _phase(parallel, path, label, action, step=None):
    """Propagate local errors BEFORE any subsequent tensor collective."""
    # Disk verification can outlast NCCL's timeout; coordinate it on the CPU.
    group = None
    if parallel.world > 1:
        if getattr(parallel, "checkpoint_cpu_group", None) is None:
            parallel.checkpoint_cpu_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=3600))
        group = parallel.checkpoint_cpu_group
    timed = parallel.rank == 0 and label in {"shard-write", "model-publish", "resume-integrity"}
    started = time.monotonic()
    if timed: print("[TTN checkpoint phase] " + json.dumps({"phase": label, "status": "begin", "step": step}), flush=True)
    try:
        result = {"rank": parallel.rank, "value": action(), "error": None}
    except Exception as error:
        record_failure(Path(path).parent, label, error, rank=parallel.rank, step=step)
        result = {"rank": parallel.rank, "error": f"{type(error).__name__}: {error}"}
    results = [None] * parallel.world
    if parallel.world > 1: dist.all_gather_object(results, result, group=group)
    else: results[0] = result
    errors = [f"rank {r['rank']}: {r['error']}" for r in results if r["error"]]
    if timed:
        print("[TTN checkpoint phase] " + json.dumps({"phase": label, "status": "failed" if errors else "completed",
              "step": step, "seconds": time.monotonic() - started, "control_backend": "gloo" if group is not None else "local"}), flush=True)
    if errors: raise RuntimeError(f"checkpoint {label} failed; " + "; ".join(errors))
    return [r["value"] for r in results]


def _gather_model(parallel, path, step):
    values, adapter = {}, {}
    def prepare():
        parallel.reshard()
        values.update({name: value.detach() for name, value in offline_state_dict(parallel.model).items()})
        return [(name, tuple(value.shape), str(value.dtype), hasattr(value, "full_tensor"))
                for name, value in values.items()]
    schemas = _phase(parallel, path, "model-gather-prepare", prepare, step)
    if any(schema != schemas[0] for schema in schemas):
        raise ValueError("checkpoint model gather schema/order differs between ranks")
    for name, value in values.items():
        try:
            if hasattr(value, "full_tensor"): value = value.full_tensor()  # EVERY rank
        except Exception as error:
            record_failure(Path(path).parent, f"model-gather:{name}", error, rank=parallel.rank, step=step)
            # A failed tensor collective may invalidate the process group; exit,
            # rather than attempting another collective to propagate this error.
            raise
        def copy():
            if parallel.rank == 0: adapter[name] = value.cpu()
        _phase(parallel, path, f"model-copy:{name}", copy, step)
    return adapter


def preflight_checkpoint(path, parallel, optimizer):
    def estimates():
        values = offline_state_dict(parallel.model).values()
        model = sum(v.numel() * v.element_size() for v in values)
        # Before the first update Adam moments do not exist yet.
        moments = 2 * sum(tensor_bytes(p) for group in optimizer.param_groups for p in group["params"])
        shard = max(moments, tensor_bytes(optimizer.state_dict())) if parallel.mode == "fsdp2" or parallel.rank == 0 else 0
        return {"model": model, "shard": shard}
    estimates = _phase(parallel, path, "storage-estimate", estimates)
    budgets = _phase(parallel, path, "disk-budget", lambda: storage_budget(
        path, estimates[0]["model"], [e["shard"] for e in estimates]) if parallel.rank == 0 else None)
    return budgets[0]


def save_training_checkpoint(path, parallel, optimizer, step, data_state=None, training_config=None):
    """All ranks participate; publish last.pt only after all shards are complete.

    Same world size/backend are required for optimizer resume. last.pt remains
    an ordinary single-card inference/initialization checkpoint (full DiT for joint training).
    """
    path = Path(path)
    budget = preflight_checkpoint(path, parallel, optimizer)
    identifier = [uuid.uuid4().hex if parallel.rank == 0 else None]
    if parallel.world > 1: dist.broadcast_object_list(identifier, src=0)
    checkpoint_id = identifier[0]
    folder = path.parent / f"{path.stem}-resume-{step:08d}-{checkpoint_id[:8]}"
    lock = path.parent / ".checkpoint-save.lock"
    locked = False
    def acquire():
        nonlocal locked
        if parallel.rank != 0: return None
        acquire_checkpoint_lock(lock, checkpoint_id)
        locked = True
        return bundle_owner(path)
    try:
        owner = _phase(parallel, path, "save-lock", acquire, step)[0]
        # A snapshot may have held the lock long enough for free space to change.
        budget = preflight_checkpoint(path, parallel, optimizer)
        _phase(parallel, path, "bundle-create", lambda: folder.mkdir(exist_ok=False) if parallel.rank == 0 else None, step)
        adapter = _gather_model(parallel, path, step)
        def write_shard():
            shard = {
                "format": "TTN-parallel-resume-v1", "rank": parallel.rank, "world_size": parallel.world,
                "mode": parallel.mode, "step": step, "rng": rng_state(), "data": data_state or {},
                "checkpoint_id": checkpoint_id,
                "optimizer": _pack(optimizer.state_dict()) if parallel.mode == "fsdp2" or parallel.rank == 0 else None,
                "optimizer_parameter_names": optimizer_parameter_names(parallel.model, optimizer)
            }
            target = folder / f"rank-{parallel.rank:05d}.pt"
            atomic_save(shard, target)
            return file_receipt(target, rank=parallel.rank)
        receipts = _phase(parallel, path, "shard-write", write_shard, step)
        def publish():
            if parallel.rank != 0: return None
            payload = checkpoint_payload(parallel.model, adapter, None, step)
            payload["distributed"] = {"format": "TTN-parallel-resume-v1", "mode": parallel.mode,
                                      "world_size": parallel.world, "resume_dir": folder.name,
                                      "checkpoint_id": checkpoint_id, "manifest": "manifest.json", "bundle_owner": owner,
                                      "training_config": training_config or {}}
            model_path = folder / "model.pt"
            atomic_save(payload, model_path)
            manifest = {"format": FORMAT, "owner": owner, "target": path.name, "created_ns": time.time_ns(),
                        "step": step, "checkpoint_id": checkpoint_id, "mode": parallel.mode, "world_size": parallel.world,
                        "stage": payload["stage"], "train_scope": payload["train_scope"], "weight_scope": payload["weight_scope"],
                        "files": [file_receipt(model_path, rank=None), *receipts]}
            atomic_json(manifest, folder / "manifest.json")
            report = audit_checkpoint(model_path, step)
            publish_model(model_path, path)
            print("[TTN checkpoint] " + json.dumps({**report, "path": str(path), "budget": budget}), flush=True)
            # Publication is already successful: a cleanup failure must not invalidate it.
            try:
                started = time.monotonic()
                print("[TTN retention begin] " + json.dumps({"step": step}), flush=True)
                print("[TTN retention] " + json.dumps({**prune_bundles(path, apply=True),
                      "seconds": time.monotonic() - started}), flush=True)
            except Exception as error:
                print(f"[TTN retention] skipped after successful save: {error}", flush=True)
            return report
        return _phase(parallel, path, "model-publish", publish, step)[0]
    finally:
        if locked:
            try: release_checkpoint_lock(lock, checkpoint_id)
            except OSError as error: print(f"[TTN checkpoint] could not release save lock: {error}", flush=True)


def _resume_identity(config):
    config = dict(config or {})
    config.setdefault("execution", {"core_backend": "reference", "psi_backend": "reference"})
    return config


def validate_training_checkpoint(payload, mode, world, training_config, *, benchmark=False):
    meta = payload.get("distributed", {})
    if meta.get("format") != "TTN-parallel-resume-v1":
        raise ValueError("resume requires a distributed training checkpoint")
    if meta.get("mode") != mode or meta.get("world_size") != world:
        raise ValueError("optimizer resume requires the same backend and world size")
    saved, requested = _resume_identity(meta.get("training_config")), _resume_identity(training_config)
    report = {"saved": saved["execution"], "requested": requested["execution"],
              "benchmark_override": bool(benchmark and saved["execution"] != requested["execution"])}
    if benchmark:
        saved.pop("execution"); requested.pop("execution")
    if saved != requested:
        raise ValueError("resume training configuration mismatch")
    if Path(meta["resume_dir"]).name != meta["resume_dir"]:
        raise ValueError("invalid resume directory")
    return report


def validate_unfreeze_checkpoint(payload, mode, world, training_config):
    meta = payload.get("distributed", {})
    old_config = _resume_identity(meta.get("training_config"))
    training_config = _resume_identity(training_config)
    validate_training_checkpoint(payload, mode, world, old_config)
    if (payload.get("train_scope") not in ("ttn-visual", "ttn-new") or payload.get("stage") != "C"
            or payload["config"].get("camera_attention") != "sana"
            or old_config.get("train_scope") != payload.get("train_scope") or training_config.get("train_scope") != "dit"):
        raise ValueError("unfreeze requires Stage C/sana-camera visual warmup -> joint DiT")
    changed = {"train_scope", "backbone_lr", "optimizer_foreach"}
    if ({k: v for k, v in old_config.items() if k not in changed}
            != {k: v for k, v in training_config.items() if k not in changed}):
        raise ValueError("unfreeze training configuration mismatch")


def _validate_shard(shard, payload, parallel, rank):
    if (shard.get("rank"), shard.get("world_size"), shard.get("mode"), shard.get("step"),
            shard.get("checkpoint_id")) != (
            rank, parallel.world, parallel.mode, payload["step"], payload["distributed"]["checkpoint_id"]):
        raise ValueError("resume shard metadata mismatch")


def _rank_shard(path, payload, parallel):
    folder = bundle_directory(path, payload["distributed"])
    shard = torch.load(folder / f"rank-{parallel.rank:05d}.pt", map_location="cpu", weights_only=False)
    _validate_shard(shard, payload, parallel, parallel.rank)
    return shard


def restore_training_progress(path, parallel, training_config):
    """Deliberate scope transition: keep RNG/cursor/step, never load old Adam state."""
    _phase(parallel, path, "resume-integrity", lambda: audit_checkpoint(path) if parallel.rank == 0 else None)
    payload = read_checkpoint(path, parallel.model)
    validate_unfreeze_checkpoint(payload, parallel.mode, parallel.world, training_config)
    shard = _rank_shard(path, payload, parallel)
    restore_rng(shard["rng"])
    return int(payload["step"]), shard["data"]


def restore_training_checkpoint(path, parallel, optimizer, training_config=None, *, benchmark=False):
    """Restore optimizer/RNG/cursor AFTER loading adapter into the unwrapped model.

    Call load_checkpoint before constructing ParallelTraining; loading ordinary
    tensors directly into FSDP2 parameters would bypass its sharding contract.
    """
    path = Path(path)
    reports = _phase(parallel, path, "resume-integrity", lambda: audit_checkpoint(path) if parallel.rank == 0 else None)
    if parallel.rank == 0: print("[TTN resume integrity] " + json.dumps(reports[0]), flush=True)
    payload = read_checkpoint(path, parallel.model, resume=True)
    execution = validate_training_checkpoint(payload, parallel.mode, parallel.world, training_config, benchmark=benchmark)
    if parallel.rank == 0: print("[TTN resume execution] " + json.dumps(execution), flush=True)
    folder = bundle_directory(path, payload["distributed"])
    shard = _rank_shard(path, payload, parallel)
    state = shard["optimizer"]
    if state is None:
        owner = torch.load(folder / "rank-00000.pt", map_location="cpu", weights_only=False)
        _validate_shard(owner, payload, parallel, 0)
        state = owner["optimizer"]
        if owner.get("optimizer_parameter_names") is not None and (
                owner["optimizer_parameter_names"] != optimizer_parameter_names(parallel.model, optimizer)):
            raise ValueError("resume owner optimizer parameter names/order mismatch")
    if shard.get("optimizer_parameter_names") is not None and (
            shard["optimizer_parameter_names"] != optimizer_parameter_names(parallel.model, optimizer)):
        raise ValueError("resume optimizer parameter names/order mismatch")
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    ids = [i for group in state["param_groups"] for i in group["params"]]
    if len(parameters) != len(ids): raise ValueError("resume optimizer parameter count mismatch")
    lookup = dict(zip(ids, parameters))
    state["state"] = {i: _unpack(values, lookup[i]) for i, values in state["state"].items()}
    optimizer.load_state_dict(state)
    restore_rng(shard["rng"])
    return int(payload["step"]), shard["data"]
