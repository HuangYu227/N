"""Plain offline weight exports plus rank-local optimizer shards and RNG for exact resume."""
from pathlib import Path
import uuid
import torch
from torch import distributed as dist
from .anchor import offline_state_dict
from .checkpoint import atomic_save, checkpoint_payload, read_checkpoint, rng_state, restore_rng
from .training_health import optimizer_parameter_names


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


def save_training_checkpoint(path, parallel, optimizer, step, data_state=None, training_config=None):
    """All ranks participate; publish last.pt only after all shards are complete.

    Same world size/backend are required for optimizer resume. last.pt remains
    an ordinary single-card inference/initialization checkpoint (full DiT for joint training).
    """
    path = Path(path)
    identifier = [uuid.uuid4().hex if parallel.rank == 0 else None]
    if parallel.world > 1: dist.broadcast_object_list(identifier, src=0)
    checkpoint_id = identifier[0]
    folder = path.parent / f"{path.stem}-resume-{step:08d}-{checkpoint_id[:8]}"
    folder.mkdir(parents=True, exist_ok=True)
    parallel.reshard()
    adapter = {}
    for name, value in offline_state_dict(parallel.model).items():
        value = value.detach()
        if hasattr(value, "full_tensor"): value = value.full_tensor()  # collective on EVERY rank
        if parallel.rank == 0: adapter[name] = value.cpu()
    shard = {
        "format": "TTN-parallel-resume-v1", "rank": parallel.rank, "world_size": parallel.world,
        "mode": parallel.mode, "step": step, "rng": rng_state(), "data": data_state or {},
        "checkpoint_id": checkpoint_id,
        "optimizer": _pack(optimizer.state_dict()) if parallel.mode == "fsdp2" or parallel.rank == 0 else None,
        "optimizer_parameter_names": optimizer_parameter_names(parallel.model, optimizer)
    }
    atomic_save(shard, folder / f"rank-{parallel.rank:05d}.pt")
    if parallel.world > 1: dist.barrier()
    if parallel.rank == 0:
        payload = checkpoint_payload(parallel.model, adapter, None, step)
        payload["distributed"] = {"format": shard["format"], "mode": parallel.mode,
                                  "world_size": parallel.world, "resume_dir": folder.name,
                                  "checkpoint_id": checkpoint_id,
                                  "training_config": training_config or {}}
        atomic_save(payload, path)
    if parallel.world > 1: dist.barrier()


def validate_training_checkpoint(payload, mode, world, training_config):
    meta = payload.get("distributed", {})
    if meta.get("format") != "TTN-parallel-resume-v1":
        raise ValueError("resume requires a distributed training checkpoint")
    if meta.get("mode") != mode or meta.get("world_size") != world:
        raise ValueError("optimizer resume requires the same backend and world size")
    if meta.get("training_config") != (training_config or {}):
        raise ValueError("resume training configuration mismatch")
    if Path(meta["resume_dir"]).name != meta["resume_dir"]:
        raise ValueError("invalid resume directory")


def validate_unfreeze_checkpoint(payload, mode, world, training_config):
    meta = payload.get("distributed", {})
    old_config = meta.get("training_config", {})
    validate_training_checkpoint(payload, mode, world, old_config)
    if (payload.get("train_scope") != "ttn-visual" or payload.get("stage") != "C"
            or payload["config"].get("camera_attention") != "sana"
            or old_config.get("train_scope") != "ttn-visual" or training_config.get("train_scope") != "dit"):
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
    folder = Path(path).parent / payload["distributed"]["resume_dir"]
    shard = torch.load(folder / f"rank-{parallel.rank:05d}.pt", map_location="cpu", weights_only=False)
    _validate_shard(shard, payload, parallel, parallel.rank)
    return shard


def restore_training_progress(path, parallel, training_config):
    """Deliberate scope transition: keep RNG/cursor/step, never load old Adam state."""
    payload = read_checkpoint(path, parallel.model)
    validate_unfreeze_checkpoint(payload, parallel.mode, parallel.world, training_config)
    shard = _rank_shard(path, payload, parallel)
    restore_rng(shard["rng"])
    return int(payload["step"]), shard["data"]


def restore_training_checkpoint(path, parallel, optimizer, training_config=None):
    """Restore optimizer/RNG/cursor AFTER loading adapter into the unwrapped model.

    Call load_checkpoint before constructing ParallelTraining; loading ordinary
    tensors directly into FSDP2 parameters would bypass its sharding contract.
    """
    path = Path(path)
    payload = read_checkpoint(path, parallel.model, resume=True)
    validate_training_checkpoint(payload, parallel.mode, parallel.world, training_config)
    folder = path.parent / payload["distributed"]["resume_dir"]
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
