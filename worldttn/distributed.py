"""Rank-local causal training under DDP/FSDP2; Slurm, Accelerate or torchrun.

Native wrapping keeps FP32 TTN parameters and direct generator access explicit.
S/psi, poses and caches are never registered as buffers or synchronized.
"""
from contextlib import nullcontext
from datetime import timedelta
import os
import ipaddress
import socket
import subprocess
import torch
from torch import distributed as dist
from .training import TTNTrainingWindow
from .parallel_checkpoint import save_training_checkpoint, restore_training_checkpoint
from .runtime import TTNChunkContext
from .anchor import TTNAnchor


def _expose_clean_state(module, args, kwargs, output):
    """FSDP must see the side-channel state that future chunk losses consume.

    Hook registered BEFORE fully_shard's post-forward hook. Its ordinary
    output alone cannot reveal gradients entering through ctx.candidates.
    """
    context = kwargs.get("ttn_chunk_context")
    if context is None:
        context = next((arg for arg in args if isinstance(arg, TTNChunkContext)), None)
    state = None
    if context is not None and context.clean_mode:
        candidate = context.candidates.get(module.attn.index)
        if candidate is not None: state = candidate[0]
    return output, state


def _restore_block_output(module, args, output):
    # Registered AFTER fully_shard's hook; the state's tensor hook remains
    # attached even when SANA receives its original output contract.
    return output[0]


def resolve_launch_environment(environ=None):
    """Normalize an srun task BEFORE CLI checks; batch shells are not workers.

    Slurm owns the GPU visibility mask. Never replace it with physical GPU IDs.
    Conflicting torchrun/Accelerate ranks indicate a nested/incorrect launch.
    """
    env = os.environ if environ is None else environ
    def number(name, default=None):
        try:
            return int(env[name]) if name in env else int(default)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a valid integer") from None
    slurm = "SLURM_PROCID" in env
    node_list = env.get("SLURM_STEP_NODELIST") or env.get("SLURM_JOB_NODELIST")
    job_id = env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID")
    master = env.get("MASTER_ADDR")
    port = env.get("MASTER_PORT")
    node_rank = None
    if slurm:
        rank, local, world, node_rank = [number(name) for name in (
            "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS", "SLURM_NODEID")]
        if number("LOCAL_WORLD_SIZE", 1) != 1:
            raise ValueError("nested launcher detected: Slurm requires one training process per node")
        if local != 0:
            raise ValueError("Slurm TTN requires one task per node, SLURM_LOCALID=0")
        count_name = next((name for name in ("SLURM_STEP_NUM_NODES", "SLURM_JOB_NUM_NODES", "SLURM_NNODES")
                           if name in env), None)
        hosts = None
        if not master or count_name is None:
            if not node_list: raise ValueError("Slurm requires a node list or explicit master and node count")
            try:
                hosts = subprocess.run(["scontrol", "show", "hostnames", node_list],
                                       check=True, capture_output=True, text=True, timeout=30).stdout.splitlines()
            except (OSError, subprocess.SubprocessError) as error:
                raise ValueError(f"cannot expand Slurm nodes with scontrol: {error}") from error
            if not hosts: raise ValueError("scontrol returned an empty Slurm node list")
        nodes = number(count_name) if count_name else len(hosts)
        if world != nodes or node_rank < 0:
            raise ValueError("Slurm TTN requires one task per node: task count must equal node count")
        if not master:
            try:
                master = socket.gethostbyname(hosts[0])
            except OSError as error:
                raise ValueError(f"cannot resolve IPv4 address for Slurm master {hosts[0]}: {error}") from error
        if port is None:
            try: port = str(15000 + int(job_id) % 40000)
            except (TypeError, ValueError):
                raise ValueError("MASTER_PORT or numeric SLURM_JOB_ID is required") from None
        values = {"RANK": rank, "LOCAL_RANK": local, "WORLD_SIZE": world, "NODE_RANK": node_rank}
        for name, value in values.items():
            if name in env and number(name) != value:
                raise ValueError(f"launcher conflict: {name}={env[name]} disagrees with Slurm {value}")
    else:
        rank, local, world = number("RANK", 0), number("LOCAL_RANK", 0), number("WORLD_SIZE", 1)
        if world > 1 and any(name not in env for name in ("RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")):
            raise ValueError("multi-process launch requires RANK, LOCAL_RANK, MASTER_ADDR and MASTER_PORT")
    if world < 1 or not 0 <= rank < world or local < 0:
        raise ValueError("invalid rank/local rank/world size")
    if master is not None:
        master = master.strip()
        if not master: raise ValueError("MASTER_ADDR must not be empty")
        try: loopback = ipaddress.ip_address(master).is_loopback
        except ValueError: loopback = master.lower() in ("localhost", "localhost.localdomain")
        if slurm and world > 1 and loopback:
            raise ValueError("multi-node MASTER_ADDR must not be a loopback address")
    if port is not None:
        try: port = int(port)
        except (TypeError, ValueError): raise ValueError("MASTER_PORT must be an integer") from None
        if not 1 <= port <= 65535: raise ValueError("MASTER_PORT must be between 1 and 65535")
    if slurm:
        env.update({name: str(value) for name, value in values.items()})
        env.update(MASTER_ADDR=master, MASTER_PORT=str(port))
    return {"launcher": "slurm" if slurm else "env", "rank": rank, "local_rank": local,
            "world_size": world, "node_rank": node_rank, "master_addr": master,
            "master_port": port, "job_id": job_id, "node_list": node_list}


def initialize(mode="auto", device="cuda", *, timeout_seconds=600):
    launch = resolve_launch_environment()
    rank, world = launch["rank"], launch["world_size"]
    if timeout_seconds <= 0: raise ValueError("distributed timeout must be positive")
    if mode == "auto":
        mode = ("ddp" if launch["launcher"] == "slurm" else "fsdp2") if world > 1 else "single"
    if mode not in ("single", "ddp", "fsdp2"): raise ValueError("invalid parallel mode")
    if mode == "single" and world != 1:
        raise ValueError("single mode requires WORLD_SIZE=1")
    if mode != "single" and world < 2:
        raise ValueError("DDP/FSDP2 require at least two processes; use srun, accelerate launch or torchrun")
    device = torch.device(device)
    if device.type == "cuda":
        if launch["launcher"] == "slurm":
            if torch.cuda.device_count() != 1: raise ValueError("Slurm TTN requires exactly one visible GPU per process")
            if device.index not in (None, 0): raise ValueError("Slurm TTN binds the local logical GPU cuda:0")
            index = 0
        else:
            index = int(os.environ.get("LOCAL_RANK", str(device.index or 0)))
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    if world > 1 and not dist.is_initialized():
        if rank == 0:
            print(f"[TTN init] host={socket.gethostname()} rank=0 local_rank={launch['local_rank']} "
                  f"world_size={world} device={device} master={launch['master_addr']}:{launch['master_port']} "
                  f"timeout={timeout_seconds}s", flush=True)
        options = {"device_id": device} if device.type == "cuda" else {}
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo", init_method="env://",
                                rank=rank, world_size=world, timeout=timedelta(seconds=timeout_seconds), **options)
    if dist.is_initialized() and (dist.get_rank(), dist.get_world_size()) != (rank, world):
        raise ValueError("initialized process group disagrees with launcher rank/world size")
    return mode, device


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def gather_records(record):
    """Compact CPU metadata only; never pass episode tensors to this function."""
    rank, world = rank_world()
    if world == 1: return [record]
    records = [None] * world
    dist.all_gather_object(records, record)
    return records


def check_distributed(device):
    """Exercise the real transport before any model/data loading; no SANA imports."""
    device = torch.device(device)
    rank, world = rank_world()
    record = {**resolve_launch_environment(), "rank": rank, "world_size": world,
              "hostname": socket.gethostname(), "device": str(device),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "backend": dist.get_backend() if dist.is_initialized() else None,
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "nccl": torch.cuda.nccl.version() if device.type == "cuda" else None}
    checksum = torch.tensor(rank + 1, dtype=torch.int64, device=device)
    if world > 1: dist.all_reduce(checksum)
    total = int(checksum.item())
    if total != world * (world + 1) // 2: raise RuntimeError("distributed all-reduce checksum mismatch")
    records = gather_records(record)
    if record["launcher"] == "slurm" and world > 1:
        if (len({row["hostname"] for row in records}) != world
                or any(row["local_rank"] != 0 for row in records)
                or [row["rank"] for row in records] != list(range(world))):
            raise ValueError("Slurm topology check requires unique nodes and one global rank per node")
    if world > 1: dist.barrier()
    return {"world_size": world, "backend": record["backend"], "all_reduce_sum": total, "ranks": records}


class ParallelTraining:
    def __init__(self, model, loss_fn, mode="single"):
        self.model = model
        self.loss_fn = loss_fn
        self.mode = mode
        self.rank, self.world = rank_world()
        self.window = TTNTrainingWindow(model, loss_fn)
        if mode == "ddp":
            if self.world < 2: raise ValueError("DDP requires an initialized multi-process group")
            from torch.nn.parallel import DistributedDataParallel
            device = next(model.parameters()).device
            self.window = DistributedDataParallel(
                self.window, device_ids=[device.index] if device.type == "cuda" else None,
                broadcast_buffers=False, gradient_as_bucket_view=True, find_unused_parameters=True)
        elif mode == "fsdp2":
            if self.world < 2: raise ValueError("FSDP2 requires an initialized multi-process group")
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
            device = next(model.parameters()).device
            mesh = init_device_mesh(device.type, (self.world,))
            # param_dtype=None preserves BF16 frozen weights and FP32 adapter/
            # beta/controller weights. FP32 reductions preserve clipping precision.
            policy = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32, cast_forward_inputs=False)
            for block in model.blocks:
                anchor = isinstance(block.attn, TTNAnchor)
                if anchor: block.register_forward_hook(_expose_clean_state, with_kwargs=True)
                fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
                if anchor: block.register_forward_hook(_restore_block_output)
            # Root owns the small controller/generators plus non-block parameters.
            # Keep these unsharded during the entire window: Stage C's analytic
            # update reads generators directly inside the anchors' clean forward.
            fully_shard(self.window, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
        elif mode != "single":
            raise ValueError("parallel mode must be single, ddp or fsdp2")

    def accumulation(self, final):
        if self.mode == "ddp" and not final: return self.window.no_sync()
        # FSDP2 reduces every window and accumulates sharded gradients. Using
        # no_sync here would retain full gradients and defeat the memory goal.
        return nullcontext()

    def reshard(self):
        if self.mode == "fsdp2": self.window.reshard()

    def validate_schedule(self, frames, tbptt, chunks, batch):
        if self.world == 1: return
        shape = torch.tensor([frames, tbptt, chunks, batch], device=next(self.model.parameters()).device)
        shapes = [torch.empty_like(shape) for _ in range(self.world)]
        dist.all_gather(shapes, shape)
        if any(not torch.equal(shape, other) for other in shapes):
            raise ValueError("all ranks must use the same frame count, TBPTT windows and per-rank batch size")

    def clip_grad_norm(self, maximum):
        params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if self.mode != "fsdp2":
            return torch.nn.utils.clip_grad_norm_(params, maximum, error_if_nonfinite=True)
        # Norm of the complete gradient, not one rank's shard. Native tensors
        # would be replicated and thus count 1/world; DTensors count their shard.
        device = next(self.model.parameters()).device
        squared = torch.zeros((), dtype=torch.float32, device=device)
        for p in params:
            grad = p.grad
            local = grad.to_local() if hasattr(grad, "to_local") else grad
            squared += local.float().square().sum() / (1 if hasattr(grad, "to_local") else self.world)
        dist.all_reduce(squared, op=dist.ReduceOp.SUM)
        norm = squared.sqrt()
        if not torch.isfinite(norm): raise FloatingPointError("nonfinite global gradient norm")
        scale = (maximum / (norm + 1e-6)).clamp_max(1.)
        for p in params:
            local = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
            local.mul_(scale)
        return norm
