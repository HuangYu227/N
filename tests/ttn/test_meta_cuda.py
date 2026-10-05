"""Small real-TTN FSDP2 checks run by one pytest process per Slurm GPU."""
import os
from pathlib import Path
import pytest
import torch
from test_training import TinyWorldModel
from test_parallel import _rank_inputs, _resume_worker
from test_fsdp_offload import _offload_worker
from worldttn.anchor import configure_train_scope
from worldttn.checkpoint import make_optimizer
from worldttn.training import train_clip, linear_flow_loss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or "META_TEST_OUTPUT" not in os.environ,
                               reason="requires isolated Slurm CUDA acceptance allocation")


def allocation(name):
    from worldttn.distributed import resolve_launch_environment
    launch = resolve_launch_environment()
    assert launch["world_size"] == 4, "use four GPUs, one visible GPU per node"
    assert torch.cuda.device_count() == 1
    output = Path(os.environ["META_TEST_OUTPUT"]) / name
    output.mkdir(parents=True, exist_ok=True)
    return launch["rank"], output


def test_meta_four_gpu_future_credit_and_offload():
    rank, output = allocation("future-credit")
    _offload_worker(rank, 4, "env://", str(output), 2, False, True, "cuda")
    # After teardown every rank independently checks the unchanged FP32
    # tolerances against the same four-example unsharded global batch.
    torch.manual_seed(17)
    reference = configure_train_scope(TinyWorldModel(local_update=True, persistent_meta=True).cuda(), "dit")
    clean, noise, t, camera = [torch.cat(values).cuda() for values in zip(*[_rank_inputs(r) for r in range(4)])]
    result = train_clip(reference, clean, torch.zeros(4, 1, 2, 8, device="cuda"), camera, make_optimizer(reference),
        linear_flow_loss, t, noise, width=100, height=100, tbptt=2, activation_offload="cpu")
    actual = torch.load(output / f"rank-{rank}.pt", map_location="cuda", weights_only=False)
    assert actual["norm"] == pytest.approx(result["outer_grad_norm"], rel=2e-5)
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(actual["state"][name].cuda(), parameter, rtol=2e-5, atol=3e-7)
        if parameter.grad is not None:
            torch.testing.assert_close(actual["grads"][name].cuda(), parameter.grad, rtol=2e-4, atol=2e-7)


def test_meta_four_gpu_exact_resume():
    rank, output = allocation("resume")
    _resume_worker(rank, 4, "env://", str(output), "fsdp2", True, "cpu", True, "cuda")
