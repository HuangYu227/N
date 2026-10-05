"""Native FSDP2 plus actual saved-tensor CPU copies across TTN windows."""
from pathlib import Path
import pytest
import torch
from test_training import TinyWorldModel
from test_parallel import _rank_inputs, _init_uri, _resume_worker
from worldttn.anchor import configure_train_scope
from worldttn.checkpoint import make_optimizer
from worldttn.training import train_clip, linear_flow_loss


def _available():
    from torch.distributed import fsdp
    return hasattr(fsdp, "fully_shard") and hasattr(torch.cpu, "Stream")


def _offload_worker(rank, size, uri, output, k, amp, meta=False, device="cpu"):
    from worldttn.distributed import ParallelTraining
    from worldttn.training_health import FirstUpdateProbe
    import torch.distributed as dist
    torch.set_num_threads(1)
    if device == "cuda": torch.cuda.set_device(0)
    dist.init_process_group("nccl" if device == "cuda" else "gloo", init_method=uri, rank=rank, world_size=size)
    try:
        torch.manual_seed(17)
        model = TinyWorldModel("C", local_update=meta, persistent_meta=meta).to(device)
        configure_train_scope(model, "dit")
        live = []
        if meta:
            def retain_gradient(module, args, out):
                context = args[2]
                if context.clean_mode:
                    g = context.candidates[0][1]
                    if g.requires_grad:
                        g.retain_grad()
                        live.append(g)
            model.blocks[3].register_forward_hook(retain_gradient)
        engine = ParallelTraining(model, linear_flow_loss, "fsdp2", activation_offload="cpu")
        optimizer = make_optimizer(model)
        probe = FirstUpdateProbe(model)
        clean, noise, t, camera = [x.to(device) for x in _rank_inputs(rank)]
        copies = []
        original = torch.autograd.graph.save_on_cpu
        class RecordCopies(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                pack = self.pack_hook
                def record(tensor):
                    packed = pack(tensor)
                    if tensor.numel():
                        copies.append((tensor.data_ptr() != packed[1].data_ptr(), tensor.numel()))
                    return packed
                self.pack_hook = record
        torch.autograd.graph.save_on_cpu = RecordCopies
        with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
            result = train_clip(model, clean, torch.zeros(1, 1, 2, 8, device=device), camera, optimizer, linear_flow_loss,
                                t, noise, width=100, height=100, tbptt=k, parallel=engine,
                                activation_offload="cpu")
        torch.autograd.graph.save_on_cpu = original
        health = probe.report(optimizer)
        if meta:
            assert len(live) == 4
            assert live[0].grad is not None and live[0].grad.norm() > 0
            assert live[1].grad is None and live[2].grad is not None and live[2].grad.norm() > 0 and live[3].grad is None
            assert result["optimizer_updates"]["groups"]["ttn_system.local_eta_logits"]["delta_norm"] > 0
        assert copies and all(copied for copied, _ in copies), "CPU tests must really copy saved tensors"
        assert not health["missing_core_gradients"]
        assert health["backbone"]["by_component"]["ffn"]["updated_parameters"] > 0
        memory = engine.storage_record(optimizer)
        assert memory["sharded_parameters"] > 0
        assert memory["replicated_parameters"] == 0
        assert memory["local_parameter_bytes"] <= memory["global_parameter_bytes"] / size + 4 * len(list(model.parameters()))
        assert memory["local_gradient_bytes"] > 0 and memory["local_optimizer_bytes"] > 0
        state, grads = {}, {}
        for name, parameter in model.named_parameters():
            value = parameter.detach()
            state[name] = (value.full_tensor() if hasattr(value, "full_tensor") else value).cpu()
            if parameter.grad is not None:
                grad = parameter.grad.detach()
                grads[name] = (grad.full_tensor() if hasattr(grad, "full_tensor") else grad).cpu()
        torch.save({"state": state, "grads": grads, "norm": result["outer_grad_norm"], "loss": result["loss"],
                    "S": result["runtime"].world_state, "psi": result["runtime"].transition_fast,
                    "commits": result["runtime"].commit_count, "predictions": result["runtime"].predict_count,
                    "memory": memory}, Path(output) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _available(), reason="CPU FSDP2 not available")
@pytest.mark.parametrize("k,amp,meta", [(1, False, False), (2, False, False), (4, False, False),
                                      (1, True, False), (2, True, False), (4, True, False), (2, False, True), (2, True, True)])
def test_fsdp_cpu_offload_real_copies_match_global_batch_and_keep_runtime_local(tmp_path, k, amp, meta):
    torch.multiprocessing.spawn(_offload_worker, args=(2, _init_uri(), str(tmp_path), k, amp, meta), nprocs=2)
    torch.manual_seed(17)
    model = TinyWorldModel("C", local_update=meta, persistent_meta=meta)
    configure_train_scope(model, "dit")
    clean, noise, t, camera = [torch.cat(values) for values in zip(*[_rank_inputs(r) for r in range(2)])]
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        result = train_clip(model, clean, torch.zeros(2, 1, 2, 8), camera, make_optimizer(model),
                            linear_flow_loss, t, noise, width=100, height=100, tbptt=k, activation_offload="cpu")
    ranks = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=False) for rank in range(2)]
    assert sum(x["loss"] for x in ranks) / 2 == pytest.approx(result["loss"], rel=2e-6)
    norm = result["outer_grad_norm"]
    if amp:
        # BF16 rounds each rank's matmul gradient before the FP32 reduction.
        # Match those local batches instead of loosening FP32 tolerances for
        # a batch-2 matmul, which rounds after summing the two contributions.
        torch.manual_seed(17)
        model = TinyWorldModel("C", local_update=meta, persistent_meta=meta)
        configure_train_scope(model, "dit")
        optimizer = make_optimizer(model)
        step, optimizer.step = optimizer.step, lambda: None
        averaged = {}
        for rank in range(2):
            clean, noise, t, camera = _rank_inputs(rank)
            with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
                local = train_clip(model, clean, torch.zeros(1, 1, 2, 8), camera, optimizer,
                                   linear_flow_loss, t, noise, width=100, height=100, tbptt=k,
                                   activation_offload="cpu")
            assert local["outer_grad_norm"] < .5, "reference averaging requires unclipped local gradients"
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    averaged[name] = averaged.get(name, torch.zeros_like(parameter.grad)) + parameter.grad / 2
        for name, parameter in model.named_parameters(): parameter.grad = averaged.get(name)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), .5))
        step()
    for row in ranks:
        assert row["commits"] == row["predictions"] == 5
        assert row["norm"] == pytest.approx(norm, rel=2e-5)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(row["state"][name], parameter, rtol=2e-5, atol=3e-7)
            if parameter.grad is not None:
                torch.testing.assert_close(row["grads"][name], parameter.grad, rtol=2e-4, atol=2e-7)
    assert not torch.equal(ranks[0]["S"], ranks[1]["S"])
    torch.testing.assert_close(torch.cat([x["S"] for x in ranks]), result["runtime"].world_state, rtol=2e-5, atol=3e-7)
    torch.testing.assert_close(torch.cat([x["psi"] for x in ranks]), result["runtime"].transition_fast, rtol=2e-5, atol=3e-7)


@pytest.mark.skipif(not _available(), reason="CPU FSDP2 not available")
def test_fsdp_offload_full_checkpoint_exact_resume_and_real_cli(tmp_path):
    from test_joint_training import _joint_cli_worker
    resumed = tmp_path / "resume"
    resumed.mkdir()
    torch.multiprocessing.spawn(_resume_worker,
                                args=(2, _init_uri(), str(resumed), "fsdp2", True, "cpu"), nprocs=2)
    cli_output = tmp_path / "cli"
    cli_output.mkdir()
    torch.multiprocessing.spawn(_joint_cli_worker,
                                args=(2, _init_uri(), str(cli_output), "fsdp2"), nprocs=2)
