import copy
import torch
import pytest
from pathlib import Path
import socket
from test_training import TinyWorldModel, inputs
from worldttn.checkpoint import make_optimizer
from worldttn.training import train_clip, linear_flow_loss
import worldttn.training as training


def _rank_inputs(rank):
    clean, noise, t, camera = inputs()
    clean = clean + .3 * rank
    noise = noise + .1 * rank
    camera = camera.clone()
    camera[..., 3] = torch.arange(13) * .01 + rank * .02
    return clean, noise, t, camera


def _init_uri():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    return f"tcp://127.0.0.1:{port}?use_libuv=0"


def _parallel_worker(rank, size, init_uri, output, mode, stage, k, bf16=False):
    from worldttn.distributed import ParallelTraining
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init_uri, rank=rank, world_size=size)
    try:
        torch.manual_seed(17)
        model = TinyWorldModel(stage)
        if bf16:
            for block in model.blocks:
                block.ffn.bfloat16()
            # Only the frozen FFNs emulate BF16 base weights; TTN stays FP32.
        engine = ParallelTraining(model, linear_flow_loss, mode)
        optimizer = make_optimizer(model)
        clean, noise, t, camera = _rank_inputs(rank)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            result = train_clip(model, clean, torch.zeros(1, 1, 2, 8), camera, optimizer, linear_flow_loss, t, noise,
                                width=100, height=100, tbptt=k, parallel=engine)
        state = {}
        for name, value in model.state_dict().items():
            if hasattr(value, "full_tensor"): value = value.full_tensor()
            state[name] = value.cpu()
        torch.save({"state": state, "loss": result["loss"], "norm": result["outer_grad_norm"],
                    "S": result["runtime"].world_state, "psi": result["runtime"].transition_fast,
                    "commits": result["runtime"].commit_count}, Path(output) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("stage,k", [("A", 2), ("B", 2), ("C", 1), ("C", 2), ("C", 4)])
def test_two_rank_ddp_matches_single_global_batch_and_keeps_memory_local(tmp_path, stage, k):
    import importlib.util
    assert importlib.util.find_spec("worldttn.distributed"), "parallel engine is missing"
    torch.multiprocessing.spawn(_parallel_worker,
                                args=(2, _init_uri(), str(tmp_path), "ddp", stage, k), nprocs=2)
    torch.manual_seed(17)
    reference = TinyWorldModel(stage)
    batches = [_rank_inputs(rank) for rank in range(2)]
    clean, noise, t, camera = [torch.cat(values) for values in zip(*batches)]
    result = train_clip(reference, clean, torch.zeros(2, 1, 2, 8), camera, make_optimizer(reference),
                        linear_flow_loss, t, noise, width=100, height=100, tbptt=k)
    ranks = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=False) for rank in range(2)]
    for rank in ranks:
        assert rank["commits"] == 5
        assert rank["norm"] == pytest.approx(result["outer_grad_norm"], rel=2e-5)
        for name, tensor in reference.state_dict().items():
            torch.testing.assert_close(rank["state"][name], tensor, atol=3e-7, rtol=2e-5)
    assert sum(r["loss"] for r in ranks) / 2 == pytest.approx(result["loss"], rel=2e-6)
    assert not torch.equal(ranks[0]["S"], ranks[1]["S"])
    assert torch.equal(ranks[0]["state"]["blocks.3.attn.qkv.weight"],
                       ranks[1]["state"]["blocks.3.attn.qkv.weight"])


@pytest.mark.skipif(not hasattr(torch.cpu, "Stream"), reason="this PyTorch build requires CUDA for FSDP2")
@pytest.mark.parametrize("stage,k", [("A", 2), ("B", 2), ("C", 1), ("C", 2), ("C", 4)])
def test_two_rank_fsdp2_matches_single_global_batch(tmp_path, stage, k):
    torch.multiprocessing.spawn(_parallel_worker,
                                args=(2, _init_uri(), str(tmp_path), "fsdp2", stage, k), nprocs=2)
    torch.manual_seed(17)
    reference = TinyWorldModel(stage)
    batches = [_rank_inputs(rank) for rank in range(2)]
    clean, noise, t, camera = [torch.cat(values) for values in zip(*batches)]
    result = train_clip(reference, clean, torch.zeros(2, 1, 2, 8), camera, make_optimizer(reference),
                        linear_flow_loss, t, noise, width=100, height=100, tbptt=k)
    for rank in range(2):
        actual = torch.load(tmp_path / f"rank{rank}.pt", weights_only=False)
        assert actual["norm"] == pytest.approx(result["outer_grad_norm"], rel=2e-5)
        for name, tensor in reference.state_dict().items():
            torch.testing.assert_close(actual["state"][name], tensor, atol=3e-7, rtol=2e-5)


@pytest.mark.skipif(not hasattr(torch.cpu, "Stream"), reason="CPU FSDP2 not available")
def test_fsdp2_preserves_fp32_ttn_under_bf16_base_and_autocast(tmp_path):
    torch.multiprocessing.spawn(_parallel_worker,
                                args=(2, _init_uri(), str(tmp_path), "fsdp2", "C", 2, True), nprocs=2)
    ranks = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=False) for rank in range(2)]
    for rank in ranks:
        assert rank["S"].dtype == rank["psi"].dtype == torch.float32
        assert rank["state"]["blocks.3.attn.beta_proj.weight"].dtype == torch.float32
        assert rank["state"]["ttn_system.generators.u"].dtype == torch.float32
        assert rank["state"]["blocks.3.ffn.weight"].dtype == torch.bfloat16
        assert torch.isfinite(rank["S"]).all() and torch.isfinite(rank["psi"]).all()
    for name in ranks[0]["state"]:
        torch.testing.assert_close(ranks[0]["state"][name], ranks[1]["state"][name], atol=0, rtol=0)


def _resume_worker(rank, size, init_uri, output, mode, joint=False, activation_offload="none", meta=False, device="cpu", tbptt=2):
    from worldttn.distributed import ParallelTraining, save_training_checkpoint, restore_training_checkpoint
    from worldttn.checkpoint import load_checkpoint
    import torch.distributed as dist
    import random
    torch.set_num_threads(1)
    if device == "cuda": torch.cuda.set_device(0)
    dist.init_process_group("nccl" if device == "cuda" else "gloo", init_method=init_uri, rank=rank, world_size=size)
    try:
        torch.manual_seed(17)
        model = TinyWorldModel("C", local_update=meta, persistent_meta=meta).to(device)
        if joint:
            from worldttn.anchor import configure_train_scope
            configure_train_scope(model, "dit")
        engine = ParallelTraining(model, linear_flow_loss, mode, activation_offload=activation_offload)
        optimizer = make_optimizer(model)
        clean, noise, t, camera = [x.to(device) for x in _rank_inputs(rank)]
        text = torch.zeros(1, 1, 2, 8, device=device)
        kw = dict(width=100, height=100, tbptt=tbptt, activation_offload=activation_offload)
        train_clip(model, clean, text, camera, optimizer, linear_flow_loss, t, noise,
                   parallel=engine, **kw)
        random.seed(200 + rank)
        torch.manual_seed(300 + rank)
        path = Path(output) / "last.pt"
        cursor = {"epoch": 2, "batch_in_epoch": 3}
        save_training_checkpoint(path, engine, optimizer, 1, cursor, {"tbptt": tbptt, "seed": 3407})
        expected_random = (random.random(), torch.rand(3))
        second_noise = torch.randn_like(noise)
        train_clip(model, clean, text, camera, optimizer, linear_flow_loss, t, second_noise,
                   parallel=engine, **kw)
        expected = {}
        for name, value in model.state_dict().items():
            if hasattr(value, "full_tensor"): value = value.full_tensor()
            expected[name] = value.clone()
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        torch.manual_seed(17)
        resumed = TinyWorldModel("C", local_update=meta, persistent_meta=meta).to(device)
        if joint: configure_train_scope(resumed, "dit")
        load_checkpoint(path, resumed)
        new_engine = ParallelTraining(resumed, linear_flow_loss, mode, activation_offload=activation_offload)
        new_optimizer = make_optimizer(resumed)
        step, data = restore_training_checkpoint(path, new_engine, new_optimizer, {"tbptt": tbptt, "seed": 3407})
        assert step == 1 and data == cursor
        assert random.random() == expected_random[0]
        assert torch.equal(torch.rand(3), expected_random[1])
        train_clip(resumed, clean, text, camera, new_optimizer, linear_flow_loss, t,
                   torch.randn_like(noise), parallel=new_engine, **kw)
        for name, value in resumed.state_dict().items():
            if hasattr(value, "full_tensor"): value = value.full_tensor()
            torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
        if meta:
            actual_optimizer = new_optimizer.state_dict()
            assert expected_optimizer["param_groups"] == actual_optimizer["param_groups"]
            for key, values in expected_optimizer["state"].items():
                for field, value in values.items():
                    torch.testing.assert_close(value, actual_optimizer["state"][key][field], atol=0, rtol=0)
        with pytest.raises(ValueError, match="training configuration"):
            restore_training_checkpoint(path, new_engine, new_optimizer, {"tbptt": 2 if tbptt == 4 else 4, "seed": 3407})
        payload = torch.load(path, weights_only=False)
        assert payload["optimizer"] is None
        assert not any(hasattr(value, "full_tensor") for value in payload["adapter"].values())
        assert all("world_state" not in key for key in payload["adapter"])
        if joint:
            assert payload["train_scope"] == payload["weight_scope"] == "dit"
            assert payload["adapter"].keys() == model.state_dict().keys()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["ddp", "fsdp2"])
def test_distributed_resume_reproduces_next_update_and_exports_plain_adapter(tmp_path, mode):
    import worldttn.distributed as distributed
    assert hasattr(distributed, "save_training_checkpoint"), "distributed checkpoint support is missing"
    if mode == "fsdp2" and not hasattr(torch.cpu, "Stream"): pytest.skip("CPU FSDP2 not available")
    torch.multiprocessing.spawn(_resume_worker, args=(2, _init_uri(), str(tmp_path), mode), nprocs=2)


@pytest.mark.skipif(not hasattr(torch.cpu, "Stream"), reason="CPU FSDP2 not available")
def test_three_rank_fsdp2_meta_offload_tbptt4_exact_resume(tmp_path):
    torch.multiprocessing.spawn(_resume_worker,
        args=(3, _init_uri(), str(tmp_path), "fsdp2", True, "cpu", True, "cpu", 4), nprocs=3)
    from worldttn.checkpoint_integrity import audit_checkpoint
    report = audit_checkpoint(tmp_path / "last.pt", 1)
    assert report["world_size"] == 3 and report["status"] == "verified"


@pytest.mark.parametrize("k", [1, 2, 4])
def test_window_boundary_covers_predict_and_clean_and_matches_reference(k):
    assert hasattr(training, "TTNTrainingWindow"), "distributed training needs an nn.Module window boundary"
    torch.manual_seed(17)
    model = TinyWorldModel("C")
    reference = copy.deepcopy(model)
    clean, noise, t, camera = inputs()
    kw = dict(width=100, height=100, tbptt=k)
    target = train_clip(reference, clean, torch.zeros(1, 1, 2, 8), camera,
                        make_optimizer(reference), linear_flow_loss, t, noise, **kw)
    window = training.TTNTrainingWindow(model, linear_flow_loss)
    calls = []
    window.register_forward_pre_hook(lambda module, args: calls.append("begin"))
    window.register_forward_hook(lambda module, args, out: calls.append("end"))
    result = train_clip(model, clean, torch.zeros(1, 1, 2, 8), camera,
                        make_optimizer(model), linear_flow_loss, t, noise, window_model=window, **kw)
    assert calls == ["begin", "end"] * (1 + (4 + k - 1) // k)  # prefill is also wrapped
    assert result["loss"] == pytest.approx(target["loss"], abs=1e-7)
    assert result["runtime"].commit_count == 5
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
