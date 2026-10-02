import json
from types import SimpleNamespace

import pytest
import torch

from worldttn.cuda_debug import cuda_diagnostics, trace_backward, tensor_metadata, CUDATrace
from worldttn.geometry import apply_complex_rope
from worldttn.training import activation_storage


def records(directory):
    return [json.loads(line) for line in (directory / "rank-0.jsonl").read_text().splitlines()]


def test_tensor_layout_metadata_is_read_only_and_reports_slice_alignment():
    base = torch.arange(80, dtype=torch.float32).reshape(2, 5, 8)
    view = base[..., 1::2].transpose(0, 1)
    before = (view.data_ptr(), view.stride(), base.clone())
    info = tensor_metadata({"view": view})[0]
    assert info["shape"] == list(view.shape) and info["dtype"] == "torch.float32"
    assert info["stride"] == list(view.stride()) and not info["contiguous"]
    assert info["storage_offset"] == 1 and info["element_size"] == 4
    assert info["pointer_mod"] == {str(n): view.data_ptr() % n for n in (16, 128, 256)}
    assert view.data_ptr() == before[0] and view.stride() == before[1] and torch.equal(base, before[2])


@pytest.mark.parametrize("storage", ["none", "cpu"])
def test_trace_preserves_complex_outputs_and_backward_and_removes_hooks(tmp_path, monkeypatch, storage):
    monkeypatch.setenv("RANK", "0")
    torch.manual_seed(3407)
    values = torch.randn(1, 2, 3, 18)
    freqs = torch.polar(torch.ones(1, 1, 3, 4, dtype=torch.float64),
                        torch.randn(1, 1, 3, 4, dtype=torch.float64))
    def evaluate(directory):
        source = values.clone().requires_grad_()
        with cuda_diagnostics(directory, "cpu"), activation_storage(storage):
            output = apply_complex_rope(source[..., 1:17:2], freqs)
            loss = output.square().sum()
            with trace_backward(loss): loss.backward(retain_graph=True)
        first = source.grad.clone()
        source.grad = None
        loss.backward()  # The diagnostic hooks must have been removed.
        assert torch.equal(source.grad, first)
        return output.detach(), first
    expected, expected_grad = evaluate(None)
    output, gradient = evaluate(tmp_path)
    assert torch.equal(output, expected) and torch.equal(gradient, expected_grad)
    log = records(tmp_path)
    assert log[0]["event"] == "trace.begin" and log[-1]["event"] == "trace.end"
    names = [r["name"] for r in log if r["event"] == "op.begin"]
    assert "aten.view_as_complex.default" in names and "aten.view_as_real.default" in names
    assert any(r["event"] == "node.begin" and "ViewAsComplexBackward" in r["name"] for r in log)
    assert any(r["event"] == "op.begin" and r.get("python_stack") for r in log)
    assert sum(r["event"] == "backward.begin" for r in log) == 1
    assert sum(r["event"] == "backward.end" for r in log) == 1
    leaves = [r for r in log if r["event"] == "node.end" and r["leaf"]]
    assert leaves and leaves[-1]["accumulated_grad"][0]["shape"] == list(values.shape)


def test_native_backward_failure_records_node_and_original_error_and_restores_context(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RANK", "0")
    class NativeKernelFailure(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x): return x.clone()
        @staticmethod
        def backward(ctx, grad): raise RuntimeError("native kernel diagnostic failure")
    with pytest.raises(RuntimeError, match="native kernel diagnostic failure"):
        with cuda_diagnostics(tmp_path, "cpu"):
            x = torch.ones(3, requires_grad=True)
            loss = NativeKernelFailure.apply(x).sum()
            with trace_backward(loss): loss.backward()
    log = records(tmp_path)
    error = next(r for r in log if r["event"] == "error")
    assert "NativeKernelFailureBackward" in error["last_node"]["name"]
    assert error["last_node"]["grad_outputs"][0]["shape"] == [3]
    assert "native kernel diagnostic failure" in error["python_stack"]
    console = capsys.readouterr()
    assert "[TTN CUDA error]" in console.out
    before = (tmp_path / "rank-0.jsonl").read_bytes()
    loss = torch.ones(2, requires_grad=True).square().sum()
    with trace_backward(loss): loss.backward()
    assert (tmp_path / "rank-0.jsonl").read_bytes() == before


def test_failure_after_operator_synchronization_keeps_last_operator_and_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    def fail(mode): raise RuntimeError("synchronous CUDA failure")
    x = torch.ones(2)
    monkeypatch.setattr(CUDATrace, "synchronize", fail)
    with pytest.raises(RuntimeError, match="synchronous CUDA failure"):
        with cuda_diagnostics(tmp_path, "cpu"): x + x
    error = next(r for r in records(tmp_path) if r["event"] == "error")
    assert error["where"] == "operator" and error["last_op"]["name"] == "aten.add.Tensor"
    assert error["last_op"]["tensors"][0]["shape"] == [2]


def test_diagnose_update_uses_same_inputs_and_rng_as_distributed_smoke_rank0(tmp_path, monkeypatch):
    from worldttn import cli, distributed
    from test_training import TinyWorldModel
    from test_parallel import _rank_inputs
    args = SimpleNamespace(seed=3407, stage="A", stages=["A"], parallel="single", frames=13,
                           output=str(tmp_path), tbptt=1, activation_offload="cpu", memory_trace=True,
                           batch_file=None, device="cpu", latent_height=22, latent_width=40)
    def build(args, stage=None):
        model = TinyWorldModel("A")
        return model, SimpleNamespace(), {"learning_rate": 1e-5, "tbptt": 2}
    def batch(*args):
        clean, noise, t, camera = _rank_inputs(0)
        return {"clean_latents": clean, "camera_conditions": camera, "y": torch.randn(1, 1, 2, 8)}
    captures = []
    class ReachedUpdate(Exception): pass
    def update(model, config, batch, optimizer, k, parallel):
        captures.append((torch.get_rng_state(), batch, {n: p.clone() for n, p in model.state_dict().items()}, k))
        raise ReachedUpdate
    monkeypatch.setattr(cli, "build", build)
    monkeypatch.setattr(cli, "synthetic_bundle", batch)
    monkeypatch.setattr(cli, "SANAFlowLoss", lambda config: None)
    monkeypatch.setattr(cli, "timed_cuda", lambda call: call())
    monkeypatch.setattr(cli, "train_update", update)
    monkeypatch.setattr(distributed, "ParallelTraining", lambda *a, **kw: None)
    monkeypatch.setattr(distributed, "rank_world", lambda: (0, 1))
    with pytest.raises(ReachedUpdate): cli.diagnose_update_command(args)
    args.parallel = "ddp"
    monkeypatch.setattr(distributed, "rank_world", lambda: (0, 4))
    with pytest.raises(ReachedUpdate): cli.distributed_smoke_command(args)
    first, second = captures
    assert torch.equal(first[0], second[0]) and first[3] == second[3] == 1
    for a, b in ((first[1], second[1]), (first[2], second[2])):
        assert a.keys() == b.keys()
        assert all(torch.equal(a[key], b[key]) for key in a)


def test_actual_single_update_diagnosis_preserves_training_states_and_optimizer(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    from worldttn import cli
    from worldttn.training import linear_flow_loss
    from test_training import TinyWorldModel, inputs
    timestep = ModuleType("train_video_scripts.train_sana_wm_stage1")
    timestep._build_time_sampler = lambda *args: None
    timestep._build_timesteps = lambda config, clean, frames, *args, **kwargs: (
        torch.ones(clean.shape[0], 1, frames) * 500, None)
    monkeypatch.setitem(sys.modules, timestep.__name__, timestep)
    monkeypatch.setenv("RANK", "0")
    models, results = [], []
    def build(args):
        model = TinyWorldModel("A")
        model.base_load_report = {"cpu_test": True}
        models.append(model)
        return model, SimpleNamespace(), {"learning_rate": 1e-5}
    def batch(*args):
        clean, noise, t, camera = inputs()
        return {"clean_latents": clean, "camera_conditions": camera, "y": torch.zeros(1, 1, 2, 8),
                "width": 100, "height": 100}
    update = cli.train_update
    def capture(*args, **kwargs):
        result = update(*args, **kwargs)
        results.append(result)
        return result
    monkeypatch.setattr(cli, "build", build)
    monkeypatch.setattr(cli, "synthetic_bundle", batch)
    monkeypatch.setattr(cli, "SANAFlowLoss", lambda config: linear_flow_loss)
    monkeypatch.setattr(cli, "train_update", capture)
    monkeypatch.setattr(cli, "timed_cuda", lambda call: (call(), {"seconds": 0}))
    args = SimpleNamespace(seed=3407, parallel="single", stage="A", frames=13, tbptt=1,
                           output=str(tmp_path), activation_offload="cpu", memory_trace=False,
                           device="cpu", batch_file=None, latent_height=22, latent_width=40,
                           launch={"world_size": 1})
    monkeypatch.delenv("TTN_CUDA_TRACE_DIR", raising=False)
    cli.diagnose_update_command(args)
    directory = tmp_path / "cuda-trace"
    monkeypatch.setenv("TTN_CUDA_TRACE_DIR", str(directory))
    cli.diagnose_update_command(args)
    for before, after in zip(models[0].parameters(), models[1].parameters()):
        assert torch.equal(before, after)
        assert (before.grad is None) == (after.grad is None)
        if before.grad is not None: assert torch.equal(before.grad, after.grad)
    a, b = results
    assert a["loss"] == b["loss"] and a["outer_grad_norm"] == b["outer_grad_norm"]
    assert torch.equal(a["runtime"].world_state, b["runtime"].world_state)
    assert torch.equal(a["runtime"].transition_fast, b["runtime"].transition_fast)
    assert a["runtime"].commit_count == b["runtime"].commit_count == 5
    report = json.loads((tmp_path / "update.json").read_text())
    assert report["train"]["parallel"] == "single" and report["tbptt"] == 1
    with (directory / "rank-0.jsonl").open() as handle:
        assert sum('"event": "backward.end"' in line for line in handle) == 4


@pytest.mark.parametrize("blocking,compile_gate,message", [
    ("0", "1", "CUDA_LAUNCH_BLOCKING=1"), ("1", "0", "GDN_DISABLE_COMPILE=1")])
def test_cli_rejects_async_or_compiled_operator_tracing_before_initialization(
        monkeypatch, capsys, blocking, compile_gate, message):
    import sys
    from worldttn import cli
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", blocking)
    monkeypatch.setenv("GDN_DISABLE_COMPILE", compile_gate)
    monkeypatch.setattr(sys, "argv", ["worldttn", "diagnose-update", "--cuda-trace"])
    with pytest.raises(SystemExit) as error: cli.main()
    assert error.value.code == 2 and message in capsys.readouterr().err


def test_actual_slurm_single_diagnosis_skips_process_group_and_binds_only_cuda_zero(tmp_path, monkeypatch):
    import sys
    from worldttn import cli, distributed
    from test_slurm import install_environment, slurm_env
    install_environment(monkeypatch, slurm_env(0, 1))
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    monkeypatch.setenv("GDN_DISABLE_COMPILE", "1")
    monkeypatch.setenv("TTN_CUDA_TRACE_DIR", "stale-parent-trace")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    bindings = []
    monkeypatch.setattr(torch.cuda, "set_device", bindings.append)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed.dist, "init_process_group", lambda *a, **kw: pytest.fail("single diagnosis must not initialize DDP/NCCL"))
    monkeypatch.setattr(distributed, "check_distributed", lambda device: {"world_size": 1})
    seen = []
    monkeypatch.setattr(cli, "diagnose_update_command", seen.append)
    monkeypatch.setattr(sys, "argv", ["worldttn", "diagnose-update", "--cuda-trace", "--parallel", "single",
                                      "--output", str(tmp_path), "--stage", "A", "--tbptt", "1", "--frames", "13"])
    cli.main()
    assert bindings == [0] and len(seen) == 1
    args = seen[0]
    assert args.parallel == "single" and args.device == "cuda:0"
    assert args.stage == "A" and args.tbptt == 1 and args.frames == 13
    import os
    assert os.environ["TTN_CUDA_TRACE_DIR"] == str(tmp_path / "cuda-trace")
