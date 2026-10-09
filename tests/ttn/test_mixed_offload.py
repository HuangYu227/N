"""A bounded saved-tensor GPU tier must preserve the native CPU-offload math."""
import copy
import gc
import weakref
from collections import Counter

import pytest
import torch

from test_activation_checkpoint import cached_ffn, cached_sana, text_attention
from worldttn import training


@pytest.mark.parametrize("budget", [-1., float("nan"), float("inf"), -float("inf"), True, False])
@pytest.mark.parametrize("entry", ["storage", "train_clip"])
def test_gpu_budget_rejects_invalid_values_before_training(entry, budget):
    with pytest.raises(ValueError, match="budget"):
        if entry == "storage":
            training.activation_storage("cpu", device="cpu", gpu_budget_gib=budget)
        else:
            training.train_clip(None, torch.zeros(1, 1, 1, 1, 1), None, None, None, None,
                                None, None, width=1, height=1, activation_offload="cpu",
                                activation_gpu_budget_gib=budget)


@pytest.mark.parametrize("entry", ["storage", "train_clip"])
def test_nonzero_gpu_budget_requires_cpu_offload(entry):
    with pytest.raises(ValueError, match="budget"):
        if entry == "storage":
            training.activation_storage("none", device="cpu", gpu_budget_gib=.01)
        else:
            training.train_clip(None, torch.zeros(1, 1, 1, 1, 1), None, None, None, None,
                                None, None, width=1, height=1, activation_gpu_budget_gib=.01)


def test_zero_budget_keeps_native_cpu_hooks_and_audit(monkeypatch):
    original = torch.autograd.graph.save_on_cpu
    native = []

    def save_on_cpu(**kwargs):
        assert kwargs == {"pin_memory": True}
        storage = original(pin_memory=False)
        native.append(storage)
        return storage

    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", save_on_cpu)
    tensor = torch.arange(12., dtype=torch.float64).reshape(3, 4).t().requires_grad_()
    for kwargs in ({}, {"gpu_budget_gib": 0.}):
        audit = Counter()
        storage = training.activation_storage("cpu", device="cuda:0", audit=audit, **kwargs)
        assert storage is native[-1]
        with storage:
            tensor.square().sum().backward()
        torch.testing.assert_close(tensor.grad, 2*tensor, rtol=0, atol=0)
        tensor.grad = None
        assert len(audit) == 1 and len(next(iter(audit))) == 5
        report = training.activation_storage_stats(audit)
        assert report["pack_calls"] == 1 and report["packed_tensor_bytes"] == 96


def test_cpu_device_budget_does_not_enable_pinned_packing(monkeypatch):
    def unexpected(**kwargs):
        raise AssertionError("CPU tensors do not need saved-tensor offload")

    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", unexpected)
    tensor = torch.arange(12., dtype=torch.float64).reshape(3, 4).t().requires_grad_()
    with training.activation_storage("cpu", device="cpu", gpu_budget_gib=.01):
        tensor.square().sum().backward()
    torch.testing.assert_close(tensor.grad, 2*tensor, rtol=0, atol=0)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA saved-tensor offload")


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.int64])
@pytest.mark.parametrize("layout", ["transpose", "slice", "expand"])
def test_gpu_pack_is_independent_and_matches_native_cpu_layout(dtype, layout):
    base = torch.arange(24, device="cuda", dtype=dtype).reshape(4, 6)
    source = {"transpose": lambda: base.t(), "slice": lambda: base[:, 1::2],
              "expand": lambda: base[:1].expand(4, 6)}[layout]()
    expected = source.clone(memory_format=torch.contiguous_format)
    payload_bytes = source.numel()*source.element_size()
    storage = training.activation_storage("cpu", device=source.device,
                                          gpu_budget_gib=payload_bytes/2**30)
    packed = storage.pack_hook(source)
    assert packed[1].device == source.device and packed[1].dtype == source.dtype
    assert packed[1].untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    native = torch.autograd.graph.save_on_cpu(pin_memory=True).pack_hook(source)
    assert packed[1].stride() == native[1].stride()
    base.zero_()
    torch.testing.assert_close(storage.unpack_hook(packed), expected, rtol=0, atol=0)


@cuda
def test_tiny_gpu_budget_routes_overflow_to_native_pinned_cpu():
    audit = Counter()
    storage = training.activation_storage("cpu", device="cuda", audit=audit, gpu_budget_gib=16/2**30)
    small = torch.arange(4., device="cuda")
    large = torch.arange(8., device="cuda")
    gpu = storage.pack_hook(small)
    cpu = storage.pack_hook(large)
    assert gpu[1].device.type == "cuda"
    assert cpu[1].device.type == "cpu" and cpu[1].is_pinned()
    torch.testing.assert_close(storage.unpack_hook(gpu), small, rtol=0, atol=0)
    torch.testing.assert_close(storage.unpack_hook(cpu), large, rtol=0, atol=0)
    report = training.activation_storage_stats(audit)
    assert report["pack_calls"] == 2 and report["packed_tensor_bytes"] == 48
    assert report["gpu_packed_tensor_bytes"] == 16 and report["cpu_packed_tensor_bytes"] == 32
    assert all(not isinstance(value, torch.Tensor) for key in audit for value in key)
    del gpu
    gc.collect()
    # The cumulative window cap remains conservative after an early payload is freed.
    assert storage.pack_hook(small)[1].device.type == "cpu"


@cuda
@pytest.mark.parametrize("budget_bytes", [0, 128])
def test_packed_weight_view_survives_fsdp_storage_release(budget_bytes):
    source = torch.arange(16., dtype=torch.float64, device="cuda").reshape(4, 4)
    view = source.t()
    expected = view.clone(memory_format=torch.contiguous_format)
    storage = training.activation_storage("cpu", device="cuda", gpu_budget_gib=budget_bytes/2**30)
    packed = storage.pack_hook(view)
    torch.cuda.synchronize()
    # FSDP resharding frees the all-gather storage while autograd still owns its weight views.
    source.untyped_storage().resize_(0)
    torch.testing.assert_close(storage.unpack_hook(packed), expected, rtol=0, atol=0)


@cuda
def test_saved_payloads_release_after_backward_and_next_window_gets_a_fresh_budget():
    source = torch.linspace(-.4, .4, 16, device="cuda", requires_grad=True)
    for _ in range(2):
        audit, snapshots = Counter(), []
        storage = training.activation_storage("cpu", device="cuda", audit=audit, gpu_budget_gib=64/2**30)
        pack = storage.pack_hook

        def record(tensor):
            packed = pack(tensor)
            snapshots.append(weakref.ref(packed[1]))
            return packed

        storage.pack_hook = record
        with storage:
            loss = source.sin().square().sum()
        assert snapshots and all(snapshot() is not None for snapshot in snapshots)
        report = training.activation_storage_stats(audit)
        assert report["gpu_packed_tensor_bytes"] == 64 and report["cpu_packed_tensor_bytes"] > 0
        loss.backward()
        gc.collect()
        assert all(snapshot() is None for snapshot in snapshots)
        source.grad = None


@cuda
def test_mixed_storage_preserves_noncontiguous_loss_gradients_and_higher_derivatives():
    torch.manual_seed(407)
    x0 = torch.randn(3, 8, dtype=torch.float64, device="cuda")
    w0 = torch.randn(4, 3, dtype=torch.float64, device="cuda")
    results = []
    for budget_bytes in (0, 256):
        x = x0.clone()[:, ::2].t().requires_grad_()
        weight = w0.clone().t().requires_grad_()
        assert not x.is_contiguous() and not weight.is_contiguous()
        audit = Counter()
        with training.activation_storage("cpu", device="cuda", audit=audit,
                                         gpu_budget_gib=budget_bytes/2**30):
            output = (x @ weight).sin()
            loss = output.square().sum() + x.exp().mean()
            gradients = torch.autograd.grad(loss, (x, weight), create_graph=True)
            higher = torch.autograd.grad(sum(grad.square().sum() for grad in gradients), (x, weight))
        results.append((loss.detach(), tuple(grad.detach() for grad in gradients), higher))
        if budget_bytes:
            report = training.activation_storage_stats(audit)
            assert 0 < report["gpu_packed_tensor_bytes"] <= budget_bytes
            assert report["cpu_packed_tensor_bytes"] > 0
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("pure_checkpoint", [False, True])
def test_train_clip_mixed_storage_preserves_meta_updates_and_resets_window_budget(
        cached_sana, cached_ffn, text_attention, pure_checkpoint):
    from test_training import inputs
    from test_full_training import model as native_model
    from test_activation_checkpoint import pure_model
    from worldttn.checkpoint import make_optimizer

    torch.manual_seed(17)
    reference = (pure_model(cached_sana, cached_ffn, text_attention) if pure_checkpoint
                 else native_model(cached_sana)).cuda()
    if pure_checkpoint:
        for block in reference.blocks:
            block.mlp.ttn_activation_checkpointing = block.cross_attn.ttn_activation_checkpointing = True
    candidate = copy.deepcopy(reference)
    clean, noise, timesteps, camera = [value.cuda() for value in inputs()]
    results = []
    reports = []

    def record(phase, **info):
        if phase == "backward_begin":
            reports.append(info["offload_saved_tensors"])

    for model, budget in ((reference, 0.), (candidate, 4096/2**30)):
        optimizer = make_optimizer(model)
        result = training.train_clip(model, clean, torch.zeros(1, 1, 2, 8, device="cuda"), camera,
                                     optimizer, training.linear_flow_loss, timesteps, noise,
                                     width=100, height=100, tbptt=2, activation_offload="cpu",
                                     activation_gpu_budget_gib=budget,
                                     memory_callback=record if budget else None)
        results.append(result)
    assert len(reports) == 2
    assert all(0 < row["gpu_packed_tensor_bytes"] <= 4096 and row["cpu_packed_tensor_bytes"] > 0
               for row in reports)
    assert results[0]["loss"] == results[1]["loss"]
    assert results[0]["outer_grad_norm"] == results[1]["outer_grad_norm"]
    for (name, expected), (_, actual) in zip(reference.named_parameters(), candidate.named_parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0, msg=name)
    assert candidate.ttn_system.local_eta_logits.grad.norm() > 0
    for name in ("world_state", "transition_fast", "replay_values"):
        torch.testing.assert_close(getattr(results[0]["runtime"], name), getattr(results[1]["runtime"], name),
                                   rtol=0, atol=0)
    assert results[0]["runtime"].commit_count == results[1]["runtime"].commit_count == 5
    assert results[0]["runtime"].sink_reference_sha256 == results[1]["runtime"].sink_reference_sha256
    assert results[0]["runtime"].replay_reference_sha256 == results[1]["runtime"].replay_reference_sha256


@cuda
def test_proximal_generated_history_mixed_storage_preserves_training_and_cache(monkeypatch, cached_sana):
    from test_full_training import EulerOracle, long_inputs
    from test_proximal_runtime import proximal_model
    from test_proximal_cuda import _runtime_snapshot
    from tools.ttn_compare_resume import identical
    from worldttn.checkpoint import make_optimizer, rng_state, restore_rng

    monkeypatch.setattr(training, "_history_scheduler", EulerOracle)
    reference = proximal_model(cached_sana).cuda()
    candidate = copy.deepcopy(reference)
    clean, noise, timesteps, camera = [value.cuda() for value in long_inputs()]
    initial_rng = copy.deepcopy(rng_state())
    snapshots = []
    for model, budget in ((reference, 0.), (candidate, 4096/2**30)):
        restore_rng(initial_rng)
        optimizer = make_optimizer(model)
        reports, boundaries = [], []

        def record(phase, **info):
            if phase == "backward_begin":
                reports.append(info["offload_saved_tensors"])

        def check_boundary(module, args):
            context = args[2]
            if not context.clean_mode and int(context.frame_ids[0, 0]) == 13:
                for cache in context.memory_caches:
                    assert all(not value.requires_grad for value in
                               (cache.observation.key, cache.observation.value, cache.observation.weight,
                                cache.query))
                boundaries.append(13)

        handle = model.blocks[3].register_forward_pre_hook(check_boundary)
        result = training.train_clip(model, clean, torch.zeros(1, 1, 2, 8, device="cuda"), camera,
                                     optimizer, training.linear_flow_loss, timesteps, noise,
                                     width=100, height=100, tbptt=4, activation_offload="cpu",
                                     activation_gpu_budget_gib=budget, memory_callback=record,
                                     history_training={"source": "generated", "steps": 2, "cached_chunks": 2})
        handle.remove()
        assert boundaries
        assert all(not parameter.requires_grad and parameter.grad is None
                   for parameter in model.ttn_system.parameters())
        snapshots.append({"loss": result["loss"], "norm": result["outer_grad_norm"],
                          "model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                          "grads": {name: value.grad.detach().cpu().clone() for name, value in model.named_parameters()
                                    if value.grad is not None},
                          "optimizer": copy.deepcopy(optimizer.state_dict()),
                          "runtime": _runtime_snapshot(result["runtime"]), "rng": copy.deepcopy(rng_state())})
        assert len(reports) == 2
        if budget:
            assert all(0 < row["gpu_packed_tensor_bytes"] <= 4096 and row["cpu_packed_tensor_bytes"] > 0
                       for row in reports)
    identical(snapshots[0], snapshots[1], "proximal generated-history mixed offload")
