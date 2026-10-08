"""Real SANA-anchor proximal-memory gates, one pytest process per Slurm GPU.

Run under the existing META_TEST_OUTPUT launcher with two, three or four ranks.
CPU collection skips these gates; it does not establish CUDA/FSDP correctness.
"""
import copy
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from test_alignment_detail import cached_sana
from test_full_training import long_inputs
from test_proximal_runtime import proximal_model
from tools.ttn_compare_resume import identical
from worldttn import training
from worldttn.checkpoint import load_checkpoint, make_optimizer, rng_state, restore_rng
from worldttn.distributed import ParallelTraining, resolve_launch_environment
from worldttn.memory_cache import detach_cache
from worldttn.parallel_checkpoint import (_pack, save_training_checkpoint,
                                         restore_training_checkpoint)
from worldttn.session import TTNSession


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or "META_TEST_OUTPUT" not in os.environ,
    reason="requires isolated CUDA acceptance allocation and shared META_TEST_OUTPUT")


@pytest.fixture
def allocation(request):
    launch = resolve_launch_environment()
    if launch["world_size"] < 2:
        pytest.skip("proximal FSDP2 acceptance requires at least two ranks")
    assert launch["world_size"] in (2, 3, 4)
    assert torch.cuda.device_count() == 1, "one visible GPU per pytest rank"
    torch.cuda.set_device(0)
    print(f"[TTN meta case] rank={launch['rank']} test={request.node.nodeid} "
          f"master={launch['master_addr']}:{launch['master_port']}", flush=True)
    dist.init_process_group("nccl", init_method="env://", rank=launch["rank"],
                            world_size=launch["world_size"])
    try:
        yield launch
    finally:
        dist.destroy_process_group()


def _inputs(rank):
    clean, noise, timestep, camera = (x.cuda() for x in long_inputs())
    return clean + .01 * rank, noise, timestep, camera


def _train(model, optimizer, engine, inputs, offload="cpu", callback=None):
    # Use the installed scheduler, not the CPU Euler oracle.
    clean, noise, timestep, camera = inputs
    def progress(phase, **info):
        indices = {key: info[key] for key in ("chunk", "first", "last") if key in info}
        print(f"[TTN meta phase] rank={engine.rank} phase={phase} {indices}", flush=True)
        if callback is not None:
            callback(phase, **info)
    return training.train_clip(model, clean, torch.zeros(1, 1, 2, 8, device="cuda"),
        camera, optimizer, training.linear_flow_loss, timestep, noise, width=100,
        height=100, tbptt=4, activation_offload=offload, parallel=engine,
        memory_callback=progress,
        history_training={"source": "generated", "steps": 4, "cached_chunks": 2})


def _snapshot(model, *, gradients=False):
    result = {}
    for name, parameter in model.named_parameters():
        value = parameter.grad if gradients else parameter
        if value is not None:
            value = value.detach()
            result[name] = (value.full_tensor() if hasattr(value, "full_tensor") else value).cpu().clone()
    return result


def _runtime_snapshot(runtime):
    assert runtime.commit_count == runtime.predict_count == 9
    assert runtime.committed_frame_ids == [set(range(25))]
    assert not runtime.world_state.requires_grad
    assert runtime.transition_fast.count_nonzero() == 0
    assert len(runtime.memory_caches) == 5 and runtime.verify_memory_prefix()
    fields = []
    for cache in runtime.memory_caches:
        observation = cache.observation
        assert observation.key.shape[2] <= 6 * cache.hw[0] * cache.hw[1]
        values = (observation.key, observation.value, observation.weight,
                  observation.token_indices, observation.frame_ids, cache.query)
        assert all(not value.requires_grad for value in values)
        fields.append(tuple(value.detach().cpu().clone() for value in values))
    return {"S": runtime.world_state.detach().cpu().clone(), "cache": fields,
            "prefix": runtime.memory_prefix_hashes}


@pytest.mark.parametrize("detach_history", [False, True], ids=["live-cache", "detached-cache-control"])
def test_proximal_fsdp_cache_only_future_credit(allocation, cached_sana, monkeypatch, detach_history):
    """Cut persistent S and native-cache paths; future loss must still reach K/V/W."""
    model = proximal_model(cached_sana).cuda()
    retained = []

    def capture(module, args, output):
        context = args[2]
        if context.clean_mode and not context.prefill_mode:
            observation = context.memory_candidates[4].observation
            tensors = (observation.key, observation.value, observation.weight)
            for tensor in tensors:
                assert tensor.requires_grad
                tensor.retain_grad()
            retained.append(tensors)

    model.blocks[19].register_forward_hook(capture)
    original = TTNSession.clean_forward

    def isolate(session, *args, **kwargs):
        result = original(session, *args, **kwargs)
        session.runtime.world_state = session.runtime.world_state.detach()
        assert not session.runtime.world_state.requires_grad
        # Native camera cache carry already detaches its tensors. Cutting only
        # this second path leaves identical forwards and isolates cache credit.
        if detach_history:
            session.runtime.memory_caches = tuple(detach_cache(cache)
                                                  for cache in session.runtime.memory_caches)
        return result

    monkeypatch.setattr(TTNSession, "clean_forward", isolate)
    engine = ParallelTraining(model, training.linear_flow_loss, "fsdp2", activation_offload="cpu")
    optimizer = make_optimizer(model)
    result = _train(model, optimizer, engine, _inputs(allocation["rank"]))
    _runtime_snapshot(result["runtime"])
    assert len(retained) == 8
    for tensor in retained[4]:
        if detach_history:
            assert tensor.grad is None
        else:
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
            assert tensor.grad.norm() > 0, "future loss did not reach clean cached K/V/W"
    for index in (3, 7):
        assert all(tensor.grad is None for tensor in retained[index]), "cache leaked across TBPTT boundary"
    # Full parameter gradients require FSDP's side-channel backward hooks, not
    # merely a nonzero retained cache-tensor gradient.
    grads = _snapshot(model, gradients=True)
    for name in ("blocks.19.attn.qkv.weight", "blocks.19.attn.beta_proj.weight"):
        assert name in grads and torch.isfinite(grads[name]).all() and grads[name].norm() > 0


def test_proximal_fsdp_offload_and_exact_resume(allocation, cached_sana):
    """Offload keeps existing FP32 tolerances; resume remains bitwise exact."""
    rank = allocation["rank"]
    baseline = proximal_model(cached_sana).cuda()
    inputs = _inputs(rank)
    engine = ParallelTraining(baseline, training.linear_flow_loss, "fsdp2")
    optimizer = make_optimizer(baseline)
    initial_rng = copy.deepcopy(rng_state())
    reference = _train(baseline, optimizer, engine, inputs, offload="none")
    reference_state, reference_grads = _snapshot(baseline), _snapshot(baseline, gradients=True)
    reference_runtime = _runtime_snapshot(reference["runtime"])
    reference_rng = copy.deepcopy(rng_state())
    del baseline, engine, optimizer, reference

    model = proximal_model(cached_sana).cuda()
    engine = ParallelTraining(model, training.linear_flow_loss, "fsdp2", activation_offload="cpu")
    optimizer = make_optimizer(model)
    restore_rng(initial_rng)
    phases = []
    result = _train(model, optimizer, engine, inputs,
                    callback=lambda phase, **info: phases.append((phase, info)))
    actual_state, actual_grads = _snapshot(model), _snapshot(model, gradients=True)
    assert actual_state.keys() == reference_state.keys()
    assert actual_grads.keys() == reference_grads.keys()
    for name in actual_state:
        torch.testing.assert_close(actual_state[name], reference_state[name], rtol=2e-5, atol=3e-7)
    for name in actual_grads:
        torch.testing.assert_close(actual_grads[name], reference_grads[name], rtol=2e-4, atol=2e-7)
    identical(reference_runtime, _runtime_snapshot(result["runtime"]), "offload runtime")
    identical(reference_rng, rng_state(), "offload RNG")
    assert any(info["offload_saved_tensors"]["packed_tensor_bytes"] > 0
               for phase, info in phases if phase == "backward_begin")

    path = Path(os.environ["META_TEST_OUTPUT"]) / "proximal-resume" / "last.pt"
    identity = {"tbptt": 4, "memory_update": "proximal",
                "history_training": training.history_settings(
                    {"source": "generated", "steps": 4, "cached_chunks": 2}),
                "execution": {"core_backend": "reference", "psi_backend": "reference"}}
    cursor = {"epoch": 1, "batch_in_epoch": 2, "rank": rank}
    save_training_checkpoint(path, engine, optimizer, 1, cursor, identity)
    expected = _train(model, optimizer, engine, inputs)
    expected_state = _snapshot(model)
    expected_optimizer = copy.deepcopy(_pack(optimizer.state_dict()))
    expected_runtime = _runtime_snapshot(expected["runtime"])
    expected_rng = copy.deepcopy(rng_state())
    del model, engine, optimizer

    resumed = proximal_model(cached_sana).cuda()
    load_checkpoint(path, resumed)
    resumed_engine = ParallelTraining(resumed, training.linear_flow_loss, "fsdp2", activation_offload="cpu")
    resumed_optimizer = make_optimizer(resumed)
    step, actual_cursor = restore_training_checkpoint(path, resumed_engine, resumed_optimizer, identity)
    assert step == 1 and actual_cursor == cursor
    actual = _train(resumed, resumed_optimizer, resumed_engine, inputs)
    assert actual["loss"] == expected["loss"]
    assert actual["outer_grad_norm"] == expected["outer_grad_norm"]
    identical(expected_state, _snapshot(resumed), "proximal model")
    identical(expected_optimizer, _pack(resumed_optimizer.state_dict()), "proximal Adam")
    identical(expected_rng, rng_state(), "proximal RNG")
    identical(expected_runtime, _runtime_snapshot(actual["runtime"]), "proximal S/cache")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["config"]["memory_update"] == "proximal"
    assert payload["distributed"]["world_size"] == allocation["world_size"]
    assert payload["optimizer"] is None
    assert all("world_state" not in name and "memory_cache" not in name for name in payload["adapter"])
