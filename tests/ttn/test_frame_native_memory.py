"""Hardware dispatch and independent saved-tensor storage for frame/fullgrad."""
import ast
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from worldttn.training import activation_storage, activation_storage_stats
from test_sequence_training import sequence_model, run_forward
from test_activation_checkpoint import cached_ffn, text_attention
from test_alignment_detail import cached_sana
from test_training import inputs


ROOT = Path(__file__).resolve().parents[2]


def test_backward_dispatch_uses_tensor_device_and_low_sram_column_tiles():
    path = ROOT / "diffusion/model/ops/fused_gdn_chunkwise_bwd.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_resolve_bwd_params", "_phase_a_kv_block_col"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    requested = []
    cuda = SimpleNamespace(is_available=lambda: True,
        get_device_capability=lambda device: requested.append(device) or (8, 9),
        get_device_properties=lambda device: SimpleNamespace(shared_memory_per_block_optin=100 * 1024))
    namespace = {"torch": SimpleNamespace(cuda=cuda), "_arch_key": lambda cap: "ada",
                 "_BWD_LAUNCH_PARAMS": {"ampere": {"BLOCK_S": 64}, "ada": {"BLOCK_S": 16}}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    assert "_phase_a_kv_block_col" in namespace, "D=112 requires the low-SRAM 128-wide backward path"
    assert namespace["_resolve_bwd_params"]("cuda:1") == {"BLOCK_S": 16}
    assert requested == ["cuda:1"]
    dispatch = namespace["_phase_a_kv_block_col"]
    assert dispatch("cuda:1", 128) == 32
    assert dispatch("cuda:1", 64) == 64
    cuda.get_device_properties = lambda device: SimpleNamespace(shared_memory_per_block_optin=164 * 1024)
    assert dispatch("cuda:1", 128) == 128


def test_pageable_cpu_input_preserves_gradients_and_validates_budget():
    value = torch.randn(3, 5, dtype=torch.float64).t().requires_grad_()
    with activation_storage("cpu-pageable", device="cpu", gpu_budget_gib=.001):
        value.square().sum().backward()
    torch.testing.assert_close(value.grad, 2 * value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="budget"):
        activation_storage("cpu-pageable", device="cpu", gpu_budget_gib=-1)


def test_last_group_trains_prefix_beyond_history_start_without_temporal_detach(sequence_model):
    sequence_model.ttn_system.config = replace(sequence_model.ttn_system.config,
        memory_capacity_frames=16, memory_prefix_frames=10, memory_recent_frames=4, memory_start_frame=13)
    for index in (3, 7, 11, 15, 19):
        sequence_model.blocks[index].attn.config = sequence_model.ttn_system.config
        assert sequence_model.blocks[index].attn.config.memory_prefix_frames == 10
    for block in sequence_model.blocks:
        block.ttn_sequence_checkpoint = True
    clean, _, time, camera = inputs()
    clean, time = [torch.cat((value, value[:, :, 1:7]), 2) for value in (clean, time)]
    camera = torch.cat((camera, camera[:, 1:7]), 1)
    value = clean.expand(-1, -1, -1, 1, 2).clone().requires_grad_()
    output, context, session = run_forward(sequence_model, value, time, camera)
    output[:, :, -3:].square().mean().backward()
    assert value.grad[:, :, :1].norm() > 0
    assert value.grad[:, :, 1:4].norm() > 0
    assert session.runtime.commit_count == 0
    for index in range(5):
        retained_ids = context.memory_candidates[index].observation.frame_ids
        assert torch.isin(torch.arange(10), retained_ids[0]).all()
        assert retained_ids.shape[-1] <= 16 * 2
        records = context.memory_stats[index][-1]["proximal"]["frames"]
        assert all(record["history_active"] == [True] for record in records)
        assert records[-1]["frame_ids"] == [[18]]


@pytest.mark.parametrize("name", ["DATA_DIR", "VAE_CACHE_DIR", "BASE_WEIGHTS"])
def test_launcher_rejects_missing_input_before_initializing_torch(name):
    git = shutil.which("git")
    bash = str(Path(git).resolve().parents[1] / "bin/bash.exe") if os.name == "nt" and git else shutil.which("bash")
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash unavailable for launcher input checks")
    env = os.environ.copy()
    for key in ("DATA_DIR", "VAE_CACHE_DIR", "BASE_WEIGHTS"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES="0,1", DATASET_ROOT=ROOT.as_posix(),
               PYTHON="python-must-not-be-called", **{name: (ROOT / "missing-input-for-test").as_posix()})
    result = subprocess.run([bash, "tools/ttn_frame_train.sh"], cwd=ROOT, env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=15)
    if "NtCreateDirectoryObject" in result.stderr:
        pytest.skip("sandbox blocks Git Bash's MSYS runtime object")
    assert result.returncode == 2
    assert "Missing" in result.stderr and "python-must-not-be-called" not in result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA saved-tensor snapshot check")
@pytest.mark.parametrize("budget_bytes", [0, 128])
def test_pageable_saved_weight_snapshot_survives_storage_release(budget_bytes):
    source = torch.arange(16., dtype=torch.float64, device="cuda").reshape(4, 4)
    expected = source.t().contiguous()
    audit = Counter()
    storage = activation_storage("cpu-pageable", device=source.device, audit=audit,
                                 gpu_budget_gib=budget_bytes / 2**30)
    packed = storage.pack_hook(source.t())
    assert packed[1].is_contiguous() and not packed[1].is_pinned()
    assert packed[1].device.type == ("cuda" if budget_bytes else "cpu")
    source.untyped_storage().resize_(0)
    torch.testing.assert_close(storage.unpack_hook(packed), expected, rtol=0, atol=0)
    report = activation_storage_stats(audit)
    assert report["packed_tensor_bytes"] == 128
    assert report["gpu_packed_tensor_bytes"] == budget_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pageable offload gradient check")
def test_pageable_and_pinned_offload_preserve_first_and_second_derivatives():
    torch.manual_seed(712)
    initial = torch.randn(4, 6, dtype=torch.float64, device="cuda")
    outputs = []
    for mode, budget in (("none", 0), ("cpu", 0), ("cpu-pageable", 0), ("cpu-pageable", 128)):
        value = initial.clone().t().requires_grad_()
        with activation_storage(mode, device=value.device, gpu_budget_gib=budget / 2**30):
            loss = value.sin().square().sum()
            first = torch.autograd.grad(loss, value, create_graph=True)[0]
            second = torch.autograd.grad(first.square().sum(), value)[0]
        outputs.append((loss.detach(), first.detach(), second))
    for actual in outputs[1:]:
        torch.testing.assert_close(actual, outputs[0], rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("FRAME_NATIVE_CUDA_TEST") != "1",
                   reason="opt in to the native Triton H20/D112 CUDA gate")
@pytest.mark.parametrize("frames", [1, 3])
def test_native_phase_a_real_head_dimension_backward(frames, monkeypatch):
    # CUDA floating point dot reductions can differ by column grouping.
    from diffusion.model.ops import fused_gdn_chunkwise_bwd as native
    assert torch.cuda.is_available()
    # Exercise the L20 column-tiled branch even when the allocated gate GPU is an A100.
    monkeypatch.setattr(native, "_phase_a_kv_block_col", lambda device, width: 32)
    torch.manual_seed(90)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    shape = (20, frames, 17, 112)
    key, value = [(torch.randn(shape, device=device, dtype=torch.bfloat16) * .05).float() for _ in range(2)]
    beta = torch.sigmoid(torch.randn(shape[:-1], device=device))
    d_a, d_p = [torch.randn(20, frames, 112, 112, device=device) * .05 for _ in range(2)]
    d_b = torch.randn(20, frames, 112, device=device) * .05
    expected_k = beta[..., None] * (key.float() @ d_p.bfloat16().float()
        + key.float() @ d_p.transpose(-1, -2).bfloat16().float()
        + value.float() @ d_a.transpose(-1, -2).bfloat16().float())
    expected_v = beta[..., None] * (key.float() @ d_a.bfloat16().float())
    expected_beta = ((key.float() @ d_p.bfloat16().float()) * key.float()).sum(-1)
    expected_beta += ((key.float() @ d_a.bfloat16().float()) * value.float()).sum(-1)
    actual = native.phase_a_kv_bwd(key, value, beta, d_a, d_p, 112)
    torch.testing.assert_close(actual, (expected_k, expected_v, expected_beta), rtol=3e-4, atol=3e-5)
    expected_z = beta[..., None] * (key.float() @ d_p.bfloat16().float()
        + key.float() @ d_p.transpose(-1, -2).bfloat16().float() + d_b[..., None, :])
    expected_z_beta = ((key.float() @ d_p.bfloat16().float()) * key.float()).sum(-1)
    expected_z_beta += (key.float() * d_b[..., None, :]).sum(-1)
    torch.testing.assert_close(native.phase_a_z_bwd(key, beta, d_b, d_p, 112),
                               (expected_z, expected_z_beta), rtol=3e-4, atol=3e-5)
