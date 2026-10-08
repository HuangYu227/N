"""Original camera operator/gradients and native cache-window regression."""
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from test_alignment_detail import cached_sana
from test_anchor import context
from test_training import TinyWorldModel
from worldttn.anchor import TTNAnchor, configure_camera_attention
from worldttn.checkpoint import read_checkpoint, save_checkpoint
from worldttn.core import TTNConfig, ANCHORS
from worldttn.session import carry_cache


def native_camera_source(cached_sana):
    source = cached_sana().double()
    path = Path(__file__).resolve().parents[2] / "diffusion/model/nets/sana_gdn_blocks.py"
    namespace = {"torch": torch, "F": F, "_SDPA_D112_DIRECT": True}
    nodes = [n for n in ast.parse(path.read_text(encoding="utf-8")).body
             if isinstance(n, ast.FunctionDef) and n.name in ("_sdpa_needs_head_pad", "_sdpa_maybe_chunk_causal")]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    # Execute the actual SANA SDPA helper, rather than the fixture's stand-in.
    source._cached_cam_branch_softmax.__func__.__globals__["_sdpa_maybe_chunk_causal"] = namespace["_sdpa_maybe_chunk_causal"]
    return source


@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("meta", [False, True, "proximal"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_restored_camera_matches_original_output_cache_and_gradients(cached_sana, save, meta, device):
    torch.manual_seed(7)
    source = native_camera_source(cached_sana).to(device)
    proximal = meta == "proximal"
    cfg = TTNConfig(heads=2, head_dim=8, generators=3, camera_attention="sana",
                    stage="C" if meta else "A", local_update=meta is True, persistent_meta=meta is True,
                    persistent_update=not proximal, memory_update="proximal" if proximal else "delta")
    anchor = TTNAnchor(source, 0, cfg)
    runtime, ctx = context(cfg, f=4, device=device)
    state, psi = runtime.world_state.clone(), runtime.transition_fast.clone()
    x = torch.randn(1, 4, 16, dtype=torch.float64, device=device, requires_grad=True)
    camera = torch.ones(1, 4, 20, dtype=torch.float64, device=device)
    fns = (lambda z: z * 2, lambda z: z * .5, lambda z: z * 3)
    incoming = [None] * 10
    incoming[2] = torch.randn(1, 2, 3, 8, dtype=torch.float64, device=device)
    incoming[3] = torch.randn_like(incoming[2])
    incoming[9] = torch.randn(1)
    before = [v.clone() if isinstance(v, torch.Tensor) else v for v in incoming]
    expected_cache = list(incoming)
    expected = source._cached_cam_branch_softmax(x, (4, 1, 1), camera, None, expected_cache,
                                                save, prope_fns=fns, chunk_size=3)
    parameters = [x, *source.q_proj_cam.parameters(), *source.k_proj_cam.parameters(),
                  *source.v_proj_cam.parameters(), *source.q_norm_cam.parameters(), *source.k_norm_cam.parameters()]
    expected_grad = torch.autograd.grad(expected.square().sum(), parameters)
    captures = {}
    out, cache = anchor(x, HW=(4, 1, 1), camera_conditions=camera, chunk_size=3,
                        ttn_chunk_context=ctx, kv_cache=incoming, save_kv_cache=save,
                        prope_fns=fns, ttn_diagnostic=lambda name, value: captures.update({name: value}))
    actual = captures["camera_raw"]
    actual_grad = torch.autograd.grad(actual.square().sum(), parameters)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    for index in (2, 3, 9):
        torch.testing.assert_close(cache[index], expected_cache[index], rtol=0, atol=0)
    assert cache[0] is cache[1] is None and cache[6].item() == 0
    for a, b in zip(incoming, before):
        if isinstance(a, torch.Tensor): torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(state, runtime.world_state) and torch.equal(psi, runtime.transition_fast)
    assert out.shape == x.shape


def test_clean_history_keeps_camera_tokens_once_and_preserves_other_caches():
    previous = [[None] * 10 for _ in range(20)]
    current = [[None] * 10 for _ in range(20)]
    current[0][0] = torch.randn(1, 2, 8, 8)
    for i in ANCHORS:
        previous[i][2] = torch.randn(1, 2, 4, 8)
        previous[i][3] = torch.randn(1, 2, 4, 8)
        current[i][2] = torch.randn(1, 2, 3, 8)
        current[i][3] = torch.randn(1, 2, 3, 8)
        current[i][6] = torch.zeros(1)
        current[i][9] = torch.randn(1)
    history = carry_cache(current, "sana", previous)
    detached_boundary = carry_cache(history, "sana")
    for i in ANCHORS:
        for slot in (2, 3):
            torch.testing.assert_close(history[i][slot], torch.cat((previous[i][slot], current[i][slot]), dim=2))
            assert detached_boundary[i][slot] is history[i][slot]
        assert history[i][0] is history[i][1] is None and history[i][9] is current[i][9]
    assert history[0][0] is current[0][0]
    assert carry_cache(current)[3][:9] == [None] * 9  # Legacy linear-camera behavior.


@pytest.mark.parametrize("sink,expected_indices", [(False, [1, 2]), (True, [0, 2])])
def test_actual_sampler_accumulates_native_camera_window_and_sink(sink, expected_indices):
    path = Path(__file__).resolve().parents[2] / "diffusion/scheduler/self_forcing_flow_euler_sampler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_NUM_CACHE_SLOTS", "_SLOT_K", "_SLOT_V", "_SLOT_BETA", "_SLOT_DECAY", "_SLOT_SHORTCONV",
             "_SLOT_TCONV", "_SLOT_TYPE_FLAG", "_CONCAT_SLOTS", "_SOFTMAX_CONCAT_SLOTS", "_LAST_CHUNK_SLOTS"}
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    nodes += [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_accumulate_softmax_kv_cache"]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    sampler = SimpleNamespace(num_cached_blocks=2, sink_token=sink, num_model_blocks=20,
                              _chunk_indices=[0, 4, 7, 10, 13],
                              model=SimpleNamespace(ttn_reference=True, ttn_system=SimpleNamespace(config=TTNConfig(camera_attention="sana"))))
    chunks = [[[None] * 10 for _ in range(20)] for _ in range(4)]
    for index in range(3):
        for block in range(20):
            slots = chunks[index][block]
            slots[6] = torch.zeros(1)
            slots[9] = torch.tensor([index])
            if block not in ANCHORS: slots[0] = slots[1] = torch.full((1, 2, 3, 8), float(index))
            slots[2] = slots[3] = torch.full((1, 2, 3, 8), float(index + 10))
    result, *_ = namespace["_accumulate_softmax_kv_cache"](sampler, chunks, 3)
    for block in ANCHORS:
        assert result[block][0] is result[block][1] is None
        expected = torch.cat([chunks[i][block][2] for i in expected_indices], dim=2)
        torch.testing.assert_close(result[block][2], expected)
        torch.testing.assert_close(result[block][3], expected)
        assert result[block][9] is chunks[2][block][9]


def test_legacy_checkpoints_retain_camera_identity_and_reject_silent_migration(tmp_path):
    model = TinyWorldModel()
    file = tmp_path / "legacy.pt"
    save_checkpoint(file, model, None, 156)
    payload = torch.load(file, weights_only=False)
    payload["config"].pop("camera_attention")
    torch.save(payload, file)
    assert read_checkpoint(file, model)["step"] == 156
    model.ttn_system.config = replace(model.ttn_system.config, camera_attention="sana")
    with pytest.raises(ValueError, match="architecture"):
        read_checkpoint(file, model)


def test_camera_ablation_changes_config_without_changing_parameters_or_visual_attention(cached_sana):
    from torch import nn
    from worldttn.anchor import install_ttn
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Module() for _ in range(20)])
    for block in model.blocks: block.attn = native_camera_source(cached_sana)
    install_ttn(model, TTNConfig(heads=2, head_dim=8, generators=3, stage="C"))
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    configure_camera_attention(model, "sana")
    assert model.ttn_system.config.camera_attention == "sana"
    for i in ANCHORS: assert model.blocks[i].attn.config == model.ttn_system.config
    for name, tensor in model.state_dict().items(): torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
