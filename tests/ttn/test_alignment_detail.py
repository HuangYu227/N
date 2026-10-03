"""Real cached SANA callbacks, native drift, update references and read-only gradients."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from test_anchor import ProjectionContract
from test_alignment import models_and_batch
from worldttn.alignment import probe_chunk, replay_anchors
from worldttn.alignment_detail import diagnostic_summary, parameter_changes, parameter_fingerprint, gradient_health
from worldttn.geometry import apply_complex_rope
from worldttn.session import TTNSession
from worldttn.training import linear_flow_loss


@pytest.fixture
def cached_sana():
    root = Path(__file__).resolve().parents[2]
    class Base(ProjectionContract):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.cam_heads, self.cam_dim = self.heads, self.heads * self.dim
            self.conv_q_cam = self.conv_k_cam = self.conv_v_cam = None
        def _apply_output_gate(self, value, x): return value * F.silu(self.output_gate(x))
    registry = SimpleNamespace(register_module=lambda: lambda cls: cls)
    namespace = {"torch": torch, "F": F, "ChunkCausalSoftmaxAttn": Base,
                 "_SoftmaxUCPESinglePathLiteLA": Base, "ATTENTION_BLOCKS": registry,
                 "GDN": SimpleNamespace(_apply_rotary_emb=lambda z, rope: apply_complex_rope(z.transpose(-1, -2), rope).transpose(-1, -2)),
                 "_slice_rope_to_current_chunk": lambda rope, n: rope[..., -n:, :],
                 "_sdpa_maybe_chunk_causal": lambda q, k, v, **kwargs: F.scaled_dot_product_attention(q, k, v)}
    names = {"_SLOT_FWD_KV", "_SLOT_FWD_Z", "_SLOT_TYPE_FLAG", "_SLOT_CAM", "_SLOT_CAM_AUX", "_TYPE_CONCAT"}
    constants = root / "diffusion/model/ops/fused_streaming.py"
    nodes = [n for n in ast.parse(constants.read_text(encoding="utf-8")).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(constants), "exec"), namespace)
    for file, names in (("sana_gdn_blocks.py", {"CachedChunkCausalSoftmaxAttn"}),
                        ("sana_gdn_camctrl_blocks.py", {"_prepare_cam_qkv_softmax", "CachedSoftmaxUCPESinglePathLiteLA"})):
        path = root / "diffusion/model/nets" / file
        nodes = [n for n in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["CachedSoftmaxUCPESinglePathLiteLA"]


@pytest.mark.parametrize("save", [False, True])
def test_actual_cached_visual_camera_observers_preserve_outputs_and_gradients(cached_sana, save):
    torch.manual_seed(6)
    module = cached_sana()
    x = torch.randn(1, 4, 16, requires_grad=True)
    rope = torch.polar(torch.ones(1, 1, 4, 4, dtype=torch.float64), torch.randn(1, 1, 4, 4, dtype=torch.float64))
    fns = (lambda z: z * 2, lambda z: z * .5, lambda z: z * 3)
    kwargs = dict(HW=(4, 1, 1), rotary_emb=rope, camera_conditions=torch.ones(1, 4, 20),
                  prope_fns=fns, save_kv_cache=save)
    expected, expected_cache = module(x, kv_cache=[None] * 10, **kwargs)
    parameters = list(module.parameters()) + [x]
    baseline_gradients = torch.autograd.grad(expected.square().sum(), parameters, allow_unused=True)
    captured = {}
    actual, actual_cache = module(x, kv_cache=[None] * 10, ttn_diagnostic=lambda name, value: captured.update({name: value}), **kwargs)
    gradients = torch.autograd.grad(actual.square().sum(), parameters, allow_unused=True)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    for a, b in zip(expected_cache, actual_cache):
        if isinstance(a, torch.Tensor): torch.testing.assert_close(a, b, rtol=0, atol=0)
        else: assert a == b
    for a, b in zip(baseline_gradients, gradients):
        if a is None: assert b is None
        else: torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert captured["visual_raw"].shape == captured["camera_raw"].shape == (1, 4, 16)
    fused = captured["visual_raw"] + module.out_proj_cam(captured["camera_raw"])
    torch.testing.assert_close(fused, captured["fused_raw"], rtol=0, atol=0)
    torch.testing.assert_close(fused * F.silu(module.output_gate(x)), captured["gated_raw"], rtol=0, atol=0)
    torch.testing.assert_close(module.proj(captured["gated_raw"]), actual, rtol=0, atol=0)


def test_detailed_probe_reports_correct_descent_and_native_accumulation():
    teacher, student, batch = models_and_batch()
    result = probe_chunk(teacher, student, batch, linear_flow_loss, [500], torch.randn_like(batch["clean_latents"]))
    probe = result["probes"][0]
    for anchor in probe["anchors"]:
        s = anchor["stages"]
        assert s["input"]["exact_equal"] and s["input"]["relative_l2"] == 0
        assert all(v["relative_l2"] == 0 for v in s["projected_qkv"].values())
        assert not s["innovation"]["weighted_loss_increased"]
        assert s["write"]["write_token_count"] == 3
        assert s["write"]["beta"]["mean"] == .5 and s["write"]["beta"]["std"] == 0
        assert s["camera_raw"] is not None and s["camera_contribution"] is not None
        assert len(s["memory"]["S_pred_norm_head"][0]) == 2
    drift = probe["representation_drift"]
    assert [(r["block"], r["point"]) for r in drift] == [(i, p) for i in (3, 7, 11, 15, 19) for p in ("before", "after")]
    assert drift[0]["relative_l2"] == 0 and drift[0]["cosine"] == pytest.approx(1)
    assert drift[1]["relative_l2"] > 0 and drift[-1]["relative_l2"] > 0
    assert probe["native_flow"]["relative_l2"] > 0
    assert result["read_only_verified"]["parameters_unchanged"]
    assert result["diagnostic_summary"]["first_contract_violation"] is None


def test_grad_probe_has_finite_connected_gradients_and_preserves_existing_grads_weights_states(monkeypatch):
    teacher, student, batch = models_and_batch()
    old = {}
    for name, p in student.named_parameters():
        if name.endswith("qkv.weight"):
            p.grad = torch.ones_like(p)
            old[name] = p.grad.clone()
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **kw: pytest.fail("diagnostic called optimizer.step"))
    result = probe_chunk(teacher, student, batch, linear_flow_loss, [500], torch.randn_like(batch["clean_latents"]), gradient_timestep=500)
    health = result["gradient_health"]
    assert health["optimizer_step"] is False and health["param_grad_written"] is False
    for row in health["anchors"]:
        for g in row["groups"].values():
            assert g["finite"] and g["unused_parameters"] == 0 and g["grad_norm"] > 0
    for name, p in student.named_parameters():
        assert not p.requires_grad
        if name in old: torch.testing.assert_close(p.grad, old[name], rtol=0, atol=0)
        else: assert p.grad is None
    assert result["states"]["native_prefill"]["committed_frames"] == [0]
    assert result["read_only_verified"]["persistent_S_psi_pose_unchanged_after_private_prefill"]


def test_parameter_change_reference_is_base_projection_and_beta_zero_not_teacher_unused_beta():
    teacher, student, _ = models_and_batch()
    result = parameter_changes(teacher, student)
    assert all(row["groups"]["qkv"]["delta_norm"] == 0 for row in result)
    assert all(row["groups"]["beta_proj"]["zero_initial_norm"] for row in result)
    with torch.no_grad():
        student.blocks[3].attn.qkv.weight.add_(.1)
        student.blocks[3].attn.beta_proj.bias.fill_(.2)
    result = parameter_changes(teacher, student)[0]["groups"]
    assert result["qkv"]["update_ratio"] > 0
    assert result["beta_proj"]["delta_norm"] == pytest.approx(.2 * 2**.5)
    assert result["beta_proj"]["relative_update_ratio"] is None
    assert result["beta_proj"]["update_ratio"] > 100000  # epsilon ratio is NOT a percent


def test_summary_checks_contracts_first_but_does_not_infer_failure_from_large_correction_ratio():
    teacher, student, batch = models_and_batch()
    rows = probe_chunk(teacher, student, batch, linear_flow_loss, [500], torch.randn_like(batch["clean_latents"]))["probes"]
    rows[0]["anchors"][0]["stages"]["memory"]["correction_ratio"] = 1e8
    assert diagnostic_summary(rows)["first_contract_violation"] is None
    rows[0]["anchors"][1]["stages"]["innovation"]["weighted_loss_increased"] = True
    assert diagnostic_summary(rows)["first_contract_violation"] == {"block": 7, "stage": "Correct_weighted_loss_increased", "timestep": 500}
    rows[0]["anchors"][2]["stages"]["input"]["exact_equal"] = False
    assert diagnostic_summary(rows)["first_contract_violation"]["stage"] == "input_mismatch"
    assert "不判定结构失败" in diagnostic_summary(rows)["text"]


def test_failure_cleans_all_detail_and_drift_hooks():
    teacher, student, batch = models_and_batch()
    def failed_loss(*args, **kw): raise RuntimeError("flow probe failed")
    with pytest.raises(RuntimeError, match="flow probe failed"):
        probe_chunk(teacher, student, batch, failed_loss, [500], torch.randn_like(batch["clean_latents"]))
    for model in (teacher, student):
        for module in model.modules():
            assert not module._forward_hooks and not module._forward_pre_hooks


def test_empty_write_statistics_are_defined_and_do_not_flag_large_formula_eta():
    teacher, student, batch = models_and_batch()
    session = TTNSession(student, batch["camera_conditions"], 1, 1)
    session.reset(1)
    context = session.begin_chunk(0, 4)
    context.write_mask.zero_()
    rows = []
    with replay_anchors(teacher, student, context, rows):
        teacher(batch["clean_latents"], torch.zeros(1), batch["y"], ttn_chunk_context=context,
                kv_cache=[[None] * 10 for _ in range(20)], camera_conditions=batch["camera_conditions"])
    for row in rows:
        stages = row["stages"]
        assert stages["write"]["write_token_count"] == 0
        assert stages["write"]["beta"]["mean"] is None
        assert stages["innovation"]["rms"] is None and not stages["innovation"]["weighted_loss_increased"]
        assert stages["memory"]["correction_norm"] == 0
        assert stages["write"]["effective_eta_s_head"] == [[0., 0.]]


def test_hash_handles_scalar_parameters_and_detects_even_a_single_changed_value():
    module = nn.Module()
    module.scalar = nn.Parameter(torch.tensor(1.))
    before = parameter_fingerprint(module)
    assert parameter_fingerprint(module) == before
    with torch.no_grad(): module.scalar.add_(1)
    assert parameter_fingerprint(module) != before


def test_failed_gradient_probe_restores_flags_and_grad_objects():
    _, student, batch = models_and_batch()
    session = TTNSession(student, batch["camera_conditions"], 1, 1)
    session.reset(1)
    ctx = session.begin_chunk(0, 4)
    flags = [p.requires_grad for p in student.parameters()]
    def fail(*args, **kwargs): raise RuntimeError("backward setup failed")
    with pytest.raises(RuntimeError, match="backward setup failed"):
        gradient_health(session, ctx, batch["clean_latents"], batch["y"], None, None,
                        torch.randn_like(batch["clean_latents"]), fail, torch.ones(1, 1, 4, 1, 1), 500)
    assert [p.requires_grad for p in student.parameters()] == flags
    assert all(p.grad is None for p in student.parameters())
