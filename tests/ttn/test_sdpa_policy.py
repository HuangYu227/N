"""Exercise the actual SANA text module without importing optional CUDA packages."""
import ast
from contextlib import nullcontext
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Optional

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from worldttn.training import activation_storage


@pytest.fixture
def text_attention():
    source = Path(__file__).resolve().parents[2] / "diffusion/model/nets/sana_blocks.py"
    node = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
                if isinstance(node, ast.ClassDef) and node.name == "MultiHeadCrossAttention")
    namespace = {"nn": nn, "F": F, "Optional": Optional, "_xformers_available": False,
                 "nullcontext": nullcontext, "SDPBackend": SDPBackend, "sdpa_kernel": sdpa_kernel}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["MultiHeadCrossAttention"]


def backend_flags():
    return {name: getattr(torch.backends.cuda, name + "_sdp_enabled")()
            for name in ("math", "flash", "mem_efficient", "cudnn")}


@pytest.mark.parametrize("dtype,heads,dim", [(torch.float64, 3, 8), (torch.bfloat16, 20, 112)])
@pytest.mark.parametrize("offload", ["none", "cpu"])
def test_local_math_preserves_forward_gradients_mask_and_strided_layout(text_attention, monkeypatch, dtype, heads, dim, offload):
    torch.manual_seed(3407)
    module = text_attention(heads * dim, heads).to(dtype)
    x = torch.randn(1, 7, heads * dim, dtype=dtype, requires_grad=True)
    cond = torch.randn(1, 1, 5, heads * dim, dtype=dtype, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    probe = torch.randn_like(x)
    original = F.scaled_dot_product_attention
    calls, gradients = [], []
    def capture(q, k, v, **kwargs):
        calls.append(([t.stride() for t in (q, k, v)], kwargs, backend_flags()))
        output = original(q, k, v, **kwargs)
        output.register_hook(lambda grad: gradients.append((grad.stride(), grad.is_contiguous())))
        return output
    monkeypatch.setitem(module.forward.__globals__, "F", SimpleNamespace(scaled_dot_product_attention=capture))
    flags = backend_flags()
    state_keys = set(module.state_dict())
    results = []
    for policy in ("auto", "math"):
        module.zero_grad(set_to_none=True)
        x.grad = cond.grad = None
        module.set_sdpa_backend(policy)
        # Reference: unmodified native call under an outer math context. The
        # candidate restricts only the module's call; backward runs outside it.
        with sdpa_kernel(SDPBackend.MATH) if policy == "auto" else nullcontext():
            with activation_storage(offload):
                output = module(x, cond, mask)
        assert backend_flags() == flags
        (output * probe).float().sum().backward()
        results.append([output.detach(), x.grad.clone(), cond.grad.clone(),
                        *[p.grad.clone() for p in module.parameters()]])
    for reference, actual in zip(*results):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert torch.isfinite(actual).all()
    assert state_keys == set(module.state_dict())
    assert gradients[0] == gradients[1] and not gradients[1][1]
    assert gradients[1][0][1:] == (dim, heads * dim, 1)
    assert calls[0][0] == calls[1][0]  # No added contiguous copies of Q/K/V.
    for _, kwargs, enabled in calls:
        assert enabled == {"math": True, "flash": False, "mem_efficient": False, "cudnn": False}
        assert kwargs["dropout_p"] == 0. and kwargs["is_causal"] is False
        torch.testing.assert_close(kwargs["attn_mask"], ((1 - mask.to(dtype)) * -10000.)[:, None, None].repeat(1, heads, 1, 1))


@pytest.mark.parametrize("policy,enabled", [("auto", None), ("math", "math"), ("flash", "flash"), ("efficient", "mem_efficient")])
@pytest.mark.parametrize("fail", [False, True])
def test_backend_is_strict_scoped_and_restored_on_error(text_attention, monkeypatch, policy, enabled, fail):
    module = text_attention(24, 3)
    module.set_sdpa_backend(policy)
    seen = []
    def probe(q, k, v, **kwargs):
        seen.append(backend_flags())
        if fail: raise RuntimeError("unsupported backend probe")
        return q  # Observe dispatch policy, never simulate a fused CUDA kernel.
    monkeypatch.setitem(module.forward.__globals__, "F", SimpleNamespace(scaled_dot_product_attention=probe))
    # A non-default enclosing policy must be restored, including on failure.
    with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
        flags = backend_flags()
        if fail:
            with pytest.raises(RuntimeError, match="unsupported backend probe"):
                module(torch.randn(1, 7, 24), torch.randn(1, 1, 5, 24))
        else:
            module(torch.randn(1, 7, 24), torch.randn(1, 1, 5, 24))
        assert backend_flags() == flags
    assert seen == [flags if enabled is None else {key: key == enabled for key in flags}]


def test_invalid_or_xformers_policy_is_rejected(text_attention, monkeypatch):
    module = text_attention(24, 3)
    with pytest.raises(ValueError, match="unknown"): module.set_sdpa_backend("typo")
    monkeypatch.setitem(module.forward.__globals__, "_xformers_available", True)
    module.set_use_xformers(True)
    with pytest.raises(ValueError, match="xFormers disabled"): module.set_sdpa_backend("math")
    module.set_use_xformers(False)
    module.set_sdpa_backend("math")
    with pytest.raises(ValueError, match="reset"): module.set_use_xformers(True)


def test_builder_selection_excludes_other_attention_and_checkpoint_state(text_attention, monkeypatch):
    from worldttn.sana import configure_cross_attention
    source = ModuleType("diffusion.model.nets.sana_blocks")
    source.MultiHeadCrossAttention = text_attention
    monkeypatch.setitem(sys.modules, source.__name__, source)
    class OtherAttention(text_attention):
        pass
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Module() for _ in range(20)])
    for block in model.blocks:
        block.cross_attn = text_attention(24, 3)
        block.attn = text_attention(24, 3)  # Same class, deliberately outside text call scope.
    model.image = nn.Module()
    model.image.cross_attn = OtherAttention(24, 3)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    report = configure_cross_attention(model, "math")
    assert report["modules"] == [f"blocks.{i}.cross_attn" for i in range(20)]
    assert report["backend"] == "math"
    assert all(block.cross_attn.sdpa_backend == SDPBackend.MATH and block.attn.sdpa_backend is None
               for block in model.blocks)
    assert model.image.cross_attn.sdpa_backend is None
    after = model.state_dict()
    assert before.keys() == after.keys() and all(torch.equal(value, after[name]) for name, value in before.items())
    with pytest.raises(ValueError, match="no standard SANA"):
        configure_cross_attention(nn.Identity(), "math")


@pytest.mark.parametrize("policy", ["math", "flash"])
def test_cli_build_applies_requested_policy_and_logs_scope(text_attention, monkeypatch, capsys, policy):
    from worldttn import cli, sana
    model = nn.Module()
    model.block = nn.Module()
    model.block.cross_attn = text_attention(24, 3)
    source = ModuleType("diffusion.model.nets.sana_blocks")
    source.MultiHeadCrossAttention = text_attention
    monkeypatch.setitem(sys.modules, source.__name__, source)
    monkeypatch.setattr(cli, "read_reference", lambda *args: (None, {"sana_config": "unused"}))
    monkeypatch.setattr(sana, "load_sana_config", lambda *args: None)
    monkeypatch.setattr(sana, "build_sana", lambda *args, **kwargs: model)
    args = SimpleNamespace(config="unused", sana_config=None, stage="A", base_weights=None,
                           device="cpu", cross_attn_backend=policy)
    result, _, _ = cli.build(args)
    assert result.cross_attention_report["backend"] == policy
    assert model.block.cross_attn.sdpa_backend == {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION}[policy]
    assert '"modules": ["block.cross_attn"]' in capsys.readouterr().out
