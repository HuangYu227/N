"""Check real SANA wrappers without importing the optional CUDA dependency stack."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from worldttn.geometry import apply_complex_rope, apply_ray_projmat, prepare_ray_apply_fns
from worldttn.training import activation_storage


def sana_helpers(monkeypatch, full, complex_only):
    for name, value in (("GDN_DISABLE_COMPILE", full), ("GDN_DISABLE_COMPLEX_COMPILE", complex_only)):
        if value is None: monkeypatch.delenv(name, raising=False)
        else: monkeypatch.setenv(name, value)
    calls = {}
    def no_backend(*args, **kwargs):
        raise AssertionError("eager helper must not invoke a compiler backend")
    def compile_helper(fn=None, *, disable):
        def wrap(function):
            calls[function.__name__] = disable
            return torch.compile(function, backend=no_backend, disable=True) if disable else function
        return wrap(fn) if fn is not None else wrap
    proxy = SimpleNamespace(Tensor=torch.Tensor, compile=compile_helper)
    root = Path(__file__).resolve().parents[2] / "diffusion/model/nets"
    namespaces = []
    for file, functions, assignments in (
        ("sana_gdn_blocks.py", {"_compute_frame_gates", "_apply_rotary_emb", "_apply_output_gate"}, set()),
        ("sana_camctrl_blocks.py", set(), {"_apply_ray_projmat", "_apply_complex_rope"}),
    ):
        source = root / file
        nodes = []
        for node in ast.parse(source.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign):
                names = {target.id for target in node.targets if isinstance(target, ast.Name)}
                if names & (assignments | {"_COMPILE_DISABLE", "_COMPLEX_COMPILE_DISABLE"}): nodes.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name in functions: nodes.append(node)
        namespace = {"os": os, "torch": proxy, "F": F,
                     "apply_complex_rope": apply_complex_rope, "apply_ray_projmat": apply_ray_projmat}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        # Wrapper construction uses the proxy; the actual Tensor operations use PyTorch.
        proxy.__dict__.update({name: getattr(torch, name) for name in (
            "view_as_complex", "view_as_real", "float32", "float64")})
        namespaces.append(namespace)
    return calls, namespaces


@pytest.mark.parametrize("full,complex_only,all_eager,rope_eager", [
    (None, None, False, False), ("0", "0", False, False), ("false", "false", False, False),
    ("1", "0", True, True), ("1", "1", True, True), ("0", "1", False, True),
])
def test_compile_switches_cover_both_rotary_paths_and_preserve_real_helpers(
        monkeypatch, full, complex_only, all_eager, rope_eager):
    calls, _ = sana_helpers(monkeypatch, full, complex_only)
    assert calls == {"_compute_frame_gates": all_eager, "_apply_rotary_emb": rope_eager,
                     "_apply_output_gate": all_eager, "apply_ray_projmat": all_eager,
                     "apply_complex_rope": rope_eager}


@pytest.mark.parametrize("mode", ["eager", "complex-eager"])
@pytest.mark.parametrize("offload", ["none", "cpu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_eager_rotary_and_camera_gradients_preserve_math_with_noncontiguous_inputs(
        monkeypatch, mode, offload, dtype):
    _, (gdn, camera) = sana_helpers(monkeypatch, "1" if mode == "eager" else "0", "1")
    torch.manual_seed(3407)
    # Include channel slicing, nonzero storage offsets and transposed upstream gradients.
    values = torch.randn(2, 2, 5, 18, dtype=dtype)
    pose = torch.eye(4).expand(2, 5, 4, 4).clone()
    pose[..., :3, 3] = torch.randn(2, 5, 3) * .01
    freqs = torch.polar(torch.ones(2, 1, 5, 2, dtype=torch.float64),
                        torch.randn(2, 1, 5, 2, dtype=torch.float64))
    probe = torch.randn(2, 2, 8, 5, dtype=dtype).transpose(-1, -2)
    def evaluate(use_wrappers, storage):
        source = values.clone().requires_grad_()
        x = source[..., 1:17:2]
        assert x.stride(-1) != 1 and x.storage_offset() != 0
        ray = camera["_apply_ray_projmat"] if use_wrappers else apply_ray_projmat
        rope = camera["_apply_complex_rope"] if use_wrappers else apply_complex_rope
        p = pose.to(dtype)
        q, kv, out = prepare_ray_apply_fns(8, p, p.transpose(-1, -2),
                                         torch.linalg.inv(pose).to(dtype), freqs, ray_apply=ray, rope_apply=rope)
        with activation_storage(storage):
            if use_wrappers:
                # SANA normally casts BF16/FP32 to FP64. For the FP64 oracle,
                # supply its existing unit-channel-stride contract explicitly.
                rotary_input = x.contiguous() if dtype == torch.float64 else x
                rotated = gdn["_apply_rotary_emb"](rotary_input.transpose(-1, -2), freqs.repeat(1, 1, 1, 2))
                rotated = rotated.transpose(-1, -2)
            else:
                rotated = apply_complex_rope(x, freqs.repeat(1, 1, 1, 2))
            result = out(q(rotated) @ (kv(x).transpose(-1, -2) @ kv(x)))
            (result * probe).float().sum().backward()
        return result.detach(), source.grad
    expected, expected_grad = evaluate(False, "none")
    actual, actual_grad = evaluate(True, offload)
    assert torch.isfinite(actual).all() and torch.isfinite(actual_grad).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
