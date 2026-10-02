"""Audit the existing SANA flow endpoint without importing CUDA model packages."""
import ast
import enum
import math
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Optional, Tuple, Union
import warnings

import numpy as np
import torch


def test_linear_flow_zero_cumprod_is_unused_by_flow_loss_and_backward():
    root = Path(__file__).resolve().parents[2] / "diffusion"
    namespace = {"enum": enum, "math": math, "random": random, "np": np, "th": torch,
                 "F": torch.nn.functional, "Optional": Optional, "Tuple": Tuple, "Union": Union}
    # Only omit imports of the heavy diffusion package; execute the real schedule,
    # GaussianDiffusion, SpacedDiffusion, timestep wrapper and training loss.
    for path in (root / "model/gaussian_diffusion.py", root / "model/respace.py", root / "scheduler/iddpm.py"):
        nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
                 if not isinstance(node, (ast.Import, ast.ImportFrom))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        if path.name == "gaussian_diffusion.py": namespace["gd"] = SimpleNamespace(**namespace)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scheduler = namespace["Scheduler"]("1000", noise_schedule="linear_flow", predict_flow_v=True,
                                            learn_sigma=False, pred_sigma=False, snr=False, flow_shift=9.95)
    assert any("divide by zero" in str(item.message) for item in caught)
    assert scheduler.betas[0] == 1. and np.all(scheduler.alphas_cumprod == 0.)
    assert np.isfinite(scheduler.sigmas).all() and scheduler.sigmas[0] == 0.
    assert np.isfinite(scheduler.alphas).all() and scheduler.alphas[0] == 1.
    np.testing.assert_allclose(scheduler.alphas + scheduler.sigmas, 1.)
    for name in ("sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod"):
        assert np.isinf(getattr(scheduler, name)).all()
        # Poison the DDPM inverse tables: any accidental use would spoil the loss.
        setattr(scheduler, name, np.full_like(getattr(scheduler, name), np.nan))
    torch.manual_seed(3407)
    clean = torch.randn(1, 2, 3, 1, 1)
    noise = torch.randn_like(clean)
    t = torch.tensor([[[0, 500, 999]]])
    weight = torch.tensor(.2, requires_grad=True)
    seen = {}
    def model(x, timestep, **kwargs):
        seen.update(x=x, timestep=timestep)
        return weight * x
    with np.errstate(all="raise"):
        result = scheduler.training_losses(model, clean, t, noise=noise)
    sigma = torch.from_numpy(scheduler.sigmas[t.numpy()]).float()[..., None, None]
    torch.testing.assert_close(seen["x"], (1 - sigma) * clean + sigma * noise)
    expected = ((noise - clean) - weight * seen["x"]).square().mean()
    torch.testing.assert_close(result["loss"].mean(), expected)
    assert torch.isfinite(result["loss"]).all() and torch.isfinite(seen["timestep"]).all()
    result["loss"].mean().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad)
