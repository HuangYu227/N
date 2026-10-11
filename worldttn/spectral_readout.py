"""Inference-only spatial reference probe; the frame-wise state writer is unchanged."""
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .core import ANCHORS
from .sink import freeze_reference, realign_reference, reference_sha256


MODES = ("baseline", "low-zero", "low-reference", "full-reference")


def spatial_response(hw, sigma, device, dtype):
    t, h, w = hw
    if any(type(n) is not int or n < 1 for n in hw) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("invalid spatial grid or Gaussian bandwidth")
    y = torch.fft.fftfreq(h, device=device, dtype=dtype) / .5
    x = torch.fft.fftfreq(w, device=device, dtype=dtype) / .5
    radius2 = y[:, None].square() + x[None, :].square()
    return torch.exp(-radius2 / (2 * sigma**2)), radius2


def spatial_lowpass(value, hw, sigma=.125):
    if value.ndim != 3 or value.shape[1] != math.prod(hw) or value.dtype not in (torch.float32, torch.float64):
        raise ValueError("spatial filter requires FP32/64 [B,T*H*W,C]")
    response, _ = spatial_response(hw, sigma, value.device, value.dtype)
    grid = value.reshape(value.shape[0], *hw, value.shape[-1])
    spectrum = torch.fft.fft2(grid, dim=(2, 3), norm="ortho")
    return torch.fft.ifft2(spectrum * response[None, None, :, :, None],
                         dim=(2, 3), norm="ortho").real.reshape_as(value)


@torch.no_grad()
def spatial_stats(value, hw, *, channel_means=False):
    grid = value.float().reshape(value.shape[0], *hw, value.shape[-1])
    spectrum = torch.fft.fft2(grid, dim=(2, 3), norm="ortho")
    energy = spectrum.real.square()+spectrum.imag.square()
    _, radius2 = spatial_response(hw, .125, value.device, grid.dtype)
    bands = (radius2 == 0, (radius2 > 0) & (radius2 <= .125**2), radius2 > .125**2)
    result = {name: (energy * mask[None, None, :, :, None]).mean((2, 3, 4)).cpu().tolist()
              for name, mask in zip(("dc_energy", "low_non_dc_energy", "higher_energy"), bands)}
    result["signed_mean"] = grid.mean((2, 3, 4)).cpu().tolist()
    result["rms"] = grid.square().mean((2, 3, 4)).sqrt().cpu().tolist()
    if channel_means:
        result["signed_channel_means"] = grid.mean((2, 3)).cpu().tolist()
    return result


class SpectralReadout:
    def __init__(self, mode, *, gain=.25, sigma=.125, noise_min=.8, diagnostics=None):
        if (mode not in MODES or isinstance(gain, bool) or not math.isfinite(gain) or not 0 <= gain <= 1
                or not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(noise_min) or not 0 <= noise_min <= 1):
            raise ValueError("invalid spectral readout protocol")
        self.mode, self.gain, self.sigma, self.noise_min = mode, gain, sigma, noise_min
        self.path = Path(diagnostics) if diagnostics else None
        self.reset()

    def reset(self):
        self.reference = self.reference_hash = self.context = self.aligned = None
        self.calls, self.pending = [], {}
        self.nonzero_anchors, self.measured_rows = set(), 0
        self.active = self.measure = False

    def capture(self, runtime):
        if not self.gain:
            return
        if torch.is_grad_enabled() or not runtime.prefilled or runtime.commit_count != 1 or self.reference is not None:
            raise ValueError("spectral reference requires exactly one completed observed prefill in inference")
        state = runtime.world_state
        if state.ndim != 5 or state.shape[1] != len(ANCHORS) or state.dtype != torch.float32 or not torch.isfinite(state).all():
            raise ValueError("invalid observed reference state")
        # Publish only after the complete clone and its audit have succeeded.
        reference = freeze_reference(state)
        digest = reference_sha256(reference)
        self.reference, self.reference_hash = reference, digest

    def begin_call(self, context, noise_sigma, call, total_calls):
        if torch.is_grad_enabled() or context.clean_mode or context.prefill_mode or self.reference is None:
            raise ValueError("spectral noisy call requires a captured inference reference")
        sigma = noise_sigma.detach().float().reshape_as(context.frame_ids)
        values = sigma.cpu()
        if not torch.isfinite(values).all() or (values < 0).any() or (values > 1.00001).any():
            raise ValueError("invalid actual scheduler sigma")
        if self.context is not context:
            self.shift = (context.frame_ids.min(-1).values - 1).clamp_min(0)
            self.aligned = realign_reference(self.reference, context.memory_rope, self.shift)
            self.context = context
        self.frame_mask = (sigma >= self.noise_min) & (context.frame_ids > 0) & context.read_mask
        self.active = self.mode != "baseline" and bool(self.frame_mask.any())
        self.measure = call in {0, total_calls//2, total_calls-1}
        self.call = call
        self.calls.append({"frame_ids": context.frame_ids.cpu().tolist(), "call": call,
                           "sigma_source": "scheduler", "sigma": values.tolist(),
                           "active": self.active, "eligible_frames": self.frame_mask.cpu().tolist()})

    @torch.no_grad()
    def apply(self, index, context, q, visual_fast, gate, gated, hw):
        source = getattr(context, "spectral_source_context", context)
        if source.clean_mode or source.prefill_mode or context.prefill_mode or not self.gain:
            return gated
        if self.context is not source:
            raise ValueError("spectral readout missing actual sampler call metadata")
        if not self.active and not self.measure:
            return gated
        frame_mask, active = self.frame_mask, self.active
        if source is not context:
            positions = context.frame_ids-self.context.frame_ids[:, :1]
            if (positions < 0).any() or (positions >= self.frame_mask.shape[1]).any():
                raise ValueError("private frame range is outside the actual noisy sampler context")
            if not torch.equal(source.frame_ids.gather(1, positions), context.frame_ids):
                raise ValueError("private frame IDs differ from the actual noisy sampler context")
            frame_mask = self.frame_mask.gather(1, positions)
            active = self.mode != "baseline" and bool(frame_mask.any())
        with torch.autocast(device_type=q.device.type, enabled=False):
            reference = (q.float() @ self.aligned[:, index]).transpose(1, 2).reshape_as(visual_fast)
        fast, anchor = visual_fast.float()*gate.float(), reference*gate.float()
        corrected = gated
        if active:
            delta = -fast if self.mode == "low-zero" else anchor-fast
            if self.mode != "full-reference":
                delta = spatial_lowpass(delta, hw, self.sigma)
            mask = frame_mask[:, :, None].expand(-1, -1, hw[1]*hw[2]).reshape(*gated.shape[:2], 1)
            corrected = gated + self.gain*torch.where(mask, delta, 0.)
        if self.measure:
            self.pending[index] = (gated, {"anchor": ANCHORS[index], "call": self.call,
                "frame_ids": context.frame_ids.cpu().tolist(), "active": active,
                "private_candidate": source is not context,
                "virtual_reference_position": self.shift.cpu().tolist(),
                "q_rms": q.float().square().mean((-1, -2)).sqrt().cpu().tolist(),
                "gate_rms": gate.float().square().mean((1, 2)).sqrt().cpu().tolist(),
                "fast_gated": spatial_stats(fast, hw, channel_means=self.call == 0),
                "reference_gated": spatial_stats(anchor, hw, channel_means=self.call == 0)})
        return corrected

    @torch.no_grad()
    def finish(self, index, context, out, proj, dtype, read):
        source = getattr(context, "spectral_source_context", context)
        if source.clean_mode or source.prefill_mode or context.prefill_mode or index not in self.pending:
            return
        prior, row = self.pending.pop(index)
        before = F.linear(prior.to(dtype), proj.weight, proj.bias)*read[..., None].to(dtype)
        delta = out.float()-before.float()
        norm = delta.norm()
        row["output_effect"] = {"dtype": str(out.dtype), "delta_norm": norm.item(),
            "relative_delta": (norm/before.float().norm().clamp_min(1e-12)).item(),
            "changed_fraction": (delta != 0).float().mean().item(),
            "measurement": "paired actual quantized outputs, same Q/gate/camera; no quality claim"}
        if row["active"] and norm > 0:
            self.nonzero_anchors.add(ANCHORS[index])
        if self.path:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
        self.measured_rows += 1

    def verify(self):
        if self.reference is None or reference_sha256(self.reference) != self.reference_hash or self.pending:
            raise AssertionError("spectral reference SHA or diagnostic lifecycle failed")
        if self.mode != "baseline" and self.gain and self.nonzero_anchors != set(ANCHORS):
            raise AssertionError("spectral correction did not reach every anchor output")
        return {"mode": self.mode, "gain": self.gain, "bandwidth_nyquist": self.sigma,
                "noise_min": self.noise_min, "position": "chunk_start-1 temporal realignment",
                "reference_source": "clean true first-frame prefill", "reference_sha256": self.reference_hash,
                "reference_verified": True, "reference_bytes": self.reference.numel()*self.reference.element_size(),
                "nonzero_quantized_anchors": sorted(self.nonzero_anchors), "diagnostic_rows": self.measured_rows,
                "calls": self.calls}


def install_spectral_readout(model, controller):
    if (torch.is_grad_enabled() or model.training or model.ttn_system.config.memory_granularity != "frame"
            or model.ttn_system.config.memory_update != "proximal"):
        raise ValueError("spectral probe requires an eval frame-proximal model under no_grad")
    if not controller.gain:
        controller = None
    model._ttn_spectral_readout = controller
    for block in ANCHORS:
        model.blocks[block].attn._ttn_spectral_readout = controller


@torch.no_grad()
def latent_diagnostics(latent):
    value = latent.float().permute(0, 2, 3, 4, 1)
    drift = (value-value[:, :1]).square().mean((0, 2, 3, 4))
    change = (value[:, 1:]-value[:, :-1]).square().mean((0, 2, 3, 4))
    hw = tuple(value.shape[1:4])
    return {"reference": "observed latent frame0; drift is not future-GT quality",
            "per_frame_drift": drift.cpu().tolist(), "mean_future_drift": drift[1:].mean().item(),
            "final_drift": drift[-1].item(), "adjacent_frame_mse": change.cpu().tolist(),
            "spatial": spatial_stats(value.reshape(value.shape[0], -1, value.shape[-1]), hw)}
