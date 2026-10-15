# Copyright 2026 ByteDance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
# Adapted from ByteDance-Seed/In-Place-TTT, commit be2324829b0e91c8fd10a74d4b43714fde6676e1.
"""Pure chunk-local In-Place TTT algebra; sequence state belongs to the runtime."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class InPlaceFFNTTT(nn.Module):
    def __init__(self, in_features, hidden_features, *, lr=3.0, target_diag_std=.02):
        super().__init__()
        if in_features < 1 or hidden_features < 1:
            raise ValueError('In-Place TTT requires positive FFN dimensions')
        if not math.isfinite(lr) or lr <= 0 or not math.isfinite(target_diag_std) or target_diag_std <= 0:
            raise ValueError('In-Place TTT learning rate and target initialization must be finite and positive')
        self.in_features, self.hidden_features, self.lr = in_features, hidden_features, lr
        self.ttt_conv = nn.Conv1d(in_features, in_features, 5, padding=2,
                                  groups=in_features, bias=False, dtype=torch.float32)
        self.ttt_proj = nn.Linear(in_features, in_features, bias=False, dtype=torch.float32)
        with torch.no_grad():
            self.ttt_conv.weight.zero_()
            self.ttt_proj.weight.zero_()
            self.ttt_proj.weight.diagonal().normal_(std=target_diag_std)

    def forward(self, z, target, delta, write_mask=None, delta_clip=None):
        """Read old delta, then optionally form a candidate using this complete chunk.

        Tokens follow the released author's flattened chunk layout. Symmetric
        convolution stays inside this chunk; its update cannot affect this read.
        """
        if z.ndim != 3 or target.shape != (*z.shape[:2], self.in_features):
            raise ValueError('In-Place TTT expects matching [batch,tokens,features] inputs')
        if z.shape[2] != self.hidden_features or delta.shape != (z.shape[0], self.in_features, self.hidden_features):
            raise ValueError('In-Place TTT fast matrix shape disagrees with the FFN')
        if delta.dtype != torch.float32:
            raise ValueError('In-Place TTT fast state must remain float32')
        if write_mask is not None and (write_mask.shape != z.shape[:2] or write_mask.dtype != torch.bool):
            raise ValueError('In-Place TTT write mask must be boolean [batch,tokens]')
        if delta_clip is not None and (not math.isfinite(delta_clip) or delta_clip <= 0):
            raise ValueError('In-Place TTT delta clip must be finite and positive')
        with torch.autocast(device_type=z.device.type, enabled=False):
            z32 = z.float()
            correction = torch.bmm(z32, delta.transpose(1, 2))
            scale = torch.ones(z.shape[0], device=z.device, dtype=torch.float32)
            raw_norm = scale * 0
            candidate = delta
            if write_mask is not None:
                target32 = target.float() * write_mask[..., None]
                convolved = F.conv1d(target32.transpose(1, 2), self.ttt_conv.weight.float(),
                                     padding=2, groups=self.in_features).transpose(1, 2)
                # Author's contraction is T @ P (not F.linear(T, P), which transposes P).
                values = convolved @ self.ttt_proj.weight.float()
                update = self.lr * torch.bmm(values.transpose(1, 2), z32 * write_mask[..., None])
                raw_norm = torch.linalg.vector_norm(update, dim=(1, 2))
                if delta_clip is not None:
                    scale = (delta_clip / raw_norm.clamp_min(torch.finfo(torch.float32).tiny)).clamp(max=1)
                    update = update * scale[:, None, None]
                candidate = delta + update
        stats = dict(delta_norm_before=torch.linalg.vector_norm(delta.detach(), dim=(1, 2)),
                     delta_norm_after=torch.linalg.vector_norm(candidate.detach(), dim=(1, 2)),
                     raw_update_norm=raw_norm.detach(),
                     applied_update_norm=(raw_norm * scale).detach(),
                     clip_scale=scale.detach(),
                     correction_norm=torch.linalg.vector_norm(correction.detach(), dim=(1, 2)))
        return correction, candidate, stats
