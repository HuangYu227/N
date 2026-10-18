"""Run-local memoization for the frozen, deterministic CPU text encoder."""
from functools import lru_cache
import torch


def frozen_text_cache(encoder, encode, maxsize):
    if maxsize < 1:
        raise ValueError("text feature cache size must be positive")
    if any(module.training for module in encoder.modules()) or any(
            p.requires_grad or p.device.type != "cpu" for p in encoder.parameters()):
        raise ValueError("text feature cache requires a frozen CPU encoder in eval mode")

    @lru_cache(maxsize=maxsize)
    @torch.no_grad()
    def cached(prompts):
        return tuple(value.detach().cpu() for value in encode(list(prompts)))

    return cached
