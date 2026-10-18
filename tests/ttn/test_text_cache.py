import pytest
import torch
from worldttn.text_cache import frozen_text_cache


def test_frozen_encoder_reuses_exact_prompts_and_keeps_training_rng():
    encoder = torch.nn.Linear(3, 3).eval().requires_grad_(False)
    calls = []
    def encode(prompts):
        calls.append(prompts)
        return encoder(torch.ones(1, 3)), torch.ones(1, 3, dtype=torch.long)
    cache = frozen_text_cache(encoder, encode, 1)
    state = torch.get_rng_state()
    first = cache(("forest",))
    again = cache(("forest",))
    assert len(calls) == 1 and first is again
    assert all(not value.requires_grad for value in first)
    assert torch.equal(state, torch.get_rng_state())
    for x, y in zip(first, encode(["forest"])):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    cache(("room",)); cache(("forest",))
    assert cache.cache_info().misses == 3 and cache.cache_info().hits == 1


def test_cache_rejects_trainable_or_random_training_mode_encoder():
    encoder = torch.nn.Linear(3, 3)
    with pytest.raises(ValueError, match="frozen CPU encoder"):
        frozen_text_cache(encoder, lambda prompts: (), 2)
    encoder.eval()
    with pytest.raises(ValueError, match="frozen CPU encoder"):
        frozen_text_cache(encoder, lambda prompts: (), 2)
