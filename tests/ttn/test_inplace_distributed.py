"""FSDP hooks expose only this block's live FFN state and preserve the public output."""
from types import SimpleNamespace

import torch
from torch import nn

from worldttn.distributed import _expose_clean_state, _restore_block_output


def test_fsdp_exposes_current_ffn_candidate_without_other_blocks():
    current = torch.randn(1, 16, 32, requires_grad=True)
    other = torch.randn(1, 16, 32, requires_grad=True)
    module = SimpleNamespace(attn=nn.Identity(), mlp=SimpleNamespace(ttn_ffn_index=7))
    context = SimpleNamespace(clean_mode=True, ffn_candidates={3: (other, {}), 7: (current, {})})
    output = (torch.randn(1, 3, 16), [None]*10)
    exposed = _expose_clean_state(module, (), {'ttn_chunk_context': context}, output)
    assert exposed[3] == (current,)
    assert _restore_block_output(module, (), exposed) is output
    context.clean_mode = False
    assert len(_expose_clean_state(module, (), {'ttn_chunk_context': context}, output)) == 3
