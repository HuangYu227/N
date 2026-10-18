"""Real CFG/head/grid sizes for the scoped frozen-read diagnostic."""
import pytest
import torch

from tools.ttn_event_memory_probe import FrozenReadProbe
from tools.ttn_event_memory_probe_math import POLICIES
from worldttn.event_memory import EventMemoryBank, EventMemoryOptions, pooled_descriptor, sufficient_statistics


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@torch.no_grad()
def test_real_cfg_grid_capture_and_frozen_norm_matched_injection():
    torch.manual_seed(113)
    bank = EventMemoryBank(EventMemoryOptions(mode='state-replay', anchors=(11,), topk=1), 2)
    for index in range(8):
        keys = torch.randn(2, 20, 880, 112, device='cuda') * .1
        values = torch.randn_like(keys)
        write = torch.ones(2, 880, dtype=torch.bool, device='cuda')
        weights = torch.full((2, 20, 880), 1/880, device='cuda')
        gram, rhs, denom = sufficient_statistics(keys, values, weights, write, 1e-6)
        descriptor = pooled_descriptor(keys, values, write, grid=(1, 22, 40))
        bank = bank.with_pending({11: dict(descriptor=descriptor, gram=gram, rhs=rhs,
            denom=denom, frame_ids=torch.full((2, 1), index, device='cuda', dtype=torch.long))}, index)
    queries = descriptor.detach().cpu().clone()
    state = torch.randn(2, 20, 112, 112, device='cuda') * .03
    class UnusedSampler:
        def sample_chunks(self):
            yield from ()
    capture = FrozenReadProbe((19,))
    with capture.install(UnusedSampler):
        selections = tuple(bank.select(b, 11, queries[b], 20) for b in range(2))
        output, _ = bank.read(state, selections, 11)
        assert output is state
    saved = capture.snapshots['20:11:0']
    frozen = dict(snapshots=capture.snapshots, bank_hashes=capture.bank_hashes)
    assert saved['common_norm'].gt(0).all()
    for policy in POLICIES:
        probe = FrozenReadProbe((19,), frozen=frozen, policy=policy, stop_index=19)
        with probe.install(UnusedSampler):
            selections = tuple(bank.select(b, 11, queries[b], 20) for b in range(2))
            actual, _ = bank.read(state, selections, 11)
        torch.testing.assert_close(actual, state + saved['matched_deltas'][policy].to('cuda'), rtol=0, atol=0)
        assert probe.injected_keys == ['20:11:0']
