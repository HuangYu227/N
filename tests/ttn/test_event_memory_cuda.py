"""Real head/grid dimensions on one GPU; no pretrained weights or training."""
import pytest
import torch

from test_anchor import ProjectionContract
from worldttn.anchor import TTNAnchor
from worldttn.core import ANCHORS, TTNConfig
from worldttn.event_memory import EventMemoryOptions
from worldttn.runtime import TTNRuntimeState, TTNSystem


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@torch.no_grad()
def test_production_cfg_grid_clean_commit_and_temporary_read():
    torch.manual_seed(91)
    device = torch.device('cuda:0')
    config = TTNConfig(heads=20, head_dim=112, generators=16, stage='C', local_update=True,
                       persistent_meta=True, eta_psi=0, state_ttt_rank=64, state_ttt_gain=.5)
    options = EventMemoryOptions(mode='state-replay', start_chunk=1)
    system = TTNSystem(config).to(device)
    anchors = [TTNAnchor(ProjectionContract(20, 112), i, config).eval().to(device) for i in range(5)]
    runtime = TTNRuntimeState.create(config, 2, device, memory_options=options)

    def begin(ids, prefill=False):
        frames = torch.tensor(ids, device=device)[None].expand(2, -1)
        poses = torch.eye(4, device=device).expand(2, len(ids), 4, 4)
        intrinsics = torch.tensor([40., 22., 20., 11.], device=device).expand(2, len(ids), 4)
        return runtime.begin_chunk(system, poses, intrinsics, frames, torch.ones_like(frames, dtype=torch.bool),
                                   40, 22, prefill=prefill)

    clean = begin([0], prefill=True).for_clean()
    features = torch.randn(2, 880, 2240, device=device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for anchor in anchors:
            anchor(features, HW=(1, 22, 40), ttn_chunk_context=clean)
    runtime.prefill(clean)
    assert len(runtime.event_memory) == 1
    assert set(clean.memory_pending) == {11, 15, 19}
    before = runtime.world_state.clone()
    context = begin([1, 2, 3])
    context.memory_sigma, context.memory_call = .25, 1
    features = torch.randn(2, 2640, 2240, device=device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for anchor in anchors:
            value = anchor(features, HW=(3, 22, 40), ttn_chunk_context=context)
            assert torch.isfinite(value).all()
    assert torch.equal(runtime.world_state, before)
    assert len(context.memory_stats) == 3
    for record in context.memory_stats:
        assert record['anchor'] in ANCHORS and record['active']
        assert 'output_effect' in record
        for branch in record['batches']:
            assert max(branch['relative_delta']) <= options.max_relative_delta + 1e-6
    assert not context.candidates and not context.memory_pending
