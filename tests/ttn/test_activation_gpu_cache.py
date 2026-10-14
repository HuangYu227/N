"""Saved snapshots keep exact values and release their bounded GPU cache."""
import gc
import pytest
import torch
from worldttn.training import activation_storage


def test_invalid_cache_budget_is_rejected(monkeypatch):
    monkeypatch.setenv('TTN_ACTIVATION_GPU_CACHE_GIB','nan')
    with pytest.raises(ValueError,match='finite'):activation_storage('cpu-pageable')
    with pytest.raises(ValueError):activation_storage('cpu-pageable',gpu_budget_bytes=-1)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA saved-tensor storage required')
def test_bounded_gpu_snapshots_reuse_views_copy_parameters_and_release_budget():
    x=torch.arange(24.,device='cuda',requires_grad=True)
    y=(x*2).reshape(4,6)
    storage=activation_storage('cpu-pageable',gpu_budget_bytes=48)
    first=storage.pack_hook(y[:,1::2]);second=storage.pack_hook(y[:,1::2])
    assert first[1] is second[1] and first[1].is_cuda
    assert first[1].untyped_storage().data_ptr()!=y.untyped_storage().data_ptr()
    spill=storage.pack_hook(y[:,::2]);assert spill[1].device.type=='cpu'
    assert storage.storage_stats['gpu_live_bytes']==48
    with torch.no_grad():y.add_(1)
    torch.testing.assert_close(storage.unpack_hook(first),y[:,1::2]-1,rtol=0,atol=0)
    del first,second;gc.collect()
    assert storage.storage_stats['gpu_live_bytes']==0
    p=torch.nn.Parameter(torch.arange(12.,device='cuda').reshape(3,4))
    packed=storage.pack_hook(p)
    with torch.no_grad():p.add_(5)
    torch.testing.assert_close(storage.unpack_hook(packed),p-5,rtol=0,atol=0)
    assert storage.storage_stats['gpu_peak_bytes']<=48
    del packed;gc.collect();assert storage.storage_stats['gpu_live_bytes']==0
    limited=activation_storage('cpu-pageable',gpu_budget_bytes=48,gpu_allocation_limit_bytes=1)
    assert limited.pack_hook(p)[1].device.type=='cpu'


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA saved-tensor backward required')
def test_cached_and_pageable_outputs_and_gradients_match_exactly():
    torch.manual_seed(881)
    x=torch.randn(7,12,device='cuda');w=torch.randn(12,8,device='cuda')
    results=[]
    for budget in (0,1<<20):
        a=x.clone().requires_grad_();b=w.clone().requires_grad_()
        storage=activation_storage('cpu-pageable',gpu_budget_bytes=budget)
        with storage:
            out=(a@b).sin();out.square().mean().backward()
        results.append((out.detach(),a.grad,b.grad))
        if budget:assert storage.storage_stats['gpu_packs']>0
    for a,b in zip(*results):torch.testing.assert_close(a,b,rtol=0,atol=0)
