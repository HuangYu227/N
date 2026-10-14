"""Production saved-tensor reuse keeps versions, weak ownership, and FSDP copies."""
import gc
import weakref
import torch
from worldttn import training

def test_pageable_snapshot_reuses_exact_activation_views_without_retaining_them():
    source = torch.arange(24., requires_grad=True)
    activation = (source * 2).reshape(4, 6)
    storage = training.activation_storage("cpu-pageable", device="cuda:0")
    first = storage.pack_hook(activation[:, 1::2])
    second = storage.pack_hook(activation[:, 1::2])
    assert first[1] is second[1]
    assert first[1].untyped_storage().data_ptr() != activation.untyped_storage().data_ptr()
    assert storage.pack_hook(activation[:, ::2])[1] is not first[1]
    with torch.no_grad(): activation.add_(1)
    changed = storage.pack_hook(activation[:, 1::2])
    assert changed[1] is not first[1]
    torch.testing.assert_close(changed[1], first[1] + 1, rtol=0, atol=0)
    root_ref, snapshot_ref = weakref.ref(activation._base), weakref.ref(first[1])
    del activation, first, second, changed
    gc.collect()
    assert root_ref() is None and snapshot_ref() is None
    # FSDP may repopulate parameter storage without changing its version counter.
    parameter = torch.nn.Parameter(torch.arange(24.).reshape(4, 6))
    assert storage.pack_hook(parameter.t())[1] is not storage.pack_hook(parameter.t())[1]
    live = source * 3
    previous = storage.pack_hook(live)
    fresh = training.activation_storage("cpu-pageable", device="cuda:0")
    assert fresh.pack_hook(live)[1] is not previous[1]
