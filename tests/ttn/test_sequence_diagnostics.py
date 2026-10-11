import torch

from test_activation_checkpoint import cached_ffn, text_attention
from test_alignment_detail import cached_sana
from test_sequence_training import sequence_model, batch, run_forward
from worldttn import anchor
from worldttn.diagnostics import diagnostics_enabled, recomputing


def test_recomputation_context_supports_repeated_backward_and_higher_derivatives():
    from worldttn.sequence import _checkpoint
    value = torch.randn(4, dtype=torch.float64, requires_grad=True)
    loss = _checkpoint(lambda x: x.sin().square().sum(), (value,), enabled=True)
    first = torch.autograd.grad(loss, value, retain_graph=True, create_graph=True)[0]
    again = torch.autograd.grad(loss, value, retain_graph=True)[0]
    torch.testing.assert_close(first, again, rtol=0, atol=0)
    second = torch.autograd.grad(first.sum(), value)[0]
    torch.testing.assert_close(second, 2*torch.cos(2*value))
    assert diagnostics_enabled()


def test_recompute_scope_is_nested_and_restored_after_errors():
    assert diagnostics_enabled()
    try:
        with recomputing():
            assert not diagnostics_enabled()
            with recomputing():
                assert not diagnostics_enabled()
            assert not diagnostics_enabled()
            raise ValueError("test")
    except ValueError:
        pass
    assert diagnostics_enabled()


def test_replay_memory_trace_only_observes_recompute_and_restores_scope():
    from worldttn.diagnostics import replay_memory_trace, trace_replay_memory
    records = []
    with replay_memory_trace(lambda phase, **info: records.append((phase, info))):
        trace_replay_memory("forward")
        with recomputing():
            trace_replay_memory("replay", block=19)
            try:
                with replay_memory_trace(None):
                    trace_replay_memory("disabled")
                    raise ValueError("test")
            except ValueError:
                pass
            trace_replay_memory("restored")
    with recomputing():
        trace_replay_memory("outside")
    assert records == [("replay", {"block": 19}), ("restored", {})]


def test_replay_memory_callback_survives_an_autograd_worker_thread():
    from concurrent.futures import ThreadPoolExecutor
    from worldttn.diagnostics import replay_memory_trace, trace_replay_memory
    records = []
    with replay_memory_trace(lambda phase, **info: records.append((phase, info))):
        scope = recomputing()
    def replay():
        with scope:
            trace_replay_memory("worker", block=19)
        trace_replay_memory("outside")
    with ThreadPoolExecutor(max_workers=1) as worker:
        worker.submit(replay).result()
    assert records == [("worker", {"block": 19})]


def test_anchor_diagnostics_are_not_repeated_during_backward(monkeypatch, sequence_model):
    calls = []
    original = anchor.clean_anchor_stats
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(anchor, "clean_anchor_stats", counted)
    for block in sequence_model.blocks:
        block.ttn_sequence_checkpoint = True
    x, _, t, cam = batch()
    out, context, _ = run_forward(sequence_model, x.requires_grad_(), t, cam)
    forward_count = len(calls)
    assert forward_count == 15  # observed + two groups at each of five anchors
    out[:, :, -3:].square().mean().backward()
    assert len(calls) == forward_count
    assert x.grad[:, :, :1].norm() > 0
    assert all(len(rows) == 3 for rows in context.memory_stats.values())
