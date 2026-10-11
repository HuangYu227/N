"""Keep telemetry out of checkpoint replay; differentiable work is unaffected."""
from contextvars import ContextVar
from contextlib import contextmanager

_recomputing = ContextVar("ttn_recomputing", default=0)
_replay_memory = ContextVar("ttn_replay_memory", default=None)


@contextmanager
def replay_memory_trace(callback):
    token = _replay_memory.set(callback)
    try:
        yield
    finally:
        _replay_memory.reset(token)


def trace_replay_memory(phase, **info):
    callback = _replay_memory.get()
    if callback is not None and not diagnostics_enabled():
        callback(phase, **info)


def diagnostics_enabled():
    return not _recomputing.get()


class recomputing:
    # Checkpoint re-enters the same context for repeated/higher-order backward.
    def __init__(self):
        # Autograd replay may run on another thread with an empty ContextVar context.
        self.memory_callback = _replay_memory.get()

    def __enter__(self):
        _recomputing.set(_recomputing.get() + 1)
        self.memory_token = _replay_memory.set(self.memory_callback)

    def __exit__(self, *error):
        _replay_memory.reset(self.memory_token)
        _recomputing.set(_recomputing.get() - 1)
