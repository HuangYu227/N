"""Two pure-Tensor Inductor graphs; no parameters, camera, transaction or mutation."""
from functools import lru_cache
import time
import torch
from .core import correct_with_aux

_warmup_enabled = False
_stable_window = False
_warmed = {}


def benchmark_warmup(enabled, *, stable=False):
    """Opt-in benchmark lifecycle, never changes ordinary training dispatch."""
    global _warmup_enabled, _stable_window
    if enabled and not stable: _warmed.clear()
    _warmup_enabled, _stable_window = enabled, stable


def warmup_report():
    return {"specializations": len(_warmed), "seconds": sum(_warmed.values()),
            "method": "independent tensors with actual layout/grad mode; RNG restored; no optimizer or runtime"}


def _dispatch(which, tensors, alpha_s, eps):
    fn = helpers()[which]
    if _warmup_enabled:
        key = (which, torch.is_grad_enabled(), alpha_s, eps, tuple(
            (tuple(x.shape), x.stride(), x.storage_offset(), x.dtype, x.device, x.requires_grad) for x in tensors))
        if key not in _warmed:
            if _stable_window: raise RuntimeError("new TTN specialization in stable benchmark window; run layout validation first")
            q = tensors[0]
            devices = [q.device.index if q.device.index is not None else torch.cuda.current_device()] if q.is_cuda else []
            started = time.perf_counter()
            with torch.random.fork_rng(devices=devices):
                clones = []
                for x in tensors:
                    size = 1 + x.storage_offset() + sum((s - 1) * stride for s, stride in zip(x.shape, x.stride()))
                    isolated = torch.empty(size, device=x.device, dtype=x.dtype).as_strided(x.shape, x.stride(), x.storage_offset())
                    isolated.fill_(True if x.dtype == torch.bool else .1)
                    isolated.requires_grad_(x.requires_grad)
                    clones.append(isolated)
                result = fn(*clones, alpha_s, eps)
                if torch.is_grad_enabled():
                    live = [x for x in clones if x.requires_grad]
                    if live:
                        loss = result.square().mean() if which == 0 else result[0].square().mean() + result[1].square().mean()
                        torch.autograd.grad(loss, live)
                if q.is_cuda: torch.cuda.synchronize(q.device)
            _warmed[key] = time.perf_counter() - started
    return fn(*tensors, alpha_s, eps)


def _read_only(q, k, v, beta, predicted, write, alpha_s, eps):
    return q @ correct_with_aux(predicted, k, v, beta, write, alpha_s, eps).candidate


def _with_aux(q, k, v, beta, predicted, write, alpha_s, eps):
    aux = correct_with_aux(predicted, k, v, beta, write, alpha_s, eps)
    return (q @ aux.candidate, *aux)


@lru_cache(maxsize=1)
def helpers():
    # Tensor layout/grad-mode specializations are audited by a separate validation run.
    # Explicit compiled requests propagate errors rather than silently falling back.
    options = dict(backend="inductor", fullgraph=True, dynamic=False, mode="default")
    return torch.compile(_read_only, **options), torch.compile(_with_aux, **options)


def read_only(q, k, v, beta, predicted, write, alpha_s, eps):
    return _dispatch(0, (q, k, v, beta, predicted, write), alpha_s, eps)


def with_aux(q, k, v, beta, predicted, write, alpha_s, eps):
    return _dispatch(1, (q, k, v, beta, predicted, write), alpha_s, eps)
