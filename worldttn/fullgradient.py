"""Full-episode credit with pure, replayable legacy chunk transactions."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import fields, is_dataclass

import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from .session import TTNSession

_RECOMPUTING = ContextVar("psi_recomputing", default=False)


@contextmanager
def recomputing():
    token = _RECOMPUTING.set(True)
    try:
        yield
    finally:
        _RECOMPUTING.reset(token)


def checkpoint_tree(function, inputs, *, enabled=True, cpu_inputs=()):
    """Existing sequence checkpoint layout contract, applied to legacy chunks."""
    if not enabled or not torch.is_grad_enabled():
        return function(*inputs)
    tensors, seen = [], {}
    def flatten(value, offload=False):
        if isinstance(value, torch.Tensor):
            if id(value) not in seen:
                seen[id(value)] = len(tensors)
                tensors.append(value.to("cpu") if offload else value)
            return ("tensor", seen[id(value)], value.stride(), value.device)
        if is_dataclass(value):
            return ("dataclass", type(value), {f.name: flatten(getattr(value, f.name), offload) for f in fields(value)})
        if isinstance(value, dict):
            return ("dict", {key: flatten(item, offload) for key, item in value.items()})
        if isinstance(value, (list, tuple)):
            return ("sequence", type(value), tuple(flatten(item, offload) for item in value))
        return ("value", value)
    spec = ("sequence", tuple, tuple(flatten(value, i in cpu_inputs) for i, value in enumerate(inputs)))
    del flatten
    def restore(node, values, restored):
        if node[0] == "tensor":
            if node[1] in restored:
                return restored[node[1]]
            value, stride = values[node[1]], node[2]
            value = value.to(node[3])
            if value.stride() != stride:
                shape = tuple(size if step else 1 for size, step in zip(value.shape, stride))
                unique = value[tuple(slice(None) if step else slice(0, 1) for step in stride)]
                value = torch.empty_strided(shape, stride, dtype=value.dtype, device=value.device).copy_(unique).expand(value.shape)
            restored[node[1]] = value
            return value
        if node[0] == "dataclass":
            return node[1](**{name: restore(item, values, restored) for name, item in node[2].items()})
        if node[0] == "dict":
            return {name: restore(item, values, restored) for name, item in node[1].items()}
        if node[0] == "sequence":
            return node[1](restore(item, values, restored) for item in node[2])
        return node[1]
    def call(*values):
        return function(*restore(spec, values, {}))
    # A replay must finish every FSDP forward/post-forward on every rank.
    with set_checkpoint_early_stop(False):
        return checkpoint(call, *tensors, use_reentrant=False,
                          context_fn=lambda: (nullcontext(), recomputing()))


def _session(episode, *, state=None, psi=None, index=-1, history=(), camera_lengths=None):
    old = episode.session
    session = TTNSession(old.model, old.camera, old.width, old.height, old.valid_mask, old.extras,
                         collect_local_stats=not _RECOMPUTING.get(), live_cache=True)
    runtime = session.reset(episode.clean.shape[0])
    if state is not None:
        runtime.world_state, runtime.transition_fast = state, psi
        prior_end = 1 if index == 0 else episode.ranges[index-1][1]
        runtime.previous_committed_pose = old.camera[:, prior_end-1, :16].reshape(-1, 4, 4)
        runtime.committed_frame_ids = [set(range(prior_end)) for _ in range(state.shape[0])]
        runtime.commit_count = runtime.predict_count = runtime.revision = index+1
        runtime.prefilled, runtime.write_history = True, history
        session.camera_cache_lengths = {i: list(v) for i, v in camera_lengths.items()}
    return session


def full_gradient_forward(runner, episode, on_prediction=None):
    """Private transactions return all live state; replay cannot mutate the episode."""
    from .training import _memory_phase
    use_checkpoint = getattr(runner.model, "ttn_fullgrad_checkpoint", True)
    cpu = episode.clean.is_cuda and getattr(runner.model, "ttn_fullgrad_cpu_inputs", False)
    def prefill(initial):
        session = _session(episode)
        session.prefill(initial, episode.y, episode.mask, episode.data_info)
        r = session.runtime
        return r.world_state, r.transition_fast, r.last_stats
    _memory_phase(episode.memory_callback, "prefill_begin")
    state, psi, episode.prefill_stats = checkpoint_tree(prefill, (episode.clean[:, :, :1],), enabled=use_checkpoint)
    _memory_phase(episode.memory_callback, "prefill_end")
    cache, history = episode.cache, ()
    camera_lengths = {i: [] for i in (3, 7, 11, 15, 19)}
    losses = []
    for index, (start, end) in enumerate(episode.ranges):
        def group(clean, timesteps, noise, state, psi, cache, history, camera_lengths,
                  index=index, start=start, end=end):
            session = _session(episode, state=state, psi=psi, index=index,
                               history=history, camera_lengths=camera_lengths)
            context = session.begin_chunk(start, end)
            predictions = []
            loss_mask = episode.loss_valid[:, None, start:end, None, None].float()
            per_example = runner.loss_fn(session, clean, timesteps, noise, episode.y, context,
                cache, start, end, episode.mask, episode.data_info, loss_mask,
                predictions.append if on_prediction else None)
            loss = (per_example * (episode.loss_valid[:, start:end].sum(-1)/episode.total)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite full-episode flow loss")
            _, cache = session.clean_forward(clean, episode.y, context, cache, start, end,
                episode.mask, episode.data_info, cached_chunks=episode.cached_chunks)
            r = session.runtime
            return (loss, r.world_state, r.transition_fast, cache, r.write_history,
                    session.camera_cache_lengths, r.last_stats,
                    predictions[-1] if on_prediction else None)
        _memory_phase(episode.memory_callback, "chunk_begin", chunk=index, start=start, end=end)
        args = (episode.clean[:, :, start:end], episode.timesteps[:, :, start:end],
                episode.noise[:, :, start:end], state, psi, cache, history, camera_lengths)
        loss, state, psi, cache, history, camera_lengths, stats, prediction = checkpoint_tree(
            group, args, enabled=use_checkpoint, cpu_inputs=(3, 4, 5, 6) if cpu else ())
        losses.append(loss)
        episode.records.append(dict(stats, chunk=index, start=start, end=end, clean_future_credit=index+1<len(episode.ranges)))
        if on_prediction:
            on_prediction(index, prediction)
        _memory_phase(episode.memory_callback, "chunk_end", chunk=index, start=start, end=end)
    runtime = episode.session.runtime
    runtime.world_state, runtime.transition_fast = state, psi
    runtime.previous_committed_pose = episode.session.camera[:, -1, :16].reshape(-1, 4, 4)
    runtime.committed_frame_ids = [set(range(episode.clean.shape[2])) for _ in range(state.shape[0])]
    runtime.commit_count = runtime.predict_count = len(episode.ranges)+1
    runtime.revision, runtime.prefilled = len(episode.ranges)+1, True
    runtime.last_stats, runtime.write_history = stats, history
    episode.cache = cache
    loss = torch.stack(losses).sum()
    episode.total_loss = float(loss.detach())
    return loss
