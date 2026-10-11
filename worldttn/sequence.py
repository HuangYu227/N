"""Functional layer-wise scan: isolated observed frame, then noisy three-frame groups.

Each DiT block is called once by FSDP. Recurrence and recomputation stay inside
that call; temporary states never commit to a runtime or detach their gradients.
"""
from contextlib import nullcontext
from dataclasses import fields, is_dataclass, replace

import torch
from torch.utils.checkpoint import checkpoint

from .anchor import TTNAnchor
from .diagnostics import diagnostics_enabled, recomputing, trace_replay_memory
from .memory_cache import storage_bytes


def _checkpoint(function, inputs, *, enabled):
    """Expose every live tensor to checkpoint storage instead of Python closures."""
    if not enabled or not torch.is_grad_enabled():
        return function(*inputs)
    tensors, seen = [], {}
    def flatten(value):
        if isinstance(value, torch.Tensor):
            if id(value) not in seen:
                seen[id(value)] = len(tensors)
                tensors.append(value)
            return ("tensor", seen[id(value)], value.stride())
        if is_dataclass(value):
            return ("dataclass", type(value), {f.name: flatten(getattr(value, f.name)) for f in fields(value)})
        if isinstance(value, dict):
            return ("dict", {key: flatten(item) for key, item in value.items()})
        if isinstance(value, (list, tuple)):
            return ("sequence", type(value), tuple(flatten(item) for item in value))
        return ("value", value)
    spec = flatten(inputs)
    # Break the recursive closure cycle before it pins GPU inputs until cyclic GC.
    del flatten
    def restore(node, values, restored):
        if node[0] == "tensor":
            if node[1] in restored:
                return restored[node[1]]
            value, stride = values[node[1]], node[2]
            if value.stride() != stride:
                # CPU storage packing must preserve original Linear/conv dispatch.
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
    return checkpoint(call, *tensors, use_reentrant=False,
                      context_fn=lambda: (nullcontext(), recomputing()))


def sequence_ranges(frames):
    if frames < 4 or frames % 3 != 1:
        raise ValueError("sequence training requires one observed frame plus 3n future frames")
    return [(0, 1)] + [(start, start + 3) for start in range(1, frames, 3)]


def _slice_rope(rope, start, end, tokens):
    if rope is None:
        return None
    # SANA WAN RoPE is [1,1,N,D/2].
    axis = rope.ndim - 2
    if rope.shape[axis] != tokens:
        raise ValueError("sequence RoPE token dimension disagrees with the patch grid")
    return rope.narrow(axis, start, end-start)


def sequence_block(block, x, y, t, mask, hw, rotary_emb, *, context, recompute=True, **kwargs):
    frames, height, width = hw
    area = height * width
    if context.frame_ids.shape[-1] != frames or not torch.equal(
            context.frame_ids, torch.arange(frames, device=x.device).expand(x.shape[0], -1)):
        raise ValueError("full sequence must start at the observed frame with contiguous frame IDs")
    if t.ndim == 2:
        t = t[:, None, None].expand(-1, 1, frames, -1)
    if t.ndim != 4 or t.shape[2] != frames:
        raise ValueError("full sequence requires per-frame timestep embeddings [B,1,F,C]")
    if kwargs.get("block_mask") is not None:
        raise ValueError("full sequence uses native within-chunk attention; custom block masks are unsupported")
    if any(kwargs.get(name) is not None for name in ("delta_pose_emb", "image_embeds")):
        raise ValueError("full sequence does not support delta-pose or image-embedding conditioning")
    anchor = block.attn.index if isinstance(block.attn, TTNAnchor) else None
    chunks = sequence_ranges(frames)

    # Clear mutable diagnostic payload before it becomes checkpoint metadata.
    snapshot = replace(context, candidates={}, memory_candidates={}, memory_stats={}, memory_trajectory=[],
                       local_stats={}, local_trajectory=[], sink_stats={}, sink_trajectory=[],
                       replay_stats={}, replay_trajectory=[])
    options = dict(kwargs)
    for name in ("kv_cache", "save_kv_cache", "prope_fns", "raymats", "cam_pos_embeds",
                 "_cross_attn_pad_to", "chunk_index", "chunk_index_global", "block_mask", "chunk_plucker"):
        options.pop(name, None)

    def run(features, conditioning, times, snapshot, sequence_mask, sequence_rope, options,
            cache, state, retained, camera_lengths, groups):
        if anchor is not None:
            trace_replay_memory("replay_span_begin", block=(3, 7, 11, 15, 19)[anchor],
                                start=groups[0][0], end=groups[-1][1])
        outputs, records = [], []
        states = snapshot.predicted
        offset = groups[0][0]
        text_kv = block.cross_attn.prepare_kv(conditioning, features.shape[0])
        for start, end in groups:
            left, right = start-offset, end-offset
            local_kwargs = dict(options, ttn_text_kv=text_kv)
            for name in ("camera_conditions", "frame_valid_mask"):
                if isinstance(local_kwargs.get(name), torch.Tensor):
                    local_kwargs[name] = local_kwargs[name][:, left:right]
            for name in ("camera_embedding", "plucker_emb"):
                if isinstance(local_kwargs.get(name), torch.Tensor):
                    local_kwargs[name] = local_kwargs[name][:, left*area:right*area]
            predicted = (torch.stack([state if i == anchor else states[:, i] for i in range(5)], 1)
                         if anchor is not None else states)
            local = replace(snapshot, predicted=predicted, previous=predicted,
                    frame_ids=snapshot.frame_ids[:, left:right], poses=snapshot.poses[:, left:right],
                    read_mask=snapshot.read_mask[:, left:right], write_mask=snapshot.write_mask[:, left:right],
                    clean_mode=True, prefill_mode=start == 0, candidates={}, memory_candidates={},
                    memory_caches=tuple(retained if i == anchor else None for i in range(5)),
                    memory_stats={}, memory_trajectory=[],
                    local_stats={}, local_trajectory=[], collect_local_stats=False)
            if getattr(block.attn, "_ttn_spectral_readout", None) is not None:
                # clean_mode here stages a private candidate, even during outer denoising.
                local.spectral_source_context = context
            group_rope = _slice_rope(sequence_rope, left*area, right*area, features.shape[1])
            def chunk(z, cond, time, incoming, prior, memory, local, group_mask, group_rope,
                      local_kwargs, start=start, end=end):
                if anchor is not None:
                    trace_replay_memory("replay_group_begin", block=(3, 7, 11, 15, 19)[anchor], start=start, end=end)
                out, next_cache = block._forward_chunk(z, cond, time, mask=group_mask,
                    THW=(end-start, height, width),
                    rotary_emb=group_rope,
                    kv_cache=list(incoming), save_kv_cache=True, ttn_live_cache=True,
                    ttn_chunk_context=local, **local_kwargs)
                if anchor is not None:
                    trace_replay_memory("replay_group_end", block=(3, 7, 11, 15, 19)[anchor], start=start, end=end)
                if anchor is None:
                    return out, next_cache, prior, memory, None
                candidate, _, stats = local.candidates[anchor]
                if diagnostics_enabled():
                    stats = dict(stats, block=(3, 7, 11, 15, 19)[anchor], prefill=start == 0,
                                 proximal=local.memory_stats[anchor], start=start, end=end,
                                 history_source="observed" if start == 0 else "noisy")
                    stats["measurement"] = ("observed prefill" if start == 0 else "noisy sequence group") + "; final frame state; detached diagnostics"
                    if start in (1, 13) or (start-1) % 12 == 0:
                        from .stability import state_dynamics
                        stats.update(state_dynamics(prior, prior, candidate, local.system.config.eps))
                        stats["inner_update_applied"] = True
                else:
                    stats = None
                return out, next_cache, candidate, local.memory_candidates.get(anchor), stats

            # CPU packing must not change Linear/conv dispatch on recomputation.
            args = (features[:, left*area:right*area].contiguous(), conditioning.contiguous(),
                    times[:, :, left:right].contiguous(), cache, state, retained, local,
                    sequence_mask, group_rope, local_kwargs)
            out, current, state, retained, stats = _checkpoint(chunk, args, enabled=recompute)
            if anchor is not None:
                # Native camera keeps two groups; slicing removes observations, never gradients of retained ones.
                camera_lengths = (camera_lengths + [current[2].shape[2] if current[2] is not None else 0])[-2:]
                limit = sum(camera_lengths)
                for index in (2, 3):
                    if current[index] is not None and cache[index] is not None:
                        current[index] = torch.cat((cache[index], current[index]), dim=2)[:, :, -limit:]
            cache = current
            outputs.append(out)
            if stats is not None:
                stats["memory_storage_bytes"] = storage_bytes(retained)
                records.append(stats)
        return torch.cat(outputs, dim=1), cache, state, retained, camera_lengths, records

    cache, camera_lengths, outputs, records = [None] * 10, [], [], []
    state = snapshot.predicted[:, anchor] if anchor is not None else None
    retained = snapshot.memory_caches[anchor] if anchor is not None and snapshot.memory_caches else None
    # Nested replay retains every group's input on GPU until its backward.
    # Bound anchor replay spans; boundary state/cache tensors stay on the live graph.
    span = 8 if recompute and anchor is not None else len(chunks)
    for first in range(0, len(chunks), span):
        groups = chunks[first:first+span]
        left, right = groups[0][0], groups[-1][1]
        part = replace(snapshot, frame_ids=snapshot.frame_ids[:, left:right],
            poses=snapshot.poses[:, left:right], read_mask=snapshot.read_mask[:, left:right],
            write_mask=snapshot.write_mask[:, left:right])
        part_options = dict(options)
        for name in ("camera_conditions", "frame_valid_mask"):
            if isinstance(part_options.get(name), torch.Tensor):
                part_options[name] = part_options[name][:, left:right].contiguous()
        for name in ("camera_embedding", "plucker_emb"):
            if isinstance(part_options.get(name), torch.Tensor):
                part_options[name] = part_options[name][:, left*area:right*area].contiguous()
        part_rope = _slice_rope(rotary_emb, left*area, right*area, frames*area)
        args = (x[:, left*area:right*area].contiguous(), y.contiguous(),
                t[:, :, left:right].contiguous(), part, mask, part_rope, part_options,
                cache, state, retained, camera_lengths, groups)
        out, cache, state, retained, camera_lengths, part_records = _checkpoint(run, args, enabled=recompute)
        outputs.append(out)
        records.extend(part_records)
    out = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
    if anchor is not None:
        # Publish detached telemetry only once, outside every checkpointed function.
        if context.sequence_mode or context.clean_mode:
            context.candidates[anchor] = (state, torch.zeros_like(context.psi[:, anchor]), records[-1])
            context.memory_candidates[anchor] = retained
            context.memory_stats[anchor] = records if context.sequence_mode else records[-1]["proximal"]
        elif context.memory_trajectory:
            context.memory_trajectory[-1]["anchors"][anchor] = records[-1]["proximal"]
    if not context.sequence_mode and not kwargs.get("save_kv_cache", False):
        # Solver calls may reuse this exact native-cache list. Only clean calls publish it.
        cache = list(kwargs["kv_cache"])
    return out, cache
