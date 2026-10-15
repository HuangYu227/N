"""Layer-local scan of the original observed/noisy/clean chunk recurrence."""
from dataclasses import replace

import torch

from .anchor import TTNAnchor
from .core import ANCHORS, CayleyFactors, clip_inner_gradient
from .fullgradient import checkpoint_tree, _RECOMPUTING
from .stability import committed_anchor_stats
from .training import chunk_ranges, _memory_phase


def legacy_sequence_block(block, x, y, t, mask, hw, rope, context, **kwargs):
    frames, height, width = hw
    _memory_phase(context.sequence_memory_callback, 'layer_begin', block=block.attn.index if isinstance(block.attn, TTNAnchor) else None)
    area, batch = height*width, x.shape[0]//3
    if x.shape[0] != 3*batch or batch < 1 or t.ndim != 4 or t.shape[2] != frames:
        raise ValueError("legacy sequence requires observed/noisy/clean streams and per-frame times")
    anchor = block.attn.index if isinstance(block.attn, TTNAnchor) else None
    inplace = context.system.config.ttt_backend == 'inplace-ffn'
    ffn_index = ANCHORS[anchor] if inplace and anchor is not None else None
    ranges = [(0, 1), *chunk_ranges(frames)]
    snapshot = replace(context, candidates={}, ffn_fast={}, ffn_candidates={}, local_stats={}, local_trajectory=[])
    options = dict(kwargs)
    for name in ('kv_cache', 'save_kv_cache', 'prope_fns', '_cross_attn_pad_to',
                 'chunk_index', 'chunk_index_global', 'block_mask'):
        options.pop(name, None)
    checkpointed = getattr(block, 'ttn_fullgrad_checkpoint', True)
    cpu = context.checkpoint_cpu_inputs
    conditioning = y.reshape(3*batch, -1, y.shape[-1])[:batch]
    group_mask = mask[:batch] if isinstance(mask, (torch.Tensor, list, tuple)) else mask
    # Weight-dependent text projections are shared only within this block forward.
    text_kv = block.cross_attn.prepare_kv(conditioning, batch)
    options.update(ttn_live_cache=True, ttn_text_kv=text_kv)
    def options_for(start, end, source=None):
        part = dict(options if source is None else source)
        for name in ('camera_conditions', 'frame_valid_mask'):
            if isinstance(part.get(name), torch.Tensor):
                part[name] = part[name][:batch, start:end].contiguous()
        for name in ('camera_embedding', 'plucker_emb'):
            if isinstance(part.get(name), torch.Tensor):
                part[name] = part[name][:batch, start*area:end*area].contiguous()
        return part

    def local_context(state, psi, delta, coefficients, start, end, prefill=False):
        previous = snapshot.previous[:batch]
        fast = snapshot.psi[:batch]
        if anchor is not None:
            previous = torch.stack([state if i == anchor else previous[:, i] for i in range(5)], 1)
            fast = torch.stack([psi if i == anchor else fast[:, i] for i in range(5)], 1)
        # FP32 storage alone does not protect the Gram/transport GEMMs from AMP.
        with torch.autocast(device_type=previous.device.type, enabled=False):
            factors = None if prefill else CayleyFactors(snapshot.system.generators.u.float(),
                snapshot.system.generators.v.float(), coefficients+snapshot.system.config.delta_psi*fast.tanh())
            predicted = previous if prefill else factors.right(previous)
        valid = snapshot.read_mask[:batch, start:end]
        ids = snapshot.frame_ids[:batch, start:end]
        return replace(snapshot, previous=previous, predicted=predicted, psi=fast, cbase=coefficients,
            live_factors=factors, poses=snapshot.poses[:batch, start:end], frame_ids=ids,
            read_mask=valid, write_mask=valid if prefill else valid & (ids != 0),
            clean_mode=False, prefill_mode=prefill, candidates={}, local_stats={}, local_trajectory=[],
            ffn_fast={ffn_index: delta} if ffn_index is not None else {}, ffn_candidates={},
            collect_local_stats=not _RECOMPUTING.get(), sequence_mode=False,
            measure_psi_output=not inplace and not prefill and start in (0, ranges[len(ranges)//2][0], frames-3),
            noise_info={key:[[row[0][start:end]] for row in value] for key,value in snapshot.noise_info.items()})

    def call_chunk(features, conditioning, times, incoming, state, psi, delta, coefficients,
                   options, group_mask, group_rope, start, end, prefill=False):
        local = local_context(state, psi, delta, coefficients, start, end, prefill)
        group_options = options
        def forward(z, time, ctx):
            return block._forward_chunk(z, conditioning, time, mask=group_mask,
                THW=(end-start, height, width), rotary_emb=group_rope,
                kv_cache=list(incoming), save_kv_cache=True, ttn_chunk_context=ctx, **group_options)
        if prefill:
            clean_context = local.for_clean()
            out, _ = forward(features, times, clean_context)
            if anchor is None: return out, state, psi, delta, None, None
            state, gradient, stats = clean_context.candidates[anchor]
            return out, state, psi, delta, None if _RECOMPUTING.get() else committed_anchor_stats(
                stats, psi, psi, gradient, torch.ones_like(psi[..., :1]),
                coefficients[:, anchor], ANCHORS[anchor], True), None
        noisy, clean = features.chunk(2, 0)
        noisy_time, clean_time = times.chunk(2, 0)
        noisy_output, _ = forward(noisy, noisy_time, local)
        clean_context = local.for_clean()
        clean_output, current = forward(clean, clean_time, clean_context)
        if anchor is not None:
            state, gradient, stats = clean_context.candidates[anchor]
            if inplace:
                updated, scale = psi, torch.ones_like(gradient[..., :1])
            else:
                clipped, scale = clip_inner_gradient(gradient, snapshot.system.config.inner_clip,
                                                      snapshot.system.config.eps)
                updated = psi-snapshot.system.config.eta_psi*clipped
            if not _RECOMPUTING.get():
                stats = committed_anchor_stats(stats, psi, updated, gradient, scale,
                                                coefficients[:, anchor], ANCHORS[anchor], False)
                stats['local'] = ({'enabled': False, 'recorded': False} if inplace else
                    local.local_stats.get(anchor, {'enabled': True, 'recorded': False}))
                stats['persistent'] = {'meta_gradient_enabled': not inplace and gradient.requires_grad,
                                       'update_enabled': not inplace, 'eta': snapshot.system.config.eta_psi}
            else: stats = None
            psi = updated
        else:
            stats = None
        ffn_stats = None
        if ffn_index is not None:
            if set(clean_context.ffn_candidates) != {ffn_index}:
                raise RuntimeError('missing layer-local FFN clean candidate')
            delta, ffn_stats = clean_context.ffn_candidates[ffn_index]
        return torch.cat((noisy_output, clean_output)), current, state, psi, delta, stats, ffn_stats

    state = snapshot.previous[:batch, anchor] if anchor is not None else None
    psi = snapshot.psi[:batch, anchor] if anchor is not None else None
    delta = context.ffn_fast[ffn_index] if ffn_index is not None else None
    prefix_coefficients = snapshot.cbase[:batch]*0
    prefix_rope = rope.narrow(rope.ndim-2, 0, area) if rope is not None else None
    def prefill(z, conditioning, times, state, psi, delta, coefficients, options, mask, rope):
        return call_chunk(z, conditioning, times, [None]*10, state, psi, delta, coefficients,
                          options, mask, rope, 0, 1, True)
    prefix, state, psi, delta, prefix_stats, _ = checkpoint_tree(prefill,
        (x[:batch, :area].contiguous(), conditioning.contiguous(), t[:batch, :, :1].contiguous(),
         state, psi, delta, prefix_coefficients, options_for(0, 1), group_mask, prefix_rope),
        enabled=checkpointed)
    cache, lengths, outputs, records, ffn_records = [None]*10, [], [], [prefix_stats], [None]
    def span_forward(z, conditioning, times, cache, state, psi, delta, lengths, coefficients, options, mask, rope, groups):
        left = groups[0][0]
        span_outputs, span_records, span_ffn_records = [], [], []
        for index, (start, end) in enumerate(groups):
            lo, hi = start-left, end-left
            group_rope = rope.narrow(rope.ndim-2, lo*area, (hi-lo)*area) if rope is not None else None
            def group(z, conditioning, times, cache, state, psi, delta, coefficient, options, mask, rope,
                      start=start, end=end):
                return call_chunk(z, conditioning, times, cache, state, psi, delta, coefficient,
                                  options, mask, rope, start, end)
            out, current, state, psi, delta, stats, ffn_stats = checkpoint_tree(group,
                (z[:,lo*area:hi*area].contiguous(), conditioning, times[:,:,lo:hi].contiguous(),
                 cache, state, psi, delta, coefficients[index], options_for(lo,hi,options), mask, group_rope),
                enabled=checkpointed, cpu_inputs=(3,4,5,6) if cpu else ())
            if anchor is not None:
                lengths = (lengths+[current[2].shape[2] if current[2] is not None else 0])[-context.camera_cached_chunks:]
                for slot in (2,3):
                    if current[slot] is not None and cache[slot] is not None:
                        current[slot] = torch.cat((cache[slot],current[slot]),dim=2)[:,:,-sum(lengths):].clone()
            cache = current
            span_outputs.append(out);span_records.append(stats);span_ffn_records.append(ffn_stats)
        return torch.cat(span_outputs,dim=1),cache,state,psi,delta,lengths,span_records,span_ffn_records
    # Reuse the proven sequence engine's bounded replay; all span boundaries keep their gradients.
    span_size = 6 if anchor is not None and checkpointed else len(ranges)-1
    for first in range(0,len(ranges)-1,span_size):
        groups = ranges[1+first:1+first+span_size]
        left,right = groups[0][0],groups[-1][1]
        span_rope = rope.narrow(rope.ndim-2,left*area,(right-left)*area) if rope is not None else None
        out,cache,state,psi,delta,lengths,span_records,span_ffn_records = checkpoint_tree(span_forward,
            (x[batch:,left*area:right*area].contiguous(),conditioning.contiguous(),
             t[batch:,:,left:right].contiguous(),cache,state,psi,delta,lengths,
             context.sequence_coefficients[first:first+len(groups)],options_for(left,right),group_mask,span_rope,groups),
            enabled=checkpointed,cpu_inputs=(3,4,5,6) if cpu else ())
        outputs.append(out);records.extend(span_records);ffn_records.extend(span_ffn_records)
    joined = torch.cat(outputs, dim=1)
    observed = torch.cat((prefix, torch.zeros_like(x[:batch, area:])), dim=1)
    if anchor is not None:
        context.candidates[anchor] = (state, psi, records)
    if ffn_index is not None:
        context.ffn_candidates[ffn_index] = (delta, ffn_records)
    _memory_phase(context.sequence_memory_callback, 'layer_end', anchor=ANCHORS[anchor] if anchor is not None else None)
    return torch.cat((observed, joined), 0), cache
