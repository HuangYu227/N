"""Read-only hooks on production TTNAnchor/StateTTT; no model replacements."""
import torch

from memory_math import make_reference, retention_metrics


class MemoryTrace:
    def __init__(self, model, anchors, probe_count=64):
        self.model, self.anchors, self.probe_count = model, tuple(anchors), probe_count
        self.references, self.prefix_hashes, self.rows = {}, {}, []
        self.calls = {'clean': 0, 'local': 0}
        self.inner, self.phase, self.chunk, self.transport = {}, 'idle', 0, None
        self.handles = []
        for block in self.anchors:
            module = model.blocks[block].attn
            self.handles.append(module.register_forward_pre_hook(self.before_anchor, with_kwargs=True))
            self.handles.append(module.state_ttt.register_forward_hook(self.after_ttt, with_kwargs=True))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def after_ttt(self, module, args, kwargs, result):
        branch = 'clean' if kwargs['clean'] else 'local'
        self.calls[branch] += 1
        if self.phase == 'clean':
            # Anchor modules invoke TTT exactly once, before their memory event.
            stats = result[1]
            self.inner[id(module)] = {key: float(stats[key].double().mean())
                                     for key in ('loss_before', 'loss_after', 'coverage',
                                                 'state_delta_relative', 'eta_trace')}

    def before_anchor(self, module, args, kwargs):
        if self.phase not in ('prefill', 'clean'):
            return
        context = kwargs['ttn_chunk_context']
        index, block = module.index, self.anchors[module.index]
        before_ttt = context.predicted[:, index]

        def observe(event, value):
            if event == 'visual_features':
                observe.q = value[0]
            if event != 'memory':
                return
            from worldttn.core import correct
            from worldttn.evaluation import tensor_sha256
            _, state, k, v, beta, _, write = value
            with torch.autocast(device_type=state.device.type, enabled=False):
                if self.phase == 'prefill':
                    self.references[index] = make_reference(observe.q, k, v, state, self.probe_count)
                    self.prefix_hashes[index] = tensor_sha256(state)
                    rotation = torch.eye(state.shape[-1], device=state.device)
                    metrics = retention_metrics(state, self.references[index], rotation)
                    self.rows.append(dict(chunk=0, anchor=block, **metrics))
                    return
                if index not in self.references or self.transport is None:
                    raise RuntimeError('Clean probe before an observed-prefix reference')
                rotation = self.transport[:, index]
                actual = retention_metrics(state, self.references[index], rotation)
                # Same pre-TTT state and same current features: isolates this update.
                control = correct(before_ttt, k, v, beta, write,
                                  module.config.alpha_s, module.config.eps)[0]
                baseline = retention_metrics(control, self.references[index], rotation)
                expected_active = context.ablation in ('full', 'no-local')
                inner = self.inner.pop(id(module.state_ttt), None)
                if expected_active != (inner is not None):
                    raise AssertionError('Observed StateTTT execution disagrees with ablation')
                if not expected_active:
                    torch.testing.assert_close(state, control, rtol=0, atol=0)
                row = dict(chunk=self.chunk, anchor=block, **actual, inner=inner,
                           correct_only={key: baseline[key] for key in
                                         ('old_key_binding_nmse', 'old_query_response_drift')},
                           ttt_old_binding_delta=actual['old_key_binding_nmse']-baseline['old_key_binding_nmse'],
                           ttt_old_response_delta=actual['old_query_response_drift']-baseline['old_query_response_drift'])
                self.rows.append(row)

        if 'ttn_diagnostic' in kwargs:
            raise RuntimeError('Refusing to replace an existing diagnostic callback')
        kwargs['ttn_diagnostic'] = observe
        return args, kwargs
