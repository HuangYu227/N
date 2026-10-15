"""Numerical acceptance for the two-L20, 121-latent In-Place TTT experiment."""
import math


def _finite(value):
    if isinstance(value, float): return math.isfinite(value)
    if isinstance(value, dict): return all(_finite(item) for item in value.values())
    if isinstance(value, list): return all(_finite(item) for item in value)
    return True


def validate_records(rows, expected):
    if [row['step'] for row in rows] != list(range(1, expected + 1)):
        raise ValueError('Missing, duplicate or unexpected completed steps')
    layers = {3, 7, 11, 15, 19}
    peak, max_ratio, max_gradient_share, min_active = 0., 0., 0., 1.
    for row in rows:
        if not _finite(row) or row['world_size'] != 2 or row['global_batch'] != 2:
            raise ValueError('Nonfinite record or wrong distributed batch')
        if row['config'].get('inplace_reduction') != 'mean' or row['config'].get('inplace_delta_clip') != .01:
            raise ValueError('Wrong mean-loss/shared-cap configuration')
        if not 0 < row['outer_grad_norm'] <= 100:
            raise ValueError('Outer gradient scale failed acceptance')
        if {rank['rank'] for rank in row['ranks']} != {0, 1} or len(row['ranks']) != 2:
            raise ValueError('Missing or duplicate rank records')
        energy = 0.
        for rank in row['ranks']:
            sequence = rank['sequence']
            if (sequence.get('protocol'), sequence.get('temporal_detaches'), sequence.get('backward_calls'),
                sequence.get('ffn_clean_commits'), sequence.get('ffn_prefill_writes'),
                sequence.get('psi_updates'), sequence.get('ffn_reduction'), sequence.get('ffn_delta_clip')) != (
                    'inplace-ffn-clean-fullgrad-v1', 0, 1, 40, 0, False, 'mean', .01):
                raise ValueError('Full-gradient/shared-update contract was not met')
            if len(rank['chunks']) != 40 or rank['commits'] != 41:
                raise ValueError('Wrong horizon or clean commit count')
            peak = max(peak, rank['peak_allocated_bytes'] / 2**30)
            if peak > 42: raise ValueError('Insufficient GPU capacity margin')
            if rank.get('parameter_update', {}).get('missing_core_gradients'):
                raise ValueError('Missing core gradients')
            groups = rank['anchor_gradients']['groups']
            for index in layers:
                for component in ('ttt_conv', 'ttt_proj'):
                    group = groups[f'blocks.{index}.mlp.inplace_ttt.{component}']
                    if group['missing_gradients'] or (component == 'ttt_conv' or row['step'] > 1) and group['grad_norm'] <= 0:
                        raise ValueError('Missing or inactive TTT learning signal')
                    energy += group['grad_norm'] ** 2
            for chunk in rank['chunks']:
                if {layer['block'] for layer in chunk['ffn_ttt']} != layers:
                    raise ValueError('Missing FFN diagnostics')
                for layer in chunk['ffn_ttt']:
                    if max(layer['applied_update_norm']) > .010001 or max(layer['delta_norm_after']) > .40004:
                        raise ValueError('Fast weight exceeded the common update budget')
                    max_ratio = max(max_ratio, *layer['relative_correction'])
            if row['step'] > 1:
                for layer in rank['chunks'][-1]['ffn_ttt']:
                    min_active = min(min_active, *layer['changed_fraction_after_cast'])
                    if min(layer['changed_fraction_after_cast']) <= 0 or min(layer['delta_norm_after']) <= 0:
                        raise ValueError('TTT readout vanished after output casting')
        if max_ratio > .5:
            raise ValueError('TTT correction dominates the original FFN projection')
        # Diagnostics are non-overlapping local shards; outer clipping uses a global 0.5 norm.
        share = energy / min(row['outer_grad_norm'], .5) ** 2
        max_gradient_share = max(max_gradient_share, share)
        if share > .9999:
            raise ValueError('TTT parameters consume virtually the entire global gradient budget')
    return dict(steps=expected, finite=True, full_gradient=True, clean_commits=40,
                gpu_peak_gib=peak, max_relative_correction=max_ratio,
                max_ttt_gradient_energy_share=max_gradient_share,
                min_final_changed_fraction=min_active, shared_update_cap=.01, reduction='mean',
                scope='engineering/numerical acceptance, not a rollout quality claim')
