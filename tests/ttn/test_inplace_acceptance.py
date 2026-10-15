import copy

import pytest

from worldttn.inplace_acceptance import validate_records


def records():
    result = []
    for step in (1, 2):
        ranks = []
        for rank in (0, 1):
            groups = {f'blocks.{i}.mlp.inplace_ttt.{name}': dict(missing_gradients=0,
                      grad_norm=.02 if name == 'ttt_conv' or step > 1 else 0.)
                      for i in (3, 7, 11, 15, 19) for name in ('ttt_conv', 'ttt_proj')}
            chunk = dict(ffn_ttt=[dict(block=i, applied_update_norm=[.001], delta_norm_after=[.02],
                relative_correction=[.05], changed_fraction_after_cast=[.1]) for i in (3, 7, 11, 15, 19)])
            ranks.append(dict(rank=rank, sequence=dict(protocol='inplace-ffn-clean-fullgrad-v1',
                temporal_detaches=0, backward_calls=1, ffn_clean_commits=40, ffn_prefill_writes=0,
                psi_updates=False, ffn_reduction='mean', ffn_delta_clip=.01),
                chunks=[copy.deepcopy(chunk) for _ in range(40)], commits=41,
                peak_allocated_bytes=37 * 2**30, anchor_gradients=dict(groups=groups)))
        result.append(dict(step=step, world_size=2, global_batch=2, outer_grad_norm=4.,
            config=dict(inplace_reduction='mean', inplace_delta_clip=.01), ranks=ranks))
    return result


def test_acceptance_rejects_finite_but_exploding_updates_and_dead_learning():
    rows = records()
    assert validate_records(rows, 2)['max_relative_correction'] == .05
    for field, value, message in (('relative_correction', [214.], 'dominates'),
                                  ('applied_update_norm', [.02], 'budget'),
                                  ('changed_fraction_after_cast', [0.], 'vanished')):
        bad = copy.deepcopy(rows)
        bad[-1]['ranks'][0]['chunks'][-1]['ffn_ttt'][-1][field] = value
        with pytest.raises(ValueError, match=message): validate_records(bad, 2)
    rows[-1]['ranks'][0]['anchor_gradients']['groups']['blocks.19.mlp.inplace_ttt.ttt_proj']['grad_norm'] = 0
    with pytest.raises(ValueError, match='learning signal'): validate_records(rows, 2)
