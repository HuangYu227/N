"""Native two-L20 FSDP2/FP8/offload acceptance, gated to an isolated test launch."""
import copy
import json
import os
import random
from pathlib import Path

import pytest
import torch
from torch import distributed as dist

from test_inplace_sequence import inplace_sequence_model, sequence_model, cached_sana, cached_ffn, text_attention
from test_training import inputs
from worldttn.anchor import configure_train_scope
from worldttn.checkpoint import load_checkpoint, make_optimizer, rng_state, restore_rng
from worldttn.core import ANCHORS
from worldttn.distributed import ParallelTraining
from worldttn.fp8 import configure_training_precision, precision_identity
from worldttn.parallel_checkpoint import save_training_checkpoint, restore_training_checkpoint
from worldttn.training import train_clip, linear_flow_loss


pytestmark = pytest.mark.skipif('INPLACE_TTT_ACCEPTANCE_OUTPUT' not in os.environ or not torch.cuda.is_available(),
    reason='requires an explicit isolated two-L20 native FP8 acceptance launch')


def local(value):
    return value.to_local() if hasattr(value, 'to_local') else value


def test_two_gpu_inplace_fullgrad_fp8_exact_resume(inplace_sequence_model):
    rank = int(os.environ['RANK'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    try:
        assert dist.get_world_size() == 2
        template = copy.deepcopy(inplace_sequence_model)
        from dataclasses import replace
        template.ttn_system.config = replace(template.ttn_system.config,
            inplace_reduction='mean', inplace_delta_clip=.01, inplace_lr=3.)
        for index in ANCHORS:
            template.blocks[index].mlp.inplace_ttt.reduction = 'mean'
            template.blocks[index].mlp.inplace_ttt.lr = 3.
        def build():
            model = configure_train_scope(copy.deepcopy(template).to(device), 'dit')
            for block in model.blocks:
                block.ttn_fullgrad_checkpoint = True
                block.mlp.ttn_activation_checkpointing = True
            report = configure_training_precision(model, 'fp8')
            assert len(report['modules']) == 100
            for index in ANCHORS:
                assert model.blocks[index].mlp.inplace_ttt.ttt_proj.weight.dtype == torch.float32
                assert not hasattr(model.blocks[index].attn, 'state_ttt')
            return model

        model = build()
        parallel = ParallelTraining(model, linear_flow_loss, 'fsdp2', activation_offload='cpu-pageable')
        optimizer = make_optimizer(model)
        initial_targets = {name: local(parameter).detach().clone()
                           for name, parameter in model.named_parameters() if '.mlp.inplace_ttt.' in name}
        clean, noise, times, camera = [value.to(device) for value in inputs()]
        clean = clean + rank * .01
        reports = []

        def update(model, parallel, optimizer):
            torch.cuda.reset_peak_memory_stats(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = train_clip(model, clean, torch.zeros(1, 1, 2, 16, device=device), camera,
                    optimizer, linear_flow_loss, times, noise, width=100, height=100,
                    tbptt=0, cached_chunks=2, parallel=parallel, activation_offload='cpu-pageable', audit_update=True)
            assert result['sequence']['backward_calls'] == 1
            assert result['sequence']['temporal_detaches'] == 0
            assert result['sequence']['psi_updates'] is False
            assert not result['optimizer_updates']['missing_core_gradients']
            gradients = {}
            for index in ANCHORS:
                for name, parameter in model.blocks[index].mlp.inplace_ttt.named_parameters():
                    assert parameter.grad is not None
                    grad = local(parameter.grad)
                    assert torch.isfinite(grad).all() and grad.norm() > 0, (index, name)
                    gradients[f'{index}.{name}'] = float(grad.norm())
                delta = result['runtime'].ffn_fast[index]
                assert delta.dtype == torch.float32 and delta.grad_fn is None and torch.isfinite(delta).all()
            for chunk in result['chunks']:
                json.dumps(chunk['ffn_ttt'], allow_nan=False)
            return result, dict(loss=result['loss'], gradients=gradients,
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(device))

        for _ in range(2):
            _, report = update(model, parallel, optimizer)
            reports.append(report)
        for name, parameter in model.named_parameters():
            if name in initial_targets:
                assert torch.count_nonzero(local(parameter).detach() - initial_targets[name]) > 0, name
        output = Path(os.environ['INPLACE_TTT_ACCEPTANCE_OUTPUT'])
        identity = {'ttt_backend': 'inplace-ffn', 'compute_precision': precision_identity('fp8'),
                    'tbptt': 0, 'activation_offload': 'cpu-pageable'}
        cursor = {'position': 2, 'epoch': 0, 'sampling_seed': 3407}
        save_training_checkpoint(output / 'last.pt', parallel, optimizer, 2, cursor, identity)
        saved_rng = rng_state()
        expected_cpu = torch.rand(7)
        expected_cuda = torch.rand(7, device=device)
        expected_python = random.random()
        restore_rng(saved_rng)
        other = build()
        load_checkpoint(output / 'last.pt', other)
        resumed = ParallelTraining(other, linear_flow_loss, 'fsdp2', activation_offload='cpu-pageable')
        other_optimizer = make_optimizer(other)
        assert restore_training_checkpoint(output / 'last.pt', resumed, other_optimizer, identity) == (2, cursor)
        torch.testing.assert_close(torch.rand(7), expected_cpu, rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(7, device=device), expected_cuda, rtol=0, atol=0)
        assert random.random() == expected_python
        restore_rng(saved_rng)
        uninterrupted, report = update(model, parallel, optimizer)
        reports.append(report)
        restore_rng(saved_rng)
        restarted, resume_report = update(other, resumed, other_optimizer)
        assert uninterrupted['loss'] == pytest.approx(restarted['loss'], abs=1e-7)
        for (name, first), (other_name, second) in zip(model.named_parameters(), other.named_parameters()):
            assert name == other_name
            torch.testing.assert_close(local(first), local(second), rtol=3e-5, atol=3e-7, msg=name)
            first_state, second_state = optimizer.state[first], other_optimizer.state[second]
            assert first_state.keys() == second_state.keys()
            for key in first_state:
                torch.testing.assert_close(local(first_state[key]), local(second_state[key]),
                                           rtol=3e-5, atol=3e-7, msg=f'{name}.{key}')
        record = {'rank': rank, 'steps': reports, 'resumed_step3': resume_report,
                  'cursor': cursor, 'rng_restored': True, 'status': 'passed'}
        records = [None, None]
        dist.all_gather_object(records, record)
        if rank == 0:
            (output / 'acceptance.json').write_text(json.dumps(records, indent=2, allow_nan=False), encoding='utf-8')
        print('INPLACE_TTT_FP8_ACCEPTANCE_PASSED', rank, flush=True)
    finally:
        dist.destroy_process_group()
