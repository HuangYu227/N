"""Four local GPUs: full-episode checkpoint/FSDP gradients against global batch."""
import os
from pathlib import Path
import pytest
import torch
from torch import distributed as dist
from test_training import inputs
from test_legacy_sequence import sequence_model, cached_sana, cached_ffn, text_attention
from worldttn.anchor import configure_train_scope
from worldttn.checkpoint import make_optimizer
from worldttn.distributed import ParallelTraining
from worldttn.training import train_clip, linear_flow_loss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or 'PSI_FULLGRAD_CUDA_OUTPUT' not in os.environ,
                               reason='requires isolated four-GPU acceptance')


def test_fullgradient_four_gpu_global_batch_and_resume(sequence_model):
    rank = int(os.environ['RANK'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', init_method='env://', device_id=device)
    try:
        assert dist.get_world_size() == 4
        def build(checkpoint):
            import copy
            m = configure_train_scope(copy.deepcopy(sequence_model).to(device), 'dit')
            m.ttn_fullgrad_checkpoint = checkpoint
            for block in m.blocks:block.ttn_fullgrad_checkpoint = checkpoint
            return m
        reference, sharded = build(False), build(True)
        parallel = ParallelTraining(sharded, linear_flow_loss, 'fsdp2', activation_offload='cpu-pageable')
        # SGD isolates gradient equivalence; Adam amplifies roundoff in null softmax-key bias gradients.
        optimizer = torch.optim.SGD(sharded.parameters(), lr=.01)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        clean, noise, t, camera = inputs()
        clean = clean.to(device)+rank*.01
        noise, t, camera = [a.to(device) for a in (noise, t, camera)]
        data = [clean, noise, t, camera]
        def train(m, opt, data, parallel=None):
            clean, noise, t, camera = data
            return train_clip(m, clean, torch.zeros(clean.shape[0], 1, 2, 16, device=device), camera,
                opt, linear_flow_loss, t, noise, width=100, height=100, tbptt=0,
                parallel=parallel, cached_chunks=2, activation_offload='cpu-pageable' if parallel else 'none')
        global_data = []
        for value in data:
            gathered = [torch.empty_like(value) for _ in range(4)]
            dist.all_gather(gathered, value)
            global_data.append(torch.cat(gathered))
        expected = train(reference, ref_optimizer, global_data)
        actual = train(sharded, optimizer, data, parallel)
        assert actual['outer_grad_norm'] == pytest.approx(expected['outer_grad_norm'], rel=3e-4, abs=1e-6)
        gradient_differences = {}
        for (name, parameter), (_, other) in zip(sharded.named_parameters(), reference.named_parameters()):
            value = parameter.full_tensor() if hasattr(parameter, 'full_tensor') else parameter
            torch.testing.assert_close(value, other, rtol=3e-5, atol=3e-7, msg=lambda message: name+'\n'+message)
            if other.grad is not None:
                grad = parameter.grad.full_tensor() if hasattr(parameter.grad, 'full_tensor') else parameter.grad
                assert grad is not None
                torch.testing.assert_close(grad, other.grad, rtol=5e-4, atol=3e-6, msg=lambda message: name+'\n'+message)
                gradient_differences[name] = float((grad-other.grad).abs().max())
        if rank == 0:
            print({'gradient_oracle': 'global-batch SGD', 'parameters': len(gradient_differences),
                   'maximum_absolute_gradient_difference': max(gradient_differences.values()),
                   'gradient_norm': actual['outer_grad_norm']}, flush=True)
        # Production Adam state is tested separately against an identical resumed execution.
        optimizer = make_optimizer(sharded)
        train(sharded, optimizer, data, parallel)
        from worldttn.parallel_checkpoint import save_training_checkpoint, restore_training_checkpoint
        directory = Path(os.environ['PSI_FULLGRAD_CUDA_OUTPUT'])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/'last.pt'
        identity = {'tbptt': 0, 'training_protocol': 'dual-psi-clean-fullgrad-v1'}
        save_training_checkpoint(path, parallel, optimizer, 2, {'position': 2}, identity)
        resumed = build(True)
        from worldttn.checkpoint import load_checkpoint
        load_checkpoint(path, resumed)
        rp = ParallelTraining(resumed, linear_flow_loss, 'fsdp2', activation_offload='cpu-pageable')
        ro = make_optimizer(resumed)
        assert restore_training_checkpoint(path, rp, ro, identity) == (2, {'position': 2})
        left, right = train(sharded, optimizer, data, parallel), train(resumed, ro, data, rp)
        assert left['loss'] == pytest.approx(right['loss'], abs=1e-7)
        for (name, a), (_, b) in zip(sharded.named_parameters(), resumed.named_parameters()):
            a = a.to_local() if hasattr(a, 'to_local') else a
            b = b.to_local() if hasattr(b, 'to_local') else b
            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-7, msg=name)
        allocated = []
        for _ in range(10):
            result = train(sharded, optimizer, data, parallel)
            assert result['lifecycle']['boundary'] == 'after episode backward and optimizer'
            torch.cuda.synchronize(device)
            allocated.append(torch.cuda.memory_allocated(device))
        assert max(allocated[-5:])-min(allocated[-5:]) < 2**20, allocated
        if rank == 0: print({'ten_update_allocated_bytes': allocated}, flush=True)
    finally:
        dist.destroy_process_group()
