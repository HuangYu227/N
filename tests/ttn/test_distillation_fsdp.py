"""Run with torchrun --nproc_per_node=1 or 2 on the selected CUDA devices."""
import copy
import os
import traceback
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn

from worldttn.distillation.adapters import shard, clip_grad, update_norms
from worldttn.fullgradient import checkpoint_tree
from worldttn.distillation.checkpoint import EMA, save, restore, export_best
from worldttn.distillation.adapters import install, parameters, GradientAccumulator
from worldttn.distillation.sampler import Transactions
from worldttn.training import activation_storage
from test_distillation import sequence_model, cached_sana, cached_ffn, text_attention, make_case


@pytest.fixture(scope='module', autouse=True)
def distributed_group():
    if 'RANK' not in os.environ:
        pytest.skip('requires one/two CUDA torchrun ranks')
    if not torch.cuda.is_available() or int(os.environ.get('WORLD_SIZE','1')) not in (1,2):
        pytest.fail('torchrun CUDA acceptance requires one/two usable CUDA ranks')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=3))
    yield
    dist.destroy_process_group()


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(16), requires_grad=False)
        self.proj = nn.Linear(16, 16)

    def forward(self, value):
        return (self.proj(value) + self.scale_shift_table).tanh()


class RecurrentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(3)])

    def forward(self, value, state):
        for block in self.blocks:
            value = block(value + state)
        return value, state * .5 + value.mean(0, keepdim=True)


def episode(model, inputs):
    state = torch.zeros_like(inputs[0, :1])
    outputs = []
    for value in inputs:
        def transaction(x, incoming):
            def predict(x, incoming):
                return model(x, incoming)[0]
            prediction = checkpoint_tree(predict, (x, incoming))
            _, outgoing = model(x * .8, incoming)
            return prediction, outgoing
        value, state = checkpoint_tree(transaction, (value, state))
        outputs.append(value)
    return torch.stack(outputs), state


@pytest.mark.parametrize('accumulation', (1,2))
def test_repeated_fsdp_recompute_matches_unsharded_updates(tmp_path,accumulation):
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    try:
        torch.manual_seed(97)
        reference = RecurrentModel().to(device)
        actual = shard(copy.deepcopy(reference), cpu_offload=True)
        torch.manual_seed(101 + dist.get_rank())
        inputs = [torch.randn(4, 2, 16, device=device) for _ in range(accumulation)]
        reference_opt = torch.optim.AdamW(reference.parameters(), lr=1e-4)
        actual_opt = torch.optim.AdamW([{'params': list(actual.parameters()), 'name': 'test'}], lr=1e-4)
        models = {'student': actual}
        optimizers = {'student': actual_opt}
        ema = EMA(actual)
        folder = [str(tmp_path) if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(folder, src=0)
        expected_resume = None
        for iteration in range(3):
            if iteration == 2:
                counts = restore(folder[0], models, optimizers, ema, {'test': 'recurrent'})
                assert counts == {'student': 1, 'critic': 0}
                reference.load_state_dict(reference_saved)
                reference_opt.load_state_dict(reference_opt_saved)
            reference_opt.zero_grad(set_to_none=True)
            actual_opt.zero_grad(set_to_none=True)
            accumulator=GradientAccumulator(actual)
            for microbatch in inputs:
                expected, expected_state = episode(reference, microbatch)
                observed, observed_state = episode(actual, microbatch)
                # CPU/GPU FP32 AdamW can differ by one ULP after an update.
                torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(observed_state, expected_state, rtol=1e-6, atol=1e-7)
                ((expected.square().mean() + expected_state.square().mean())/accumulation).backward()
                ((observed.square().mean() + observed_state.square().mean())/accumulation).backward()
                accumulator.capture()
            accumulator.apply()
            for name, parameter in actual.named_parameters():
                original = dict(reference.named_parameters())[name]
                if original.grad is None:
                    assert parameter.grad is None
                else:
                    dist.all_reduce(original.grad)
                    original.grad.div_(dist.get_world_size())
                    torch.testing.assert_close(parameter.grad.to(device).full_tensor(), original.grad,
                                               rtol=1e-5, atol=1e-7)
            norm = clip_grad(actual, .25)
            expected_norm = sum(p.grad.square().sum() for p in reference.parameters() if p.grad is not None).sqrt()
            torch.testing.assert_close(torch.tensor(norm, device=device), expected_norm, rtol=1e-5, atol=1e-7)
            for parameter in reference.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_((.25/(expected_norm+1e-6)).clamp_max(1))
            before = {n:p.to_local().detach().clone() for n,p in parameters(actual).items()}
            reference_before = {n:p.detach().clone() for n,p in parameters(reference).items()}
            reference_opt.step()
            actual_opt.step()
            measured = update_norms(actual, before)
            expected_delta = sum((p-reference_before[n]).square().sum()
                                 for n,p in parameters(reference).items()).sqrt()
            torch.testing.assert_close(torch.tensor(measured['lora'],device=device),expected_delta,
                                       rtol=1e-4,atol=1e-7)
            assert measured['state_ttt'] == 0.
            ema.update(actual)
            for name, parameter in actual.named_parameters():
                torch.testing.assert_close(parameter.to(device).full_tensor(), dict(reference.named_parameters())[name],
                                           rtol=1e-5, atol=1e-7)
            if iteration == 0:
                reference_saved = copy.deepcopy(reference.state_dict())
                reference_opt_saved = copy.deepcopy(reference_opt.state_dict())
                save(folder[0], models, optimizers, ema, {'student': 1, 'critic': 0},
                     {'test': 'recurrent'})
                best = Path(folder[0])/'best.pt'
                export_best(best, models, {'test': 'recurrent'}, {'student': 1, 'critic': 0}, {})
                if dist.get_rank() == 0:
                    saved = torch.load(best, map_location='cpu', weights_only=True)['weights']['student']
                    for name, parameter in reference.named_parameters():
                        if parameter.requires_grad:
                            torch.testing.assert_close(saved[name], parameter.detach().cpu(), rtol=1e-5, atol=1e-7)
            elif iteration == 1:
                expected_resume = {n: p.to_local().detach().clone() for n, p in actual.named_parameters()}
            else:
                for name, parameter in actual.named_parameters():
                    torch.testing.assert_close(parameter.to_local(), expected_resume[name], rtol=0, atol=0)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        dist.barrier()


def test_ttn_state_and_lora_recompute_gradients(sequence_model):
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    try:
        reference, batch, noise = make_case(sequence_model)
        reference = reference.to(device)
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
        noise = noise.to(device)
        install(reference, 2)
        actual = shard(copy.deepcopy(reference), cpu_offload=True)
        for scoring in (True, False):
            reference.zero_grad(set_to_none=True)
            actual.zero_grad(set_to_none=True)
            results = []
            for model in (reference, actual):
                with activation_storage('cpu-pageable', device=device):
                    transaction = Transactions(model, batch, checkpointed=True)
                    if scoring:
                        value = transaction.score(noise, torch.full((1, 7), .5, device=device), noise)
                    else:
                        value, _ = transaction.generate(noise, 2, 9.8, 4.5)
                    value[:, :, -3:].square().mean().backward()
                    results.append(value.detach())
            torch.testing.assert_close(*results, rtol=1e-5, atol=1e-7)
            for name, parameter in parameters(actual).items():
                original = parameters(reference)[name]
                if original.grad is None:
                    assert parameter.grad is None
                else:
                    dist.all_reduce(original.grad)
                    original.grad.div_(dist.get_world_size())
                    assert parameter.grad is not None, name
                    torch.testing.assert_close(parameter.grad.to(device).full_tensor(), original.grad,
                                               rtol=1e-4, atol=1e-7, msg=name)
            assert all(p.grad is not None for n, p in parameters(actual).items() if '.state_ttt.' in n)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        dist.barrier()
