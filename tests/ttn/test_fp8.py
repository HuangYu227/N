"""CPU wiring tests; CUDA tests exercise the real FP8 backend separately."""
import copy
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def test_fp8_training_policy_exists():
    assert importlib.util.find_spec('worldttn.fp8') is not None


def model():
    m = nn.Module()
    block = nn.Module()
    block.cross_attn = nn.Module()
    for name in ('q_linear', 'kv_linear', 'proj'):
        setattr(block.cross_attn, name, nn.Linear(16, 16))
    block.mlp = nn.Module()
    for name in ('inverted_conv', 'point_conv'):
        layer = nn.Module()
        layer.conv = nn.Conv2d(16, 16, 1, bias=True)
        setattr(block.mlp, name, layer)
    block.mlp.t_conv = nn.Conv2d(16, 16, (3, 1), padding=(1, 0), groups=16)
    block.attn = nn.Sequential(nn.Linear(16, 16))
    m.blocks = nn.ModuleList([block])
    m.ttn_system = nn.Sequential(nn.Linear(16, 16))
    return m


@pytest.fixture
def wiring_backend(monkeypatch):
    from worldttn import fp8

    class Linear(nn.Linear):
        @classmethod
        def from_float(cls, original, config):
            value = cls(original.in_features, original.out_features,
                        bias=original.bias is not None, device='meta')
            value.weight, value.bias = original.weight, original.bias
            return value

    backend = SimpleNamespace(linear=Linear, config=object(), version='test-wiring',
                              matmul=lambda x, w: x @ w)
    monkeypatch.setattr(fp8, '_load_backend', lambda: backend)
    monkeypatch.setattr(fp8, '_check_device', lambda model: None)
    return fp8


def test_bf16_does_not_import_or_change_modules(monkeypatch):
    from worldttn import fp8
    m = model()
    before = dict(m.named_modules())
    monkeypatch.setattr(fp8, '_load_backend', lambda: pytest.fail('BF16 imported FP8 backend'))
    assert fp8.configure_training_precision(m, 'bf16')['precision'] == 'bf16'
    assert dict(m.named_modules()) == before


def test_selection_preserves_parameters_checkpoint_and_core(wiring_backend):
    m = model().eval()
    original = copy.deepcopy(m)
    parameters = dict(m.named_parameters())
    attention, system, temporal = m.blocks[0].attn, m.ttn_system, m.blocks[0].mlp.t_conv
    report = wiring_backend.configure_training_precision(m, 'fp8')
    assert len(report['modules']) == 5
    assert m.blocks[0].attn is attention and m.ttn_system is system
    assert m.blocks[0].mlp.t_conv is temporal
    assert list(m.named_parameters()) == list(parameters.items())
    assert m.state_dict().keys() == original.state_dict().keys()
    for name, value in m.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[name], rtol=0, atol=0)
    assert all(p.dtype == torch.float32 for p in m.parameters())
    assert not m.blocks[0].cross_attn.q_linear.training


def test_pointwise_adapter_keeps_conv_layout_bias_and_gradients(wiring_backend):
    m = model()
    original = copy.deepcopy(m.blocks[0].mlp.point_conv.conv)
    wiring_backend.configure_training_precision(m, 'fp8')
    changed = m.blocks[0].mlp.point_conv.conv
    x = torch.randn(2, 16, 3, 5, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    actual, expected = changed(x), original(other)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().mean().backward(); expected.square().mean().backward()
    for a, b in [(x.grad, other.grad), (changed.weight.grad, original.weight.grad),
                 (changed.bias.grad, original.bias.grad)]:
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_incompatible_shape_rejected_before_mutation(wiring_backend):
    m = model()
    m.blocks[0].cross_attn.proj = nn.Linear(16, 17)
    before = dict(m.named_modules())
    with pytest.raises(ValueError, match='multiple of 16'):
        wiring_backend.configure_training_precision(m, 'fp8')
    assert dict(m.named_modules()) == before


def test_custom_gemm_output_can_feed_original_inplace_silu(wiring_backend):
    m = model()
    m.blocks[0].mlp.inverted_conv.conv = nn.Conv2d(16,16,1,bias=False)
    wiring_backend.configure_training_precision(m,'fp8')
    conv=m.blocks[0].mlp.inverted_conv.conv

    class ViewMM(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,w):
            ctx.save_for_backward(x,w)
            return (x@w).view(x.shape[0],w.shape[1])

        @staticmethod
        def backward(ctx,gradient):
            x,w=ctx.saved_tensors
            return gradient@w.t(),x.t()@gradient

    conv.fp8_backend.matmul=ViewMM.apply
    x=torch.randn(2,16,1,1,requires_grad=True)
    output=nn.SiLU(inplace=True)(conv(x))
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(conv.weight.grad).all()


def test_unknown_precision_and_empty_target_rejected():
    from worldttn.fp8 import configure_training_precision
    with pytest.raises(ValueError, match='precision'):
        configure_training_precision(model(), 'fp4')
    with pytest.raises(ValueError, match='no eligible'):
        configure_training_precision(nn.Linear(16, 16), 'fp8')


def test_fp8_resume_cannot_silently_change_compute_precision():
    from worldttn.fp8 import precision_identity
    from worldttn.parallel_checkpoint import validate_training_checkpoint
    config = {'tbptt': 0, 'execution': {'core_backend': 'reference', 'psi_backend': 'reference'},
              'compute_precision': precision_identity('fp8')}
    payload = {'distributed': {'format': 'TTN-parallel-resume-v1', 'mode': 'fsdp2',
                              'world_size': 2, 'resume_dir': 'last-resume-1', 'training_config': config}}
    validate_training_checkpoint(payload, 'fsdp2', 2, config)
    changed = {k: v for k, v in config.items() if k != 'compute_precision'}
    with pytest.raises(ValueError, match='configuration mismatch'):
        validate_training_checkpoint(payload, 'fsdp2', 2, changed, benchmark=True)


def test_cli_precision_override_preserves_default_resume_identity():
    from dataclasses import dataclass
    from worldttn.cli import _training_identity

    @dataclass
    class Scheduler:
        noise: str = 'flow'

    config=SimpleNamespace(scheduler=Scheduler(),model=None,text_encoder=None)
    args=SimpleNamespace(seed=3407,batch_file='batch.pt',train_scope='dit',train_precision=None)
    settings={'training_latent_frames':121,'training_protocol':'dual-psi-clean-fullgrad-v1'}
    assert 'compute_precision' not in _training_identity(args,config,settings,0)
    settings['training_precision']='fp8'
    assert _training_identity(args,config,settings,0)['compute_precision']['precision']=='fp8'
    args.train_precision='bf16'
    assert 'compute_precision' not in _training_identity(args,config,settings,0)
