"""Selective, stateless FP8 GEMMs; recurrent state and master weights stay FP32."""
from types import SimpleNamespace

import torch
from torch import nn


RECIPE = 'tensorwise-dynamic-e4m3-e5m2-v1'


def precision_identity(precision):
    if precision not in ('bf16', 'fp8'):
        raise ValueError('training precision must be bf16 or fp8')
    return {'precision': precision, 'backend': 'torchao', 'recipe': RECIPE,
            'master_dtype': 'float32', 'core_dtype': 'float32',
            'fp8_all_gather': False} if precision == 'fp8' else {'precision': 'bf16'}


def _check_device(model):
    parameter = next(model.parameters())
    if parameter.device.type != 'cuda' or torch.cuda.get_device_capability(parameter.device) < (8, 9):
        raise RuntimeError('native FP8 training requires a CUDA GPU with compute capability >= 8.9')


def _load_backend():
    try:
        from importlib.metadata import version
        from torchao.float8.config import Float8LinearConfig, Float8GemmConfig
        from torchao.float8.float8_linear import Float8Linear, matmul_with_hp_or_float8_args
    except ImportError as error:
        raise RuntimeError('FP8 requires compatible torchao; validated dependency is torchao==0.14.1 with PyTorch 2.9') from error
    config = Float8LinearConfig(gemm_config_output=Float8GemmConfig(use_fast_accum=False),
                               pad_inner_dim=True, enable_fsdp_float8_all_gather=False)
    # Current scaling has no amax history for a checkpoint replay to mutate.
    prototype = Float8Linear(16, 16, device='meta', config=config)
    mm = prototype.linear_mm_config
    return SimpleNamespace(linear=Float8Linear, config=config, version=version('torchao'),
        matmul=lambda x, w: matmul_with_hp_or_float8_args.apply(x, w, mm, config))


class FP8PointwiseConv2d(nn.Conv2d):
    """The same 1x1 projection and parameter layout, using TorchAO's FP8 GEMM."""
    @classmethod
    def from_float(cls, original, backend):
        value = cls(original.in_channels, original.out_channels, 1,
                    bias=original.bias is not None, device='meta')
        value.weight, value.bias = original.weight, original.bias
        value.fp8_backend = backend
        return value.train(original.training)

    def forward(self, x):
        batch, _, height, width = x.shape
        features = x.permute(0, 2, 3, 1).reshape(-1, self.in_channels)
        if torch.is_autocast_enabled(x.device.type):
            features = features.to(torch.get_autocast_dtype(x.device.type))
        output = self.fp8_backend.matmul(features, self.weight.flatten(1).t())
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        # The original ConvLayer can apply SiLU in place; custom-function views cannot.
        return output.reshape(batch, height, width, self.out_channels).permute(0, 3, 1, 2).clone()


def configure_training_precision(model, precision='bf16'):
    report = precision_identity(precision)
    if precision == 'bf16':
        return report
    targets = []
    for index, block in enumerate(getattr(model, 'blocks', ())):
        cross = getattr(block, 'cross_attn', None)
        for name in ('q_linear', 'kv_linear', 'proj'):
            module = getattr(cross, name, None)
            if type(module) is nn.Linear:
                targets.append((f'blocks.{index}.cross_attn.{name}', cross, name, module))
        mlp = getattr(block, 'mlp', None)
        for name in ('inverted_conv', 'point_conv'):
            parent = getattr(mlp, name, None)
            module = getattr(parent, 'conv', None)
            if type(module) is nn.Conv2d:
                if (module.kernel_size, module.stride, module.padding, module.dilation,
                    module.groups, module.padding_mode) != ((1, 1), (1, 1), (0, 0), (1, 1), 1, 'zeros'):
                    raise ValueError(f'FP8 requires an ordinary 1x1 projection: blocks.{index}.mlp.{name}.conv')
                targets.append((f'blocks.{index}.mlp.{name}.conv', parent, 'conv', module))
    if not targets:
        raise ValueError('no eligible SANA FFN/text projection modules for FP8')
    for name, _, _, module in targets:
        if any(size % 16 for size in module.weight.shape[:2]):
            raise ValueError(f'FP8 projection dimensions must be a multiple of 16: {name}')
        if module.weight.dtype != torch.float32:
            raise ValueError(f'FP8 training requires FP32 master weights: {name}')
    _check_device(model)
    backend = _load_backend()
    for _, parent, name, module in targets:
        converted = (backend.linear.from_float(module, config=backend.config).train(module.training)
                     if type(module) is nn.Linear else FP8PointwiseConv2d.from_float(module, backend))
        setattr(parent, name, converted)
    report.update(torchao_version=backend.version, modules=[name for name, *_ in targets],
                  recurrent_attention='unchanged', scaling_state='none')
    model.training_precision_report = report
    return report
