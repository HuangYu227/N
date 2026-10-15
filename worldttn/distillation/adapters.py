"""Independent PEFT adapters and native FP32 StateTTT parameters per role."""
import torch
from torch import nn


def install(model, rank=128):
    from peft import LoraConfig, inject_adapter_in_model
    targets = []
    for name,module in model.named_modules():
        if not name.startswith('blocks.') or '.state_ttt.' in name:
            continue
        if isinstance(module,nn.Linear) or (isinstance(module,nn.Conv2d)
                and module.kernel_size == (1,1) and module.groups == 1 and '.mlp.' in name):
            targets.append(name)
    if not targets:
        raise ValueError('No TTN projection modules selected')
    model.requires_grad_(False)
    inject_adapter_in_model(LoraConfig(r=rank,lora_alpha=rank,lora_dropout=0.,
                                      target_modules=sorted(targets)),model)
    # SANA's spatial autochunker reads the wrapped convolution geometry.
    # PEFT forwards weights but does not expose these immutable attributes.
    for name in targets:
        wrapped = model.get_submodule(name)
        base = wrapped.get_base_layer()
        if isinstance(base,nn.Conv2d):
            for field in ('in_channels','out_channels','kernel_size','stride',
                          'padding','dilation','groups','padding_mode'):
                setattr(wrapped,field,getattr(base,field))
    state = []
    for name,param in model.named_parameters():
        if '.state_ttt.' in name:
            param.requires_grad_(True);state.append(name)
    if len(state) != 10:
        raise ValueError('Expected exactly ten StateTTT basis/gain tensors')
    return targets


def parameters(model):
    return {name:p for name,p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def load_adapter_weights(model, saved):
    current = parameters(model)
    if current.keys() != saved.keys():
        raise ValueError('Adapter schema changed')
    for name, param in current.items():
        if param.shape != saved[name].shape:
            raise ValueError('Adapter shape changed: '+name)
        param.copy_(saved[name].to(param.device))


def optimizer(model, lr):
    grouped = [dict(params=[p for n,p in parameters(model).items() if '.state_ttt.' not in n],lr=lr,name='lora'),
               dict(params=[p for n,p in parameters(model).items() if '.state_ttt.' in n],lr=lr*.1,name='state_ttt')]
    return torch.optim.AdamW(grouped,betas=(0.,.999),weight_decay=0.,foreach=False)


def shard(model):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from worldttn.distributed import _expose_clean_state, _restore_block_output
    from worldttn.anchor import TTNAnchor
    mesh = init_device_mesh('cuda',(torch.distributed.get_world_size(),))
    policy = MixedPrecisionPolicy(param_dtype=None,reduce_dtype=torch.float32,cast_forward_inputs=False)
    for block in model.blocks:
        if isinstance(block.attn,TTNAnchor):
            block.register_forward_hook(_expose_clean_state,with_kwargs=True)
        fully_shard(block,mesh=mesh,mp_policy=policy,reshard_after_forward=True)
        if isinstance(block.attn,TTNAnchor):
            block.register_forward_hook(_restore_block_output)
    fully_shard(model,mesh=mesh,mp_policy=policy,reshard_after_forward=False)
    return model


def clip_grad(model, maximum=1.):
    params = list(parameters(model).values())
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    squared = torch.zeros((),dtype=torch.float32,device=next(model.parameters()).device)
    for p in params:
        if p.grad is not None:
            g = p.grad.to_local() if hasattr(p.grad,'to_local') else p.grad
            if not torch.isfinite(g).all(): raise FloatingPointError('Nonfinite adapter gradient')
            squared += g.float().square().sum()/(1 if hasattr(p.grad,'to_local') else world)
    if world > 1: torch.distributed.all_reduce(squared)
    norm = squared.sqrt()
    if not torch.isfinite(norm): raise FloatingPointError('Nonfinite adapter gradient norm')
    for p in params:
        if p.grad is not None:
            g = p.grad.to_local() if hasattr(p.grad,'to_local') else p.grad
            g.mul_((maximum/(norm+1e-6)).clamp_max(1))
    return float(norm)
