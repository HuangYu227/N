"""Exercise the actual GDN gate methods without importing optional CUDA kernels."""
import ast
import copy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_gdn_gate_lora_zero_parity_input_gradients_and_all_adapter_branches(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('requires CUDA')
    source=Path(__file__).resolve().parents[2]/'diffusion/model/nets/sana_gdn_blocks.py'
    nodes=ast.parse(source.read_text(encoding='utf-8')).body
    helpers=[n for n in nodes if isinstance(n,ast.FunctionDef)
             and n.name in ('_compute_frame_gates','_apply_output_gate')]
    for node in helpers: node.decorator_list=[]
    gdn=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name=='GDN')
    methods=[n for n in gdn.body if isinstance(n,ast.FunctionDef)
             and n.name in ('_compute_frame_gates','_apply_output_gate')]
    namespace={'torch':torch,'nn':nn,'F':F}
    exec(compile(ast.Module(body=helpers+methods,type_ignores=[]),str(source),'exec'),namespace)
    # The bound methods and compiled-helper equivalents need distinct global names.
    helper_namespace={'torch':torch,'nn':nn,'F':F}
    exec(compile(ast.Module(body=helpers,type_ignores=[]),str(source),'exec'),helper_namespace)
    gate_method=namespace['_apply_output_gate'];frame_method=namespace['_compute_frame_gates']
    namespace.update(helper_namespace)
    class Gates(nn.Module):
        _apply_output_gate=gate_method
        _compute_frame_gates=frame_method
        def __init__(self):
            super().__init__()
            self.heads=2;self.use_output_gate=True
            self.beta_proj=nn.Linear(8,2);self.gate_proj=nn.Linear(8,2)
            self.output_gate=nn.Linear(8,8)
            self.A_log=nn.Parameter(torch.zeros(2));self.dt_bias=nn.Parameter(torch.full((2,),-.5))
        def forward(self,x):
            beta,decay=self._compute_frame_gates(x,(3,2,2))
            return beta,decay,self._apply_output_gate(x*.7,x)
    torch.manual_seed(7)
    reference=Gates().to(device).requires_grad_(False)
    actual=copy.deepcopy(reference)
    from peft import LoraConfig,inject_adapter_in_model
    inject_adapter_in_model(LoraConfig(r=2,lora_alpha=2,lora_dropout=0,
        target_modules=['beta_proj','gate_proj','output_gate']),actual)
    x=torch.randn(2,12,8,device=device,requires_grad=True)
    y=x.detach().clone().requires_grad_(True)
    expected=reference(x);observed=actual(y)
    for a,b in zip(expected,observed): torch.testing.assert_close(a,b,rtol=0,atol=0)
    sum(v.square().mean() for v in expected).backward()
    sum(v.square().mean() for v in observed).backward()
    torch.testing.assert_close(x.grad,y.grad,rtol=1e-5,atol=1e-7)
    selected=[p for name,p in actual.named_parameters() if 'lora_' in name]
    assert len(selected)==6 and all(p.grad is not None for p in selected)
    up=[p for name,p in actual.named_parameters() if 'lora_B' in name]
    assert len(up)==3 and all(p.grad.norm()>0 for p in up)
    with torch.no_grad():
        for p in up:p.add_(.1)
    assert all(not torch.equal(a,b) for a,b in zip(expected,actual(y)))
