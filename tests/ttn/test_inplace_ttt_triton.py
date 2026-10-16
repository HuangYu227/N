import pytest
import torch
from worldttn.inplace_ttt_triton import available, read_add, update

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_read_add_equivalence():
    if not available(): pytest.skip("Triton unavailable")
    torch.manual_seed(3407)
    B,N,C,H=2,257,37,113
    z=torch.randn(B,N,H,device="cuda",dtype=torch.bfloat16)
    d=torch.randn(B,C,H,device="cuda",dtype=torch.float32)*.01
    b=torch.randn(B,N,C,device="cuda",dtype=torch.bfloat16)
    got=read_add(z,d,b)
    ref=(b.float()+torch.bmm(z.float(),d.transpose(1,2))).to(b.dtype)
    torch.testing.assert_close(got,ref,rtol=2e-3,atol=2e-3)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_update_equivalence():
    if not available(): pytest.skip("Triton unavailable")
    torch.manual_seed(3407)
    B,N,C,H=2,257,37,113
    v=torch.randn(B,N,C,device="cuda",dtype=torch.float32)
    z=torch.randn(B,N,H,device="cuda",dtype=torch.float32)
    d=torch.randn(B,C,H,device="cuda",dtype=torch.float32)
    s=torch.tensor([.01,.02],device="cuda")
    got=update(v,z,d,s)
    ref=d+s[:,None,None]*torch.bmm(v.transpose(1,2),z)
    torch.testing.assert_close(got,ref,rtol=2e-3,atol=2e-3)
