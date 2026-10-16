import argparse
import statistics
import time
import torch

from worldttn.inplace_ttt_triton import available, read_add, update

def median_ms(fn, warmup=10, repeats=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs=[]
    for _ in range(repeats):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        xs.append(a.elapsed_time(b))
    return statistics.median(xs)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--n",type=int,default=880)
    p.add_argument("--b",type=int,default=1)
    p.add_argument("--c",type=int,default=2240)
    p.add_argument("--h",type=int,default=6720)
    args=p.parse_args()
    if not torch.cuda.is_available() or not available():
        raise SystemExit("CUDA + Triton required")
    torch.manual_seed(3407)
    z=torch.randn(args.b,args.n,args.h,device="cuda",dtype=torch.bfloat16)
    delta=torch.randn(args.b,args.c,args.h,device="cuda",dtype=torch.float32)*.01
    base=torch.randn(args.b,args.n,args.c,device="cuda",dtype=torch.bfloat16)
    v=torch.randn(args.b,args.n,args.c,device="cuda",dtype=torch.float32)
    scale=torch.full((args.b,),.01,device="cuda")
    ref_read=lambda:(base.float()+torch.bmm(z.float(),delta.transpose(1,2))).to(base.dtype)
    ref_up=lambda:delta+scale[:,None,None]*torch.bmm(v.transpose(1,2),z.float())
    tr_read=lambda:read_add(z,delta,base)
    tr_up=lambda:update(v,z.float(),delta,scale)
    # Warm once to compile/autotune before comparing.
    tr_read(); tr_up(); torch.cuda.synchronize()
    r1=median_ms(ref_read); t1=median_ms(tr_read)
    r2=median_ms(ref_up); t2=median_ms(tr_up)
    print({"read_reference_ms":r1,"read_triton_ms":t1,
           "read_speedup":r1/t1,"update_reference_ms":r2,"update_triton_ms":t2,
           "update_speedup":r2/t2})
if __name__=="__main__": main()
