"""Additive, differentiable low-rank S adaptation with per-head step calibration."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class StateTTT(nn.Module):
    def __init__(self,config,index):
        super().__init__()
        self.policy=config.state_ttt_policy
        h,d,r=config.heads,config.head_dim,config.state_ttt_rank
        # Match the tested rank16 columns and their rank64 orthogonal extension.
        a=torch.linalg.qr(torch.randn(5,h,d,min(16,d),
            generator=torch.Generator().manual_seed(3598))).Q
        if r>a.shape[-1]:
            extra=torch.randn(5,h,d,r-a.shape[-1],generator=torch.Generator().manual_seed(3599))
            for _ in range(2):extra=extra-a@(a.transpose(-1,-2)@extra)
            a=torch.cat((a,torch.linalg.qr(extra).Q),-1)
        self.raw_basis=nn.Parameter(a[index,...,:r].contiguous())
        gain=config.state_ttt_gain
        logit=math.log(gain/(2-gain)) if self.policy=='curvature' else math.log(math.expm1(gain))
        self.step_logits=nn.Parameter(torch.full((2,h),logit))

    def forward(self,pred,k,v,w,mask,*,clean,eps,clip):
        with torch.autocast(device_type=pred.device.type,enabled=False):
            dtype=torch.float64 if pred.dtype==torch.float64 else torch.float32
            pred,k,v,w=[x.to(dtype) for x in (pred,k,v,w)]
            # QR keeps gradient coverage meaningful as outer training learns A.
            a=torch.linalg.qr(self.raw_basis.to(dtype)).Q
            logits=self.step_logits[int(clean)].to(dtype)
            gain=2*logits.sigmoid() if self.policy=='curvature' else F.softplus(logits)
            selected=mask[:,None,:].bool()
            k,v=[torch.where(selected[...,None],x,0.) for x in (k,v)]
            w=torch.where(selected,w,0.)
            z=(eps+(w*v.square().sum(-1)).sum(-1) if not clean
               else 2*k.shape[-1]*mask.sum(-1).clamp_min(1).to(dtype)[:,None])
            error=k@pred-v
            before=(w*error.square().sum(-1)).sum(-1)/z
            h=(2/z)[...,None,None]*(k.transpose(-1,-2)@(w[...,None]*error))
            gradient=a.transpose(-1,-2)@h
            norm=gradient.norm(dim=(-1,-2),keepdim=True)
            scale=(clip/norm.clamp_min(eps)).clamp_max(1)
            ka=k@a
            trace=2*(w*ka.square().sum(-1)).sum(-1)/z
            eta=torch.where(trace>0,gain/(trace+eps),0.) if self.policy=='curvature' else gain.expand_as(trace)
            factor=-eta[...,None,None]*gradient*scale
            delta=a@factor
            updated=pred+delta
            after=(w*(k@updated-v).square().sum(-1)).sum(-1)/z
            return updated,dict(loss_before=before,loss_after=after,effective_eta=eta,
                curvature_trace=trace,eta_trace=eta*trace,clip_scale=scale,
                gradient_norm=norm.squeeze(-1).squeeze(-1),delta_norm=delta.norm(dim=(-1,-2)),
                state_delta_relative=delta.norm(dim=(-1,-2))/pred.norm(dim=(-1,-2)).clamp_min(eps),
                coverage=gradient.square().sum((-1,-2))/h.square().sum((-1,-2)).clamp_min(1e-30))


@torch.no_grad()
def state_stats(values,*,clean,policy):
    """Detach telemetry after the update; replay skips CPU synchronizations."""
    stats={name:float(value.detach().double().mean()) for name,value in values.items()}
    stats.update(branch='clean' if clean else 'local',policy=policy,
        effective_eta_min=float(values['effective_eta'].min()),effective_eta_max=float(values['effective_eta'].max()),
        clip_fraction=float((values['clip_scale']<1).float().mean()),
        per_head={name:value.detach().cpu().tolist() for name,value in values.items()})
    return stats
