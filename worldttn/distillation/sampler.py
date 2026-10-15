"""Pure chunk transactions: every solver step and clean write remains differentiable."""
from dataclasses import dataclass
import torch

from worldttn.core import ANCHORS
from worldttn.session import TTNSession, repeat_batch, validate_clean_output
from worldttn.fullgradient import checkpoint_tree, _RECOMPUTING
from worldttn.training import chunk_ranges


def finite(value, name):
    if not torch.isfinite(value).all():
        raise FloatingPointError('Nonfinite ' + name)
    return value


def cfg_velocity(value, scale):
    uncond, cond = value.chunk(2, 0)
    return uncond + scale * (cond - uncond)


def flow_x0(noisy, sigma, velocity):
    while sigma.ndim < noisy.ndim:
        sigma = sigma.unsqueeze(-1)
    return noisy.float() - sigma * velocity.float()


def shifted_schedule(steps, shift, device):
    # Use the actual SANA scheduler, including its 1/1000 terminal grid point.
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler(shift=float(shift))
    scheduler.set_timesteps(int(steps), device=device)
    return scheduler.sigmas.to(device), scheduler.timesteps.to(device)


@dataclass
class Snapshot:
    state: torch.Tensor
    psi: torch.Tensor
    cache: list
    lengths: dict


class Transactions:
    def __init__(self, model, batch, *, checkpointed=True, collect=False, ablation='full'):
        self.model, self.batch = model, batch
        self.checkpointed, self.collect, self.ablation = checkpointed, collect, ablation
        self.frames = batch['camera_conditions'].shape[1]
        self.records = []
        self.y = torch.cat((batch['uncondition'], batch['y']), 0)
        mask = batch.get('mask')
        self.mask = mask.reshape(mask.shape[0], -1) if mask is not None else None

    def session(self, snapshot=None, start=0):
        b = self.batch
        session = TTNSession(self.model, b['camera_conditions'], b['width'], b['height'],
            extras={k: b[k] for k in ('chunk_plucker',) if k in b},
            collect_local_stats=self.collect and not _RECOMPUTING.get(), live_cache=True,
            ablation=self.ablation)
        runtime = session.reset(self.y.shape[0])
        if snapshot is not None:
            runtime.world_state, runtime.transition_fast = snapshot.state, snapshot.psi
            runtime.prefilled = True
            runtime.previous_committed_pose = repeat_batch(b['camera_conditions'][:, max(start-1,0), :16],
                                                           self.y.shape[0]).reshape(-1,4,4)
            runtime.committed_frame_ids = [set(range(max(1,start))) for _ in range(self.y.shape[0])]
            runtime.commit_count = runtime.revision = runtime.predict_count = 1 + (start+1)//3
            session.camera_cache_lengths = {k:list(v) for k,v in snapshot.lengths.items()}
        return session

    def extract(self, context, cache, lengths):
        if set(context.candidates) != set(range(5)):
            raise RuntimeError('All five anchors must write once per clean transaction')
        state = torch.stack([context.candidates[i][0] for i in range(5)], 1)
        # This backend deliberately accepts only the pinned eta_psi=0 StateTTT model.
        finite(state, 'clean S')
        return Snapshot(state, context.psi, cache, lengths)

    def prefill(self):
        def call(initial):
            session = self.session()
            context = session.begin_chunk(0,1,prefill=True).for_clean()
            scratch = [[None]*10 for _ in self.model.blocks]
            out,_ = session.forward(repeat_batch(initial,self.y.shape[0]),
                initial.new_zeros(self.y.shape[0]),self.y,context,scratch,0,1,
                self.mask,self.batch.get('data_info'),save=True)
            validate_clean_output(out,repeat_batch(initial,self.y.shape[0]))
            return self.extract(context,scratch,{i:[] for i in ANCHORS})
        return checkpoint_tree(call,(self.batch['initial_latent'],),enabled=self.checkpointed)

    def commit(self, session, context, clean, cache, start, end):
        ctx = context.for_clean()
        expanded = repeat_batch(clean,self.y.shape[0])
        out,current = session.forward(expanded,clean.new_zeros(self.y.shape[0]),self.y,ctx,
            cache,start,end,self.mask,self.batch.get('data_info'),save=True)
        validate_clean_output(out,expanded)
        carried = session.carry_clean_cache(current,cache,2)
        result = self.extract(ctx,carried,session.camera_cache_lengths)
        stats = [ctx.candidates[i][2] for i in range(5)] if not _RECOMPUTING.get() else []
        return result, stats

    def generate(self, noise, steps, shift, scale):
        if noise.shape[2] != self.frames or self.frames < 4 or self.frames % 3 != 1:
            raise ValueError('Complete 1+3n noise/camera episode required')
        sigmas,timesteps = shifted_schedule(steps,shift,noise.device)
        snapshot = self.prefill()
        outputs = []
        for index,(start,end) in enumerate(chunk_ranges(self.frames)):
            def chunk(latent, incoming, *, start=start,end=end,index=index):
                session = self.session(incoming,start)
                context = session.begin_chunk(start,end)
                local_records = []
                query_records = []
                for j,t in enumerate(timesteps):
                    context.collect_local_stats = self.collect and j == len(timesteps)-1 and not _RECOMPUTING.get()
                    context.measure_psi_output = context.collect_local_stats
                    ts = t.expand(self.y.shape[0],1,end-start).clone()
                    if start == 0:
                        latent = torch.cat((self.batch['initial_latent'],latent[:,:,1:]),2)
                        ts[:,:,0] = 0
                    def denoise(value, time, history):
                        query = self.session(history,start)
                        ctx = query.begin_chunk(start,end)
                        ctx.collect_local_stats = self.collect and j == len(timesteps)-1 and not _RECOMPUTING.get()
                        ctx.measure_psi_output = ctx.collect_local_stats
                        velocity,_ = query.forward(repeat_batch(value,self.y.shape[0]),time,self.y,
                            ctx,history.cache,start,end,self.mask,self.batch.get('data_info'),noise_sigma=time/1000)
                        if ctx.collect_local_stats and not _RECOMPUTING.get():
                            query_records.append(ctx.local_stats)
                        return velocity
                    velocity = checkpoint_tree(denoise,(latent,ts,incoming),enabled=self.checkpointed)
                    velocity = cfg_velocity(velocity,scale)
                    latent = finite(latent + (sigmas[j+1]-sigmas[j])*velocity.float(),'student Euler step')
                    if start == 0:
                        latent = torch.cat((self.batch['initial_latent'],latent[:,:,1:]),2)
                outgoing,stats = self.commit(session,context,latent,incoming.cache,start,end)
                if not _RECOMPUTING.get():
                    local_records = [dict(chunk=index,start=start,end=end,anchors=stats,
                                          local=query_records[-1] if query_records else {},
                                          state=outgoing.state.detach())]
                return latent,outgoing,local_records
            latent,snapshot,records = checkpoint_tree(chunk,(noise[:,:,start:end],snapshot),
                enabled=self.checkpointed,cpu_inputs=(1,) if noise.is_cuda else ())
            outputs.append(latent);self.records.extend(records)
        result = torch.cat(outputs,2)
        if not torch.equal(result[:,:,:1],self.batch['initial_latent']):
            raise RuntimeError('Observed initial latent changed')
        return result,snapshot

    def score(self, noisy, sigma, history, conditional=True):
        """Replay only preceding generated clean chunks; never expose future GT."""
        if history.shape != noisy.shape:
            raise ValueError('Score history and noisy episode shapes differ')
        self.y = self.batch['y'] if conditional else self.batch['uncondition']
        snapshot = self.prefill()
        predictions = []
        for start,end in chunk_ranges(self.frames):
            def chunk(x,clean,incoming,*,start=start,end=end):
                session = self.session(incoming,start)
                context = session.begin_chunk(start,end)
                t = sigma[:,None,start:end]*1000
                if start == 0:
                    x = torch.cat((self.batch['initial_latent'],x[:,:,1:]),2)
                    t = t.clone();t[:,:,0] = 0
                v,_ = session.forward(x,t,self.y,context,incoming.cache,start,end,
                    self.mask,self.batch.get('data_info'),noise_sigma=t/1000)
                outgoing,_ = self.commit(session,context,clean,incoming.cache,start,end)
                return v,outgoing
            value,snapshot = checkpoint_tree(chunk,(noisy[:,:,start:end],history[:,:,start:end],snapshot),
                enabled=self.checkpointed,cpu_inputs=(2,) if noisy.is_cuda else ())
            predictions.append(value)
        return torch.cat(predictions,2)
