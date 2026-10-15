"""Two-rank TTN backend for LongLive 2.0's alternating DMD training."""
from dataclasses import asdict
from pathlib import Path
from datetime import timedelta
import gc
import hashlib
import json
import os
import time
import random
import torch
import torch.distributed as dist
from worldttn.core import TTNConfig
from worldttn.sana import build_sana, load_sana_config, configure_activation_checkpointing, configure_cross_attention
from worldttn.checkpoint import read_checkpoint, apply_checkpoint_weights
from worldttn.training import activation_storage
from worldttn.performance import configure_execution
from .adapters import install, optimizer, shard, clip_grad, parameters, load_adapter_weights
from .checkpoint import EMA, save, restore, export_best, digest
from .data import Data
from .objective import StateDMD
from .sampler import Transactions, finite


def emit(record):
    print('[TTN DMD] '+json.dumps(record,default=str,ensure_ascii=False),flush=True)


def gradient_report(model):
    outcome = {'lora':{'tensors':0,'with_grad':0,'norm_sq':0.},
               'state_ttt':{'tensors':0,'with_grad':0,'norm_sq':0.}}
    for name,p in parameters(model).items():
        group = outcome['state_ttt' if '.state_ttt.' in name else 'lora']
        group['tensors'] += 1
        if p.grad is not None:
            value = p.grad.to_local() if hasattr(p.grad,'to_local') else p.grad
            group['with_grad'] += 1
            group['norm_sq'] += float(value.float().square().sum())
    for group in outcome.values():
        group['norm'] = group.pop('norm_sq')**.5
    return outcome


def state_metrics(records):
    result=[]
    for record in records[-5:]:
        state=record['state'].float()
        singular=torch.linalg.svdvals(state)
        sq=singular.square()
        energy=sq.sum(-1).clamp_min(1e-12)
        anchors={}
        for i,index in enumerate((3,7,11,15,19)):
            stats=record['anchors'][i]
            local=record['local'].get(index,{})
            anchors[str(index)]={'s_rms':float(state[:,i].square().mean().sqrt()),
                'stable_rank':float((energy[:,i]/sq[:,i,0].clamp_min(1e-12)).mean()),
                'top1_energy':float((sq[:,i,0]/energy[:,i]).mean()),
                'clean_delta_relative':stats.get('state_ttt',{}).get('state_delta_relative'),
                'local_output_effect':local.get('state_ttt',{}).get('output_effect')}
        result.append({'chunk':record['chunk'],'anchors':anchors})
    return result


class StateTTTTrainer:
    def __init__(self, config):
        self.config=config
        self.output=Path(config.logdir).expanduser().resolve()
        self.output.mkdir(parents=True,exist_ok=True)
        if int(os.environ.get('WORLD_SIZE','1'))!=2:
            raise ValueError('This recipe requires exactly two FSDP2 ranks on GPU 6,7')
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='6,7':
            raise ValueError('Expected physical CUDA_VISIBLE_DEVICES=6,7')
        self.rank=int(os.environ['RANK']); self.local_rank=int(os.environ['LOCAL_RANK'])
        self.device=torch.device('cuda',self.local_rank)
        torch.cuda.set_device(self.device)
        dist.init_process_group('nccl',timeout=timedelta(minutes=60))
        self.control=dist.new_group(backend='gloo',timeout=timedelta(minutes=60))
        # Limits apply per process; the enclosing scope enforces the total RAM cap.
        torch.cuda.set_per_process_memory_fraction(40*1024/46068,self.device)
        # Full-sequence DMD peaks during backward recomputation; keeping any
        # saved activation snapshot on these 46GB cards crowds FSDP all-gather.
        # Pageable CPU snapshots preserve values and gradients exactly.
        os.environ['TTN_ACTIVATION_GPU_CACHE_GIB']='0'
        os.environ['TTN_ACTIVATION_GPU_ALLOCATION_LIMIT_GIB']='38'
        random.seed(config.seed+self.rank);torch.manual_seed(config.seed+self.rank)
        self.teacher_path=Path(config.teacher).resolve()
        if (self.teacher_path.stat().st_size!=10659820965 or
                self.teacher_path.stat().st_mtime_ns!=1792050723201920478):
            raise ValueError('Pinned step200 checkpoint size/mtime changed')
        self.checkpoint_hash=digest(self.teacher_path)
        if self.checkpoint_hash!='aa8bfc12706ee0890bd94b7fc1227396fee0e62e169cc19f9b7280b87d967f56':
            raise ValueError('Pinned step200 SHA256 changed')
        self.payload=torch.load(self.teacher_path,map_location='cpu',weights_only=False,mmap=True)
        if (self.payload['step']!=200 or self.payload['distributed']['checkpoint_id']!=
                '11199e5c315b44e69eb381953e31f484'):
            raise ValueError('Teacher step/checkpoint ID differs')
        self.ttn_config=TTNConfig(**self.payload['config'])
        if (self.ttn_config.state_ttt_rank!=64 or self.ttn_config.eta_psi!=0 or
                self.ttn_config.state_ttt_policy!='curvature'):
            raise ValueError('Teacher is not the enhanced rank64 StateTTT method')
        self.base_path=Path(config.base_weights).resolve()
        if digest(self.base_path)!=self.payload['base_sha256']:
            raise ValueError('Frozen SANA base SHA256 mismatch')
        self.sana_config=load_sana_config(config.sana_config)
        identity=self.payload['distributed']['training_config']
        from dataclasses import replace
        for destination,source in (('model','sana_model'),('text_encoder','text_encoder'),
                                   ('scheduler','scheduler'),('data','data')):
            setattr(self.sana_config,destination,replace(getattr(self.sana_config,destination),**identity[source]))
        data_root=Path(config.dataset_root)
        self.sana_config.data.data_dir={'sekai_game':str(data_root/'data/sekai_game_train_961frames_16fps_ovl640')}
        self.sana_config.data.vae_cache_dir=str(data_root/'data/vae_cache/LTX2VAE_diffusers_704x1280/sekai_game_train_961frames_16fps_ovl640')
        self.sana_config.data.num_frames=961
        self.shift=self.sana_config.scheduler.inference_flow_shift
        if self.shift!=9.8:raise ValueError('Teacher inference shift changed')
        self.data=Data(self.sana_config,self.output,self.device,config.seed)
        self.stage4=None
        if config.get('stage4_source'):
            if config.student_steps != 4:
                raise ValueError('Stage-4 initialization requires four solver steps')
            source=Path(config.stage4_source).resolve()
            source_hash=digest(source)
            review=Path(config.stage4_review).resolve()
            approved=json.loads(review.read_text())
            if not approved.get('video_review') or approved.get('source_sha256')!=source_hash:
                raise ValueError('Stage-4 requires video review of this exact best-8 adapter')
            self.stage4=torch.load(source,map_location='cpu',weights_only=True)
            previous=self.stage4['identity']
            if (self.stage4['format']!='TTN-DMD-adapters-v1' or
                    previous['config']['student_steps']!=8 or
                    previous['checkpoint_sha256']!=self.checkpoint_hash or
                    previous['base_sha256']!=self.payload['base_sha256'] or
                    previous['dataset_sha256']!=self.data.identity or
                    previous['config']['lora_rank']!=config.lora_rank or
                    previous['config']['cfg']!=config.cfg):
                raise ValueError('Stage-4 source is not the validated eight-step student')
            from .evaluate import phase4_gate, video_review_ready
            evaluated=source.parent/f'eval-{self.stage4["counts"]["student"]:06d}'/'metrics.json'
            if (not video_review_ready(source.parent,source_hash) or
                    not phase4_gate(json.loads(evaluated.read_text())['results'],True)):
                raise ValueError('Stage-4 numerical and speed gate failed')
        self.identity={'checkpoint_sha256':self.checkpoint_hash,'base_sha256':self.payload['base_sha256'],
            'dataset_sha256':self.data.identity,'config':{k:config[k] for k in
                ('student_steps','target_updates','critic_ratio','cfg','lora_rank','lr_student','lr_fake','seed')}}
        if self.stage4 is not None:self.identity['stage4_source_sha256']=source_hash
        self.models={}
        for role in ('teacher','student','fake'):
            model=build_sana(self.sana_config,self.ttn_config,str(self.base_path),device=self.device,
                             dtype=torch.float32)
            model.ttn_training_protocol='dual-psi-clean-fullgrad-v1'
            model.ttn_training_latent_frames=121
            model.ttn_fullgrad_checkpoint=True
            for block in model.blocks:block.ttn_fullgrad_checkpoint=True
            configure_activation_checkpointing(model,'ffn-cross-attn')
            # L20 efficient SDPA preserves the padding mask and Q/K/V gradients
            # while avoiding the large materialized math attention matrix.
            configure_cross_attention(model,'efficient')
            configure_execution(model)
            # The real checkpoint validator is used before any LoRA schema change.
            validated=read_checkpoint(self.teacher_path,model)
            apply_checkpoint_weights(model,validated)
            if role=='teacher': model.requires_grad_(False)
            else:
                install(model,config.lora_rank)
                if self.stage4 is not None:
                    load_adapter_weights(model,self.stage4['weights'][role])
            shard(model)
            self.models[role]=model
            emit({'phase':'role_loaded','role':role,'rank':self.rank,
                  'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'allocated_gib':torch.cuda.memory_allocated(self.device)/2**30})
            gc.collect();torch.cuda.empty_cache()
        self.teacher=self.models['teacher']
        self.optimizers={name:optimizer(self.models[name],config.lr_student if name=='student' else config.lr_fake)
                         for name in ('student','fake')}
        self.ema=EMA(self.models['student'],.99)
        self.counts={'student':0,'critic':0}
        self.best=None
        if (self.output/'last.json').exists():
            self.counts=restore(self.output,self.models,self.optimizers,self.ema,self.identity)
        previous_best=self.output/'best-adapters.pt'
        if previous_best.is_file():
            best_payload=torch.load(previous_best,map_location='cpu',weights_only=True)
            if best_payload['identity']!=self.identity: raise ValueError('Best model belongs to another run')
            self.best=(best_payload['metrics']['lpips'],best_payload['metrics']['tail_mse'])
        self.metrics_path=self.output/f'metrics-rank{self.rank}.jsonl'
        self.wandb=None
        if self.rank==0 and not config.disable_wandb:
            import wandb
            self.wandb=wandb.init(project=config.wandb_project,entity=config.wandb_entity,
                dir=str(self.output),config=dict(self.identity['config']),resume='allow',
                id=hashlib.sha256(str(self.output).encode()).hexdigest()[:8],
                mode='online' if os.environ.get('WANDB_API_KEY') else 'offline')
            self.wandb.define_metric('student/*',step_metric='student_step')
            self.wandb.define_metric('validation/*',step_metric='student_step')
            self.wandb.define_metric('critic/*',step_metric='critic_step')
        if self.rank==0: emit({'phase':'ready','identity':self.identity,'videos':len(self.data.partition['train']),
                              'heldout':len(self.data.partition['validation'])})

    def record(self, row, axis):
        if getattr(self, '_audit_mode', False):
            return
        row.update(student_step=self.counts['student'],critic_step=self.counts['critic'],rank=self.rank,
                   axis=axis,time=time.time(),gpu_peak_gib=torch.cuda.max_memory_allocated(self.device)/2**30)
        with self.metrics_path.open('a',encoding='utf-8') as stream: stream.write(json.dumps(row,default=str)+'\n')
        if self.rank==0:
            emit(row)
            if self.wandb:
                values={'student_step':self.counts['student'],'critic_step':self.counts['critic']}
                values.update({f'{axis}/{key}':value for key,value in row.items()
                               if isinstance(value,(int,float)) and key not in ('student_step','critic_step')})
                self.wandb.log(values)

    def batch(self,frames,counter):
        index=self.data.index(counter,self.rank,dist.get_world_size())
        return self.data.batch(index,frames)[0]

    def generate(self,batch,model,collect=False):
        latent=torch.randn((1,128,batch['camera_conditions'].shape[1],22,40),device=self.device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            runner=Transactions(model,batch,checkpointed=True,collect=collect)
            output,_=runner.generate(latent,self.config.student_steps,self.shift,self.config.cfg)
        return finite(output,'generated latent'),runner.records

    def critic_update(self,batch):
        model=self.models['fake']; opt=self.optimizers['fake']
        opt.zero_grad(set_to_none=True)
        with torch.no_grad(): generated,_=self.generate(batch,self.models['student'])
        dmd=StateDMD(self.teacher,model,batch,shift=self.shift,cfg=self.config.cfg)
        with activation_storage('cpu-pageable',device=self.device):
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,log=dmd.critic_loss(generated.detach())
            if self.config.get('capacity_only',False):
                emit({'phase':'critic_backward_begin','rank':self.rank,
                      'allocated_gib':torch.cuda.memory_allocated(self.device)/2**30,
                      'peak_gib':torch.cuda.max_memory_allocated(self.device)/2**30})
            loss.backward()
        gradients=gradient_report(model)
        if gradients['state_ttt']['with_grad']!=10 or gradients['lora']['with_grad']==0:
            raise RuntimeError('Missing fake-score StateTTT or LoRA gradient')
        norm=clip_grad(model)
        opt.step()
        self.counts['critic']+=1
        self.record({'phase':'critic','loss':float(loss),'grad_norm':norm,'groups':gradients},'critic')
        del generated,dmd,loss

    def student_update(self,batch):
        model=self.models['student'];opt=self.optimizers['student']
        opt.zero_grad(set_to_none=True)
        collect=(self.counts['student']+1)%25==0
        with activation_storage('cpu-pageable',device=self.device):
            generated,records=self.generate(batch,model,collect)
            dmd=StateDMD(self.teacher,self.models['fake'],batch,shift=self.shift,cfg=self.config.cfg)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,log=dmd.student_loss(generated)
            loss.backward()
        gradients=gradient_report(model)
        if gradients['state_ttt']['with_grad']!=10 or gradients['lora']['with_grad']==0:
            raise RuntimeError('Missing student StateTTT or LoRA gradient')
        norm=clip_grad(model)
        before={n:p.detach().clone() for n,p in parameters(model).items()} if collect else None
        opt.step();self.ema.update(model)
        self.counts['student']+=1
        row={'phase':'student','loss':float(loss),'grad_norm':norm,'groups':gradients,
             'dmd_direction':float(log['dmdtrain_gradient_norm']),
             'normalizer_min':float(log['normalizer_min']),
             'normalizer_mean':float(log['normalizer_mean'])}
        if collect:
            row['state']=state_metrics(records)
            row['update_norm']={kind:sum(float((p-before[n]).float().square().sum()) for n,p in parameters(model).items()
                if ('.state_ttt.' in n)==(kind=='state_ttt'))**.5 for kind in ('lora','state_ttt')}
        self.record(row,'student')
        del generated,records,dmd,loss

    def capacity(self):
        # Complete real forward, backward, optimizer and state/camera retention.
        for frames in (25,61,121):
            start=time.monotonic()
            batch=self.batch(frames,0)
            self.critic_update(batch)
            self.student_update(batch)
            torch.cuda.synchronize(self.device)
            self.record({'phase':'capacity_pass','frames':frames,'seconds':time.monotonic()-start,
                'host_rss_gib':__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss/2**20},'capacity')
            dist.barrier(group=self.control)
        if self.rank==0:(self.output/'capacity-passed.json').write_text(json.dumps(self.identity))

    def resume_audit(self):
        """Exercise a real two-rank next update on both sides of recovery."""
        if self.counts != {'student': 0, 'critic': 0}:
            return  # A resumed formal run already has a verified recovery bundle.
        audit = self.output/'resume-audit'
        batch = self.batch(25, 0)
        save(audit,self.models,self.optimizers,self.ema,self.counts,self.identity,self.control)
        self._audit_mode = True
        try:
            expected = None
            for attempt in range(2):
                if attempt:
                    self.counts = restore(audit,self.models,self.optimizers,self.ema,self.identity)
                self.critic_update(batch)
                self.student_update(batch)
                snapshot = {role:{name:p.detach().to_local().cpu().clone() if hasattr(p,'to_local')
                                  else p.detach().cpu().clone()
                                  for name,p in parameters(self.models[role]).items()}
                            for role in ('student','fake')}
                if expected is None:
                    expected = snapshot
                else:
                    mismatch = [(role,name) for role in expected for name in expected[role]
                                if not torch.equal(expected[role][name],snapshot[role][name])]
                    reports = [None]*dist.get_world_size()
                    dist.all_gather_object(reports,mismatch,group=self.control)
                    if any(reports):
                        raise RuntimeError('Two-rank next-update recovery differs: '+repr(reports)[:1000])
            self.counts = restore(audit,self.models,self.optimizers,self.ema,self.identity)
            dist.barrier(group=self.control)
            if self.rank==0:
                (audit/'passed.json').write_text(json.dumps({'identity':self.identity,
                    'world_size':2,'frames':25,'next_critic_updates':1,'next_student_updates':1,
                    'comparison':'exact local parameters on both ranks'}))
        finally:
            self._audit_mode = False

    def train(self):
        try:
            if self.config.get('capacity_only',False):
                self.capacity(); return
            proof=self.output/'capacity-passed.json'
            if not proof.is_file() or json.loads(proof.read_text())!=self.identity:
                raise RuntimeError('121-latent two-rank capacity acceptance must pass first')
            self.resume_audit()
            target=self.config.target_updates
            while self.counts['student']<target:
                for _ in range(self.config.critic_ratio):
                    self.critic_update(self.batch(121,self.counts['critic']))
                self.student_update(self.batch(121,self.counts['student']))
                if self.counts['student']%25==0:
                    # Ensure both ranks have completed the same optimizer step.
                    dist.barrier(group=self.control)
                    save(self.output,self.models,self.optimizers,self.ema,self.counts,self.identity,self.control)
                    from .evaluate import evaluate
                    with self.ema.applied(self.models['student']):
                        score,results=evaluate(self)
                        improved=self.best is None or score<self.best
                        if improved:
                            export_best(self.output/'best-adapters.pt',self.models,self.identity,self.counts,
                                        {'lpips':score[0],'tail_mse':score[1],'selection':'EMA'})
                    if self.rank==0:
                        self.record({'phase':'validation','lpips':score[0],'tail_mse':score[1],
                                     'cases':results},'validation')
                    if improved:self.best=score
                    dist.barrier(group=self.control)
            if self.rank==0:emit({'phase':'complete','counts':self.counts,'best':self.best})
        finally:
            if self.wandb:self.wandb.finish()
            dist.destroy_process_group()
