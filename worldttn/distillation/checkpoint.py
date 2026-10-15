"""Adapter-only, atomic two-role recovery; the frozen step200 base is referenced."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import shutil
import uuid
import torch
import torch.distributed as dist
from worldttn.checkpoint import atomic_save, rng_state, restore_rng
from worldttn.checkpoint_integrity import atomic_json
from worldttn.parallel_checkpoint import _pack, _unpack
from .adapters import parameters


def local(value):
    return value.to_local() if hasattr(value, 'to_local') else value


class EMA:
    """LongLive's FP32 EMA rule, restricted to FSDP2 trainable local shards."""
    def __init__(self, model, decay=.99):
        self.decay = decay
        self.shadow = {n:local(p).detach().clone().float() for n,p in parameters(model).items()}

    @torch.no_grad()
    def update(self, model):
        for n,p in parameters(model).items():
            self.shadow[n].lerp_(local(p).float(), 1-self.decay)

    @contextmanager
    def applied(self, model):
        saved = {n:local(p).detach().clone() for n,p in parameters(model).items()}
        try:
            with torch.no_grad():
                for n,p in parameters(model).items(): local(p).copy_(self.shadow[n])
            yield
        finally:
            with torch.no_grad():
                for n,p in parameters(model).items(): local(p).copy_(saved[n])


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20), b''): result.update(block)
    return result.hexdigest()


def save(folder, models, optimizers, ema, counts, identity, control=None):
    root = Path(folder).resolve()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    token = [uuid.uuid4().hex if rank==0 else None]
    if world>1: dist.broadcast_object_list(token, src=0, group=control)
    bundle = root/f'resume-{counts["student"]:06d}-{token[0][:8]}'
    bundle.mkdir(parents=True, exist_ok=True)
    state = {'format':'TTN-DMD-resume-v1', 'identity':identity, 'world':world, 'rank':rank,
             'counts':counts, 'rng':rng_state(),
             'parameters': {role:{n:local(p).detach().cpu() for n,p in parameters(m).items()}
                            for role,m in models.items()},
             'optimizer':{role:_pack(o.state_dict()) for role,o in optimizers.items()},
             'ema':{n:p.cpu() for n,p in ema.shadow.items()}}
    path = bundle/f'rank-{rank:02d}.pt'
    error = None
    try:
        atomic_save(state,path)
        receipt = {'name':path.name, 'sha256':digest(path), 'bytes':path.stat().st_size}
    except Exception as exc:
        error = repr(exc); receipt = None
    reports = [(error,receipt)]
    if world>1:
        reports = [None]*world
        dist.all_gather_object(reports,(error,receipt),group=control)
    if any(r[0] for r in reports): raise RuntimeError('Recovery write failed: '+repr(reports))
    if rank==0:
        manifest = {'format':'TTN-DMD-resume-v1', 'identity':identity, 'counts':counts,
                    'world':world, 'files':[r[1] for r in reports]}
        atomic_json(manifest,bundle/'manifest.json')
        atomic_json({'bundle':bundle.name,'counts':counts},root/'last.json')
        candidates = sorted((p for p in root.glob('resume-*') if (p/'manifest.json').is_file()),
                            key=lambda p:p.stat().st_mtime_ns)
        for old in candidates[:-2]:
            # Only complete bundles created by this exact run are eligible.
            metadata = json.loads((old/'manifest.json').read_text())
            if old.resolve().parent==root and metadata['identity']==identity:
                shutil.rmtree(old)
    if world>1: dist.barrier(group=control)
    return bundle


def restore(folder, models, optimizers, ema, identity):
    root = Path(folder).resolve()
    pointer = json.loads((root/'last.json').read_text())
    bundle = (root/pointer['bundle']).resolve()
    if bundle.parent!=root: raise ValueError('Recovery path escapes run directory')
    metadata = json.loads((bundle/'manifest.json').read_text())
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if metadata['world']!=world or metadata['identity']!=identity:
        raise ValueError('Recovery world size/base/config identity changed')
    receipt = metadata['files'][rank]
    path = bundle/receipt['name']
    if digest(path)!=receipt['sha256']: raise ValueError('Recovery shard SHA256 mismatch')
    state = torch.load(path,map_location='cpu',weights_only=False)
    with torch.no_grad():
        for role,model in models.items():
            expected = parameters(model)
            if expected.keys()!=state['parameters'][role].keys(): raise ValueError('Adapter schema mismatch')
            for name,p in expected.items(): local(p).copy_(state['parameters'][role][name])
            if role not in optimizers:
                continue  # The frozen teacher has neither trainable tensors nor optimizer.
            opt = optimizers[role]
            saved = state['optimizer'][role]
            for group,stored in zip(opt.param_groups,saved['param_groups']):
                for p,index in zip(group['params'],stored['params']):
                    if index in saved['state']:
                        saved['state'][index] = _unpack(saved['state'][index],p)
            opt.load_state_dict(saved)
        for name,value in state['ema'].items(): ema.shadow[name].copy_(value)
    restore_rng(state['rng'])
    return state['counts']


@torch.no_grad()
def export_best(path, models, identity, counts, metrics):
    rank = dist.get_rank() if dist.is_initialized() else 0
    weights = {}
    for role, model in models.items():
        weights[role] = {}
        for name,param in parameters(model).items():
            full = param.full_tensor() if hasattr(param,'full_tensor') else param
            if rank==0: weights[role][name] = full.detach().cpu()
    if rank==0:
        atomic_save({'format':'TTN-DMD-adapters-v1','identity':identity,'counts':counts,
                     'metrics':metrics,'weights':weights},path)
