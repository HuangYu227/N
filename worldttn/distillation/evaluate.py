"""Fixed free-rollout validation against decoded dataset latents, before MP4."""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from worldttn.checkpoint import atomic_save
from worldttn.checkpoint_integrity import atomic_json
from .sampler import Transactions


def segments(frames):
    tail = np.arange(1,frames)
    return dict(zip(('early','middle','tail'),np.array_split(tail,3)))


def image_metrics(video, reference, device):
    from tools.metrics.sana_wm.eval_unified import compute_ssim, compute_lpips
    result = {}
    for segment,ids in segments(len(video)).items():
        values = []
        for i in ids:
            a,b = video[i],reference[i]
            mse = float(np.square(a.astype(np.float32)/255-b.astype(np.float32)/255).mean())
            psnr = -10*math.log10(max(mse,1e-12))
            values.append((mse,psnr,compute_ssim(a,b),compute_lpips(a,b,str(device))))
        result[segment] = dict(zip(('rgb_mse','psnr','ssim','lpips'),np.mean(values,axis=0).tolist()))
    result['lpips'] = float(np.mean([v['lpips'] for v in result.values()]))
    return result


@torch.no_grad()
def evaluate(trainer):
    from diffusion.model.builder import get_vae,vae_decode
    from sana.tools import resolve_hf_path
    from tools.ttn_decode_comparison import pixels
    t = trainer
    root = t.output/f'eval-{t.counts["student"]:06d}'
    root.mkdir(exist_ok=True)
    results = []
    for index in t.data.validation(t.config.validation_cases):
        batch,reference = t.data.batch(index,121)
        generator = torch.Generator(device=t.device).manual_seed(t.config.seed+index)
        noise = torch.randn(reference.shape,generator=generator,device=t.device)
        outputs, times = {},{}
        for name,model,steps in [('teacher20',t.teacher,20),('undistilled8',t.teacher,8),
                                  ('student8',t.models['student'],t.config.student_steps)]:
            cache = t.output/f'baseline-{index}-{name}.pt'
            cached = [name!='student8' and cache.is_file() if t.rank==0 else None]
            dist.broadcast_object_list(cached,src=0,group=t.control)
            if cached[0]:
                saved = torch.load(cache,map_location='cpu',weights_only=True)
                outputs[name],times[name] = saved['latent'],saved['seconds']
            else:
                torch.cuda.synchronize(); started = time.monotonic()
                value,_ = Transactions(model,batch,checkpointed=False).generate(
                    noise,steps,t.shift,t.config.cfg)
                torch.cuda.synchronize()
                times[name] = time.monotonic()-started
                outputs[name] = value.cpu()
                if t.rank==0 and name!='student8':
                    atomic_save({'latent':outputs[name],'seconds':times[name]},cache)
                del value
            dist.barrier(group=t.control)
        # All FSDP sampling collectives are complete before rank0-only decode.
        if t.rank==0:
            dtype = getattr(torch,t.sana_config.vae.weight_dtype)
            vae = get_vae(t.sana_config.vae.vae_type,resolve_hf_path(t.sana_config.vae.vae_pretrained),
                          device=t.device,dtype=dtype,config=t.sana_config.vae)
            if hasattr(vae,'enable_tiling'): vae.enable_tiling()
            if hasattr(vae,'use_framewise_decoding'):
                vae.use_framewise_decoding = True
                vae.tile_sample_stride_num_frames = t.sana_config.vae.tile_sample_stride_num_frames
                vae.tile_sample_min_num_frames = t.sana_config.vae.tile_sample_min_num_frames
            target = pixels(vae_decode(t.sana_config.vae.vae_type,vae,reference.to(t.device,dtype)))
            for name,value in outputs.items():
                decoded = pixels(vae_decode(t.sana_config.vae.vae_type,vae,value.to(t.device,dtype)))
                metrics = image_metrics(decoded,target,t.device)
                errors = (value-reference).square().mean((0,1,3,4)).numpy()
                for segment,ids in segments(121).items(): metrics[segment]['latent_mse'] = float(errors[ids].mean())
                metrics.update(seconds=times[name],fps=len(decoded)/times[name])
                results.append({'case':t.data.dataset.dataset[index]['key'],'method':name,'metrics':metrics})
                torch.save(value,root/f'{index}-{name}.pt')
            del vae,decoded,target
            torch.cuda.empty_cache()
        dist.barrier(group=t.control)
    result = [results if t.rank==0 else None]
    dist.broadcast_object_list(result,src=0,group=t.control)
    current = [r['metrics'] for r in result[0] if r['method']=='student8']
    score = (sum(r['lpips'] for r in current)/len(current),
             sum(r['tail']['latent_mse'] for r in current)/len(current))
    if t.rank==0:
        atomic_json({'student_step':t.counts['student'],'score':score,
                     'results':result[0]},root/'metrics.json')
    return score,result[0]


def phase4_gate(results, video_review):
    """Numerical gate never substitutes for the explicit 30/60s video review."""
    if video_review is not True: return False
    students = {r['case']:r['metrics'] for r in results if r['method']=='student8'}
    teachers = {r['case']:r['metrics'] for r in results if r['method']=='teacher20'}
    if not students or students.keys()!=teachers.keys(): return False
    return all(s['lpips']<=1.05*max(teachers[k]['lpips'],1e-12)
        and s['tail']['latent_mse']<=1.05*max(teachers[k]['tail']['latent_mse'],1e-12)
        and s['fps']>teachers[k]['fps'] for k,s in students.items())


def video_review_ready(eight_run, best_sha256):
    root = Path(eight_run)/'review-videos'
    path = root/'manifest.json'
    if not path.is_file():
        return False
    manifest = json.loads(path.read_text())
    expected = {(seconds, method) for seconds in (30, 60) for method in
                ('teacher20', 'undistilled8', 'student8', 'student8-ttt-off')}
    actual = {(row['seconds'], row['method']) for row in manifest.get('videos', [])}
    return (manifest.get('status') == 'completed' and
            manifest.get('best_sha256') == best_sha256 and actual == expected and
            all((root/f'{seconds}s-{method}.mp4').is_file() and
                (root/f'{seconds}s-{method}.mp4').stat().st_size > 0
                for seconds, method in expected))
