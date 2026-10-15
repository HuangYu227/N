"""Two-rank, fixed-input 30/60-second review of a completed eight-step run."""
import argparse
from dataclasses import replace
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from worldttn.checkpoint import apply_checkpoint_weights, read_checkpoint
from worldttn.checkpoint_integrity import atomic_json
from worldttn.core import TTNConfig
from worldttn.distillation.adapters import install, load_adapter_weights, shard
from worldttn.distillation.checkpoint import digest
from worldttn.distillation.data import Data
from worldttn.distillation.sampler import Transactions
from worldttn.performance import configure_execution
from worldttn.sana import build_sana, configure_cross_attention, load_sana_config


def training_config(config, checkpoint, dataset_root):
    sana = load_sana_config(config.sana_config)
    original = checkpoint['distributed']['training_config']
    for destination, source in (('model', 'sana_model'), ('text_encoder', 'text_encoder'),
                                ('scheduler', 'scheduler'), ('data', 'data')):
        setattr(sana, destination, replace(getattr(sana, destination), **original[source]))
    root = Path(dataset_root)
    sana.data.data_dir = {'sekai_game': str(root/'data/sekai_game_train_961frames_16fps_ovl640')}
    sana.data.vae_cache_dir = str(root/'data/vae_cache/LTX2VAE_diffusers_704x1280/sekai_game_train_961frames_16fps_ovl640')
    sana.data.num_frames = 961
    return sana


def build_role(sana, ttn, base, teacher_path, weights, rank):
    model = build_sana(sana, ttn, str(base), device=torch.device('cuda', rank), dtype=torch.float32)
    model.ttn_training_protocol = 'dual-psi-clean-fullgrad-v1'
    model.ttn_training_latent_frames = 121
    model.ttn_fullgrad_checkpoint = True
    for block in model.blocks:
        block.ttn_fullgrad_checkpoint = True
    configure_cross_attention(model, 'efficient')
    configure_execution(model)
    apply_checkpoint_weights(model, read_checkpoint(teacher_path, model))
    if weights is not None:
        install(model, 128)
        load_adapter_weights(model, weights)
    model.eval().requires_grad_(False)
    shard(model)
    return model


def render(videos, sana, root, rank):
    if rank != 0:
        return
    from diffusion.model.builder import get_vae, vae_decode
    from sana.tools import resolve_hf_path
    from tools.ttn_decode_comparison import pixels
    import imageio.v2 as iio
    import imageio_ffmpeg

    imageio_ffmpeg.get_ffmpeg_exe()
    dtype = getattr(torch, sana.vae.weight_dtype)
    vae = get_vae(sana.vae.vae_type, resolve_hf_path(sana.vae.vae_pretrained),
                  device='cuda', dtype=dtype, config=sana.vae)
    if hasattr(vae, 'enable_tiling'):
        vae.enable_tiling()
    if hasattr(vae, 'use_framewise_decoding'):
        vae.use_framewise_decoding = True
        vae.tile_sample_stride_num_frames = sana.vae.tile_sample_stride_num_frames
        vae.tile_sample_min_num_frames = sana.vae.tile_sample_min_num_frames
    for seconds, method in videos:
        source = root/f'{seconds}s-{method}.pt'
        latent = torch.load(source, map_location='cpu', weights_only=True)
        with torch.no_grad():
            frames = pixels(vae_decode(sana.vae.vae_type, vae, latent.to('cuda', dtype)))
        expected = (seconds*16+1, 704, 1280, 3)
        if frames.shape != expected:
            raise ValueError(f'{source.name}: decoded {frames.shape}, expected {expected}')
        iio.mimwrite(root/f'{seconds}s-{method}.mp4', frames, fps=16,
                     codec='libx264', macro_block_size=1)
        del frames, latent
        torch.cuda.empty_cache()


def review(run):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '6,7' or int(os.environ.get('WORLD_SIZE', '0')) != 2:
        raise ValueError('Review requires two FSDP2 ranks on physical GPU 6 and 7')
    run = Path(run).resolve()
    if json.loads((run/'status.json').read_text())['status'] != 'completed':
        raise ValueError('Complete the eight-step training before video review')
    launch = json.loads((run/'launch.json').read_text())
    config_path = Path(launch['config'])
    if digest(config_path) != launch['config_sha256']:
        raise ValueError('Training config changed after launch')
    config = OmegaConf.load(config_path)
    if config.student_steps != 8:
        raise ValueError('Review expects the completed eight-step stage')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    if torch.cuda.mem_get_info(rank)[0] < 43*(1 << 30):
        raise RuntimeError(f'GPU {6+rank} needs at least 43GiB free before video review')
    dist.init_process_group('nccl', timeout=timedelta(hours=6))
    control = dist.new_group(backend='gloo', timeout=timedelta(hours=6))
    torch.cuda.set_per_process_memory_fraction(40*1024/46068, rank)
    root = run/'review-videos'
    root.mkdir(exist_ok=True)
    best_path = run/'best-adapters.pt'
    best_sha = digest(best_path)
    best = torch.load(best_path, map_location='cpu', weights_only=True)
    best_step = int(best['counts']['student'])
    if rank == 0:
        atomic_json({'status':'running','best_sha256':best_sha,'best_step':best_step},root/'manifest.json')
    teacher_path = Path(config.teacher)
    base = Path(config.base_weights)
    if best['format'] != 'TTN-DMD-adapters-v1' or best['identity']['checkpoint_sha256'] != digest(teacher_path):
        raise ValueError('Best adapter and teacher identity disagree')
    checkpoint = torch.load(teacher_path, map_location='cpu', weights_only=False, mmap=True)
    if checkpoint['step'] != 200 or checkpoint['distributed']['checkpoint_id'] != '11199e5c315b44e69eb381953e31f484':
        raise ValueError('Expected fixed H100 step200 teacher')
    if checkpoint['base_sha256'] != digest(base) or best['identity']['base_sha256'] != checkpoint['base_sha256']:
        raise ValueError('SANA base identity changed')
    sana = training_config(config, checkpoint, config.dataset_root)
    data = Data(sana, root, torch.device('cuda', rank), config.seed)
    if data.identity != best['identity']['dataset_sha256']:
        raise ValueError('Distillation dataset changed')
    case_index = data.validation(1)[0]
    case_id = data.dataset.dataset[case_index]['key']
    ttn = TTNConfig(**checkpoint['config'])
    teacher = build_role(sana, ttn, base, teacher_path, None, rank)
    student = build_role(sana, ttn, base, teacher_path, best['weights']['student'], rank)
    del checkpoint, best
    gc.collect()
    rows, videos = [], []
    for seconds, frames in ((30, 61), (60, 121)):
        batch, _ = data.batch(case_index, frames)
        generator = torch.Generator(device=torch.device('cuda', rank)).manual_seed(config.seed+case_index)
        noise = torch.randn((1, 128, frames, 22, 40), generator=generator,
                            device=torch.device('cuda', rank))
        for method, model, steps, ablation in (
                ('teacher20', teacher, 20, 'full'),
                ('undistilled8', teacher, 8, 'full'),
                ('student8', student, 8, 'full'),
                ('student8-ttt-off', student, 8, 'no-ttt')):
            torch.cuda.synchronize(rank)
            started = time.monotonic()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                latent, _ = Transactions(model, batch, checkpointed=False,
                                         ablation=ablation).generate(noise, steps,
                                                                    sana.scheduler.inference_flow_shift,
                                                                    config.cfg)
            torch.cuda.synchronize(rank)
            elapsed = time.monotonic()-started
            if rank == 0:
                torch.save(latent.cpu(), root/f'{seconds}s-{method}.pt')
                rows.append({'seconds': seconds, 'method': method, 'solver_steps': steps,
                             'sample_seconds': elapsed, 'ablation': ablation})
                videos.append((seconds, method))
            del latent
            dist.barrier(group=control)
    del teacher, student, data, noise, batch
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier(group=control)
    render(videos, sana, root, rank)
    if rank == 0:
        atomic_json({'status':'completed','best_sha256':best_sha,'best_step':best_step,
            'case_index':case_index,'case_id':case_id, 'seed':config.seed+case_index,
            'same_input_and_noise_within_each_duration':True,
            'videos':rows,'scope':'qualitative review; no generated future ground truth'},
            root/'manifest.json')
    dist.barrier(group=control)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    review(parser.parse_args().run)
