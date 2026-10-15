"""Sequential fixed-checkpoint diagnostics; every CUDA worker has a conservative budget."""
import argparse
from collections import Counter
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F

REPO = Path(os.environ.get('TTN_EVAL_REPO', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(REPO))
from tools.ttn_state_ttt_video import VideoMemoryBudget, load_milestone
from worldttn.evaluation import tensor_sha256

CASES = ('00100100001_0004650_0006450_s000000', '00300200002_0048306_0050106_s000000')
INPUTS = ('initial_latent', 'camera_conditions', 'chunk_plucker', 'y', 'mask', 'uncondition')


class Budget(VideoMemoryBudget):
    ceiling_mib, allocator_mib, stop_mib = 25000, 20000, 22500

    def sample(self):
        used = super().sample()
        if torch.cuda.mem_get_info(0)[0] < 4096 * 2**20:
            raise RuntimeError('Less than 4096MiB free globally; stopping this evaluation worker')
        return used


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def hashes(batch):
    return {key: tensor_sha256(batch[key]) for key in INPUTS}


def validate_pair(batch, noise, case):
    if hashes(batch) != case['input_sha256'] or tensor_sha256(noise) != case['noise_sha256']:
        raise ValueError('Fixed conditioning/noise changed: ' + case['case_id'])


def config_for(run):
    from worldttn.evaluation import evaluation_config
    source = run['arguments'].get('sana_config') or 'configs/worldttn/sana_teacher_single_gpu.yaml'
    return evaluation_config(run, REPO / source)


def prepare(args):
    import imageio.v2
    import imageio_ffmpeg
    imageio_ffmpeg.get_ffmpeg_exe()  # Refuse a missing video encoder before expensive sampling.
    from worldttn.evaluation import select_cases, diagnostic_noise
    from worldttn.cli import seed_everything
    from diffusion.data.datasets.video.sana_wm_zip_latent_data import SanaWMZipLatentDataset
    from diffusion.model.builder import get_tokenizer_and_text_encoder
    from train_video_scripts.train_sana_wm_stage1 import _encode_prompts
    from torch.utils.data import default_collate
    from tools.ttn_custom_inference import load_case, make_geometry, encode_first_frame

    run, payload, _, _ = load_milestone(args.training_run, args.milestones[0])
    config = config_for(run)
    configs = [payload['config']]
    del payload
    for step in args.milestones[1:]:
        _, payload, _, _ = load_milestone(args.training_run, step)
        configs.append(payload['config'])
        del payload
    if any(c != configs[0] for c in configs[1:]):
        raise ValueError('Checkpoint architectures/configs differ')
    if (not configs[0]['local_update'] or not configs[0]['persistent_meta']
            or configs[0]['eta_psi'] != 0):
        raise ValueError('Requires full enhanced StateTTT with the old psi update disabled')
    data = asdict(config.data)
    data.update(num_frames=481, data_repeat=1, sort_dataset=True, shuffle_dataset=False)
    dataset = SanaWMZipLatentDataset(**data, resolution=config.data.image_size)
    selected = []
    for key in CASES:
        matches = [item for item in dataset.dataset if item['key'] == key]
        if len(matches) != 1:
            raise ValueError('Expected exactly one dataset case: ' + key)
        selected.extend(matches)
    dataset.dataset = selected
    cases, _ = select_cases(dataset, 61, 2, 3407,
                           dict(min_gap=30, distance_fraction=.02, angle_deg=5., max_pairs=5))
    prepared = []
    for case in cases:
        reference = case['reference'].float().cpu()
        batch = dict(initial_latent=reference[:, :, :1].clone(), camera_conditions=case['camera'],
                     chunk_plucker=case['plucker'][None], width=40, height=22,
                     data_info=default_collate([case['info']]))
        prepared.append(dict(case_id=case['case_id'], prompt=case['prompt'], seed=case['seed'],
                             batch=batch, reference=reference, reference_kind='dataset-future-GT-latents',
                             revisits=case['revisit_pairs']))
    custom, image, prompt = load_case(REPO / 'assets/worldttn/study_static/case.json')
    if custom['latent_frames'] != 61 or custom['camera']['trajectory'] != 'static':
        raise ValueError('Expected the original static study case')
    seed_everything(custom['seed'])
    batch = make_geometry(custom, config.vae.vae_stride)
    batch['initial_latent'] = encode_first_frame(custom, image, config, args.output)
    batch.update(width=1280, height=704, data_info={'img_hw': torch.tensor([[704, 1280]])})
    prepared.append(dict(case_id=custom['case_id'], prompt=prompt, seed=custom['seed'], batch=batch,
                         reference=None, reference_kind='observed-first-frame-consistency-no-future-GT', revisits=[]))
    tokenizer, encoder = get_tokenizer_and_text_encoder(config.text_encoder.text_encoder_name, 'cpu')
    encoder.eval().requires_grad_(False)
    with torch.no_grad():
        null, _ = _encode_prompts([''], tokenizer, encoder, config, 'cpu')
        for case in prepared:
            batch = case['batch']
            batch['y'], batch['mask'] = _encode_prompts([case['prompt']], tokenizer, encoder, config, 'cpu')
            batch['uncondition'] = null.clone()
            noise = diagnostic_noise((1, 128, 61, 22, 40), 'cuda', case['seed'], 61).cpu()
            case.update(noise=noise, input_sha256=hashes(batch), noise_sha256=tensor_sha256(noise))
    torch.save(dict(format='TTN-checkpoint-diagnostic-inputs-v1', cases=prepared), args.output / 'inputs.pt')
    write(args.output / 'inputs.json', [{k: c[k] for k in
          ('case_id', 'prompt', 'seed', 'reference_kind', 'input_sha256', 'noise_sha256')} for c in prepared])
    print('[Checkpoint eval] fixed inputs prepared once for every checkpoint', flush=True)


def generate(args):
    from worldttn.core import ANCHORS, TTNConfig
    from worldttn.cli import rollout, timed_cuda, to_device, seed_everything
    from worldttn.anchor import offline_state_dict
    from worldttn.checkpoint import apply_checkpoint_weights
    from worldttn.sana import build_sana, configure_cross_attention
    from worldttn.performance import configure_execution, ExecutionOptions
    from worldttn.evaluation import file_sha256, latent_metrics

    run, payload, path, digest = load_milestone(args.training_run, args.step)
    config = config_for(run)
    ttn = TTNConfig(**payload['config'])
    model = build_sana(config, ttn, run['base']['source'], 'cpu', dtype=torch.float32)
    if model.base_load_report['sha256'] != run['base']['sha256'] or payload['weight_scope'] != 'dit':
        raise ValueError('Full DiT/base identity mismatch')
    expected = offline_state_dict(model, 'dit')
    if expected.keys() != payload['adapter'].keys() or any(
            expected[k].shape != payload['adapter'][k].shape for k in expected):
        raise ValueError('Checkpoint names/shapes mismatch')
    apply_checkpoint_weights(model, payload)
    del payload, expected
    gc.collect()
    configure_execution(model, ExecutionOptions())
    configure_cross_attention(model, 'math')
    model.eval().requires_grad_(False).to('cuda')
    cases = torch.load(args.output / 'inputs.pt', map_location='cpu', weights_only=False)['cases']
    directory = args.output / f'step-{args.step:08d}'
    directory.mkdir()
    rows = []
    for index, case in enumerate(cases):
        seed_everything(case['seed'])
        batch, noise = to_device(case['batch'], 'cuda'), case['noise'].to('cuda')
        validate_pair(batch, noise, case)
        calls, hooks = {str(i): Counter() for i in ANCHORS}, []
        for i in ANCHORS:
            def count(module, positional, keywords, result, i=i):
                calls[str(i)]['clean' if keywords['clean'] else 'local'] += 1
            hooks.append(model.blocks[i].attn.state_ttt.register_forward_hook(count, with_kwargs=True))
        def progress(chunk):
            print('[Checkpoint eval chunk] ' + json.dumps(dict(step=args.step, case=case['case_id'],
                **{k: chunk[k] for k in ('chunk', 'start', 'end', 'seconds')})), flush=True)
        try:
            with torch.no_grad():
                (latent, runtime, chunks), timing = timed_cuda(lambda: rollout(
                    model, config, batch, 20, 4.5, 2, initial_noise=noise,
                    ttn_ablation='full', on_chunk=progress))
        finally:
            for hook in hooks:
                hook.remove()
        if runtime.commit_count != 21 or len(chunks) != 20 or not all(
                c['clean'] == 20 and c['local'] == 400 for c in calls.values()):
            raise ValueError('Incomplete full-StateTTT rollout')
        validate_pair(batch, noise, case)
        saved = latent.cpu()
        if not torch.equal(saved[:, :, :1], case['batch']['initial_latent']):
            raise ValueError('Observed latent changed')
        torch.save(dict(latents=saved, case_id=case['case_id'], step=args.step), directory / f'case-{index:03d}.pt')
        reference = case['reference']
        metric = latent_metrics(saved, reference, case['revisits']) if reference is not None else {
            'mean_future_observed_latent_drift': (saved[:, :, 1:] - saved[:, :, :1]).square().mean().item()}
        latent_errors = (saved - (reference if reference is not None else saved[:, :, :1])).square().mean((0,1,3,4))
        rows.append(dict(step=args.step, case_id=case['case_id'], reference_kind=case['reference_kind'],
                         input_sha256=case['input_sha256'], noise_sha256=case['noise_sha256'],
                         latent_metrics=metric,
                         latent_windows=time_windows([dict(latent_mse=x) for x in latent_errors[1:].tolist()]),
                         state_ttt_calls=calls, timing=timing))
        write(directory / 'summary.json', dict(status='running', checkpoint=str(path), checkpoint_sha256=digest, cases=rows))
        del batch, noise, latent, saved, runtime
        gc.collect()
        torch.cuda.empty_cache()
    if file_sha256(path) != digest:
        raise ValueError('Milestone changed during evaluation')
    write(directory / 'summary.json', dict(status='completed', checkpoint=str(path), checkpoint_sha256=digest, cases=rows))


def rgb_scores(actual, reference):
    """RGB [0,1] PSNR/SSIM; fixed 352x640 evaluation, 11x11 Gaussian valid SSIM."""
    if actual.shape != reference.shape or actual.ndim != 4 or actual.shape[1] != 3:
        raise ValueError('Expected identical [N,3,H,W] RGB frames')
    if min(actual.shape[-2:]) < 11 or not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError('Invalid RGB frames')
    if (actual.min() < 0 or actual.max() > 1 or reference.min() < 0 or reference.max() > 1):
        raise ValueError('Expected RGB in [0,1]')
    axis = torch.arange(11, dtype=actual.dtype, device=actual.device) - 5
    kernel = torch.exp(-axis.square() / (2 * 1.5**2))
    kernel /= kernel.sum()
    def gaussian(x):
        x = F.conv2d(x, kernel.view(1, 1, 1, 11).expand(3, 1, 1, 11), groups=3)
        return F.conv2d(x, kernel.view(1, 1, 11, 1).expand(3, 1, 11, 1), groups=3)
    mse, ssim = [], []
    for a, b in zip(actual.split(2), reference.split(2)):
        u, v = gaussian(a), gaussian(b)
        va, vb = (gaussian(a*a)-u*u).clamp_min(0), (gaussian(b*b)-v*v).clamp_min(0)
        cov = gaussian(a*b)-u*v
        score = ((2*u*v+.01**2)*(2*cov+.03**2) /
                 ((u*u+v*v+.01**2)*(va+vb+.03**2))).mean((1,2,3)).clamp(-1,1)
        mse.extend((a-b).square().mean((1,2,3)).tolist())
        ssim.extend(score.tolist())
    return [dict(rgb_mse=m, psnr_db=-10*math.log10(max(m, 1e-12)), ssim=s)
            for m, s in zip(mse, ssim)]


def time_windows(scores):
    if len(scores) != 60:
        raise ValueError('Expected 60 samples at raw frame IDs 8,16,...,480')
    def mean(rows):
        return {k: sum(r[k] for r in rows)/len(rows) for k in rows[0]}
    return {'all_future': mean(scores), **{f'{i*10}-{(i+1)*10}s': mean(scores[i*20:(i+1)*20]) for i in range(3)}}


def decode(args):
    from diffusion.model.builder import get_vae, vae_decode
    from sana.tools import resolve_hf_path
    from tools.ttn_decode_comparison import pixels
    import imageio.v2 as iio
    run = json.loads((args.training_run / 'run_config.json').read_text())
    config = config_for(run)
    dtype = getattr(torch, config.vae.weight_dtype)
    vae = get_vae(config.vae.vae_type, resolve_hf_path(config.vae.vae_pretrained), device='cuda', dtype=dtype, config=config.vae)
    if hasattr(vae, 'enable_tiling'):
        vae.enable_tiling()
    if hasattr(vae, 'use_framewise_encoding'):
        vae.use_framewise_encoding = vae.use_framewise_decoding = True
        vae.tile_sample_stride_num_frames = config.vae.tile_sample_stride_num_frames
        vae.tile_sample_min_num_frames = config.vae.tile_sample_min_num_frames
    def video(latent):
        with torch.inference_mode():
            return pixels(vae_decode(config.vae.vae_type, vae, latent.to('cuda', dtype)))
    def resized(frames):
        tensor = torch.from_numpy(frames.copy()).permute(0,3,1,2).float()/255
        return F.interpolate(tensor, size=(352,640), mode='bilinear', align_corners=False, antialias=True)
    cases = torch.load(args.output / 'inputs.pt', map_location='cpu', weights_only=False)['cases']
    rows = []
    for index, case in enumerate(cases):
        reference = case['reference']
        target = video(reference if reference is not None else case['batch']['initial_latent'])
        reference_rgb = resized(target[8::8]) if reference is not None else resized(target[:1]).expand(60,-1,-1,-1)
        del target
        for step in args.milestones:
            directory = args.output / f'step-{step:08d}'
            latent = torch.load(directory / f'case-{index:03d}.pt', map_location='cpu', weights_only=True)['latents']
            generated = video(latent)
            if generated.shape != (481,704,1280,3):
                raise ValueError('Unexpected decoded video shape')
            iio.mimwrite(directory / f'case-{index:03d}.mp4', generated, fps=16, codec='libx264', macro_block_size=1,
                         output_params=['-threads','2'])
            scores = rgb_scores(resized(generated[8::8]), reference_rgb)
            latent_summary = json.loads((directory / 'summary.json').read_text())['cases'][index]
            if latent_summary['case_id'] != case['case_id'] or latent_summary['input_sha256'] != case['input_sha256']:
                raise ValueError('Latent/decoded case identities disagree')
            row = dict(step=step, case_id=case['case_id'], reference_kind=case['reference_kind'],
                       windows=time_windows(scores), latent_windows=latent_summary['latent_windows'],
                       frame_ids=list(range(8,481,8)), per_sample=scores)
            rows.append(row)
            write(args.output / 'decoded-metrics.json', dict(status='running', cases=rows))
            print('[Checkpoint eval RGB] ' + json.dumps(dict(step=step,case=case['case_id'],**row['windows'])), flush=True)
            del generated, latent
            gc.collect()
            torch.cuda.empty_cache()
    write(args.output / 'decoded-metrics.json', dict(status='completed', cases=rows,
        protocol='RGB [0,1], resized to 352x640 with antialiased bilinear, samples at 2fps excluding observed frame; Gaussian SSIM11 sigma1.5, population covariance; PSNR capped120dB',
        missing_metrics=['LPIPS', 'VBench', 'estimated-camera-trajectory-error']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--milestones', type=int, nargs='+', default=[100,250])
    parser.add_argument('--phase', choices=['prepare','generate','decode'])
    parser.add_argument('--step', type=int)
    args = parser.parse_args()
    args.training_run, args.output = args.training_run.resolve(), args.output.resolve()
    if len(set(args.milestones)) != len(args.milestones) or min(args.milestones) < 1:
        parser.error('Milestones must be positive and distinct')
    if args.phase == 'generate' and args.step not in args.milestones:
        parser.error('Generation step must be one of the fixed milestones')
    if args.phase:
        guard = args.output / f'memory-{args.phase}-{args.step or "shared"}'
        guard.mkdir()
        torch.set_num_threads(2)
        with Budget(guard):
            {'prepare': prepare, 'generate': generate, 'decode': decode}[args.phase](args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), milestones=args.milestones,
                 ceiling_mib=25000, allocator_mib=20000, pid_stop_mib=22500,
                 scope='Two known training-dataset cases plus static user study; diagnostic, not held-out validation; no automatic best selection')
    write(args.output / 'status.json', state)
    base = [sys.executable, '-u', str(Path(__file__).resolve()), '--training-run', str(args.training_run),
            '--output', str(args.output), '--milestones', *map(str,args.milestones)]
    try:
        for phase, step in [('prepare',None), *[('generate',s) for s in args.milestones], ('decode',None)]:
            command = [*base, '--phase',phase] + (['--step',str(step)] if step else [])
            subprocess.run(command, cwd=REPO, check=True)
        decoded = json.loads((args.output / 'decoded-metrics.json').read_text())
        report = ['step | case | reference | 20-30s latent MSE | 20-30s PSNR(dB) | 20-30s SSIM',
                  '--- | --- | --- | --- | --- | ---']
        for row in decoded['cases']:
            tail = row['windows']['20-30s']
            mse = row['latent_windows']['20-30s']['latent_mse']
            report.append(f"{row['step']} | {row['case_id']} | {row['reference_kind']} | {mse:.6f} | {tail['psnr_db']:.4f} | {tail['ssim']:.6f}")
        (args.output / 'report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
        state['status'] = 'completed'
        print('\n'.join(report), flush=True)
    except BaseException as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        write(args.output / 'status.json', state)


if __name__ == '__main__':
    main()
