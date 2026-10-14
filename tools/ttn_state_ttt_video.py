"""Fixed State-TTT milestone, static 30s on/off pair with a conservative GPU budget."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import threading

import torch

from worldttn.core import BASE_REVISION, TTNConfig
from worldttn.evaluation import file_sha256, tensor_sha256

REPO = Path(__file__).resolve().parents[1]


def process_memory_mib(csv, pid):
    values = []
    for line in csv.splitlines():
        fields = [value.strip() for value in line.split(',')]
        if len(fields) == 2 and fields[0] == str(pid):
            try:
                values.append(int(fields[1]))
            except ValueError as error:
                raise ValueError('Per-process GPU memory measurement unavailable') from error
    return sum(values)


class VideoMemoryBudget:
    """Allocator cap is hard for PyTorch; PID sampling is NOT driver isolation."""
    ceiling_mib, allocator_mib, stop_mib = 23000, 18000, 21000

    def __init__(self, output):
        self.output = Path(output)
        self.peak = 0
        self.stop = threading.Event()

    def sample(self):
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                 '--format=csv,noheader,nounits'], check=True,
                                capture_output=True, text=True, timeout=5)
        value = process_memory_mib(result.stdout, os.getpid())
        self.peak = max(self.peak, value)
        return value

    def report(self, status, **extra):
        report = {'status': status, 'pid': os.getpid(), 'requested_ceiling_mib': self.ceiling_mib,
                  'torch_allocator_cap_mib': self.allocator_mib, 'pid_stop_threshold_mib': self.stop_mib,
                  'sampled_process_peak_mib': self.peak, 'sampling_interval_seconds': 0.25,
                  'limitation': 'PID sampling can miss transient peaks; no driver-level memory partition', **extra}
        temporary = self.output / 'memory.json.tmp'
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(self.output / 'memory.json')

    def prepare(self):
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES=5')
        torch.cuda.set_device(0)
        if torch.cuda.get_allocator_backend() != 'native':
            raise RuntimeError('Memory cap requires the native PyTorch CUDA allocator')
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(self.allocator_mib * 2**20 / total, 0)
        free, _ = torch.cuda.mem_get_info()
        if free < (self.ceiling_mib + 2048) * 2**20:
            raise RuntimeError('Insufficient free GPU memory plus training headroom; inference not started')
        self.sample()  # Fail before model loading if PID accounting is unsupported.

    def poll_once(self):
        try:
            if self.sample() >= self.stop_mib:
                raise RuntimeError('Generation PID reached the conservative GPU stop threshold')
        except Exception as error:
            try:
                self.report('stopped', reason=str(error))
                print(f'[TTN video memory] {error}; stopping only generation PID {os.getpid()}', flush=True)
            finally:
                os._exit(75)

    def __enter__(self):
        self.prepare()
        self.report('running')

        def watch():
            while not self.stop.wait(0.25):
                self.poll_once()

        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, error, traceback):
        self.stop.set()
        self.thread.join(timeout=6)
        self.report('completed' if kind is None else 'failed', error=None if error is None else str(error))


def load_milestone(directory, step):
    directory = Path(directory).resolve()
    path = directory / 'models' / f'step-{step:08d}.pt'
    receipt = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    run = json.loads((directory / 'run_config.json').read_text(encoding='utf-8'))
    if (receipt.get('status') != 'verified' or receipt.get('step') != step or payload.get('step') != step
            or payload.get('format') != 'TTN-SANA-WM-v0.1' or payload.get('base_revision') != BASE_REVISION
            or receipt['checkpoint_id'] != payload.get('distributed', {}).get('checkpoint_id')
            or payload.get('base_sha256') != run['base']['sha256']):
        raise ValueError('Fixed milestone receipt/model/base identity mismatch')
    ttn = TTNConfig(**payload['config'])
    if ttn.state_ttt_rank != 64 or ttn.stage != 'C' or ttn.camera_attention != 'sana':
        raise ValueError('This experiment requires the trained rank64 Stage-C State-TTT checkpoint')
    return run, payload, path, file_sha256(path)


def generate_pair(model, config, batch, noise, output, *, steps, cfg_scale):
    from worldttn.cli import rollout, seed_everything, timed_cuda, to_device
    output = Path(output)
    rows = []
    input_hash = {key: tensor_sha256(value) for key, value in batch.items() if isinstance(value, torch.Tensor)}
    noise_hash = tensor_sha256(noise)
    for ablation in ('no-ttt', 'full'):
        seed_everything(3407)
        gpu_batch, gpu_noise = to_device(batch, 'cuda'), to_device(noise, 'cuda')

        def progress(row):
            print('[TTN video chunk] ' + json.dumps({'method': ablation, **{
                key: row[key] for key in ('chunk', 'start', 'end', 'seconds')}}), flush=True)

        with torch.no_grad():
            (latents, runtime, chunks), timing = timed_cuda(lambda: rollout(
                model, config, gpu_batch, steps, cfg_scale, 2, initial_noise=gpu_noise,
                ttn_ablation=ablation, on_chunk=progress))
        if not torch.equal(latents[:, :, :1].cpu(), batch['initial_latent']):
            raise ValueError('Rollout changed the observed first latent')
        if any(tensor_sha256(gpu_batch[key]) != value for key, value in input_hash.items()):
            raise ValueError('Rollout changed paired conditioning')
        if tensor_sha256(gpu_noise) != noise_hash:
            raise ValueError('Rollout changed paired noise')
        torch.save({'latents': latents.cpu(), 'method': ablation}, output / f'{ablation}.pt')
        row = {'method': ablation, 'input_sha256': input_hash, 'initial_noise_sha256': noise_hash,
               'timing': timing, 'commits': runtime.commit_count, 'chunks': chunks}
        rows.append(row)
        with (output / 'episodes.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        del gpu_batch, gpu_noise, latents, runtime
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def decode_pair(config, output, step):
    from diffusion.model.builder import get_vae, vae_decode
    from sana.tools import resolve_hf_path
    from tools.ttn_decode_comparison import pixels
    import imageio.v2 as iio
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    imageio_ffmpeg.get_ffmpeg_exe()
    output = Path(output)
    videos = output / 'videos'
    videos.mkdir(exist_ok=False)
    dtype = getattr(torch, config.vae.weight_dtype)
    vae = get_vae(config.vae.vae_type, resolve_hf_path(config.vae.vae_pretrained),
                  device='cuda', dtype=dtype, config=config.vae)
    if hasattr(vae, 'enable_tiling'):
        vae.enable_tiling()
    decoded = []
    for method in ('no-ttt', 'full'):
        latent = torch.load(output / f'{method}.pt', map_location='cpu', weights_only=True)['latents']
        with torch.no_grad():
            video = pixels(vae_decode(config.vae.vae_type, vae, latent.to('cuda', dtype)))
        if video.shape != (481, 704, 1280, 3):
            raise ValueError(f'Unexpected decoded shape: {video.shape}')
        iio.mimwrite(videos / f'{method}.mp4', video, fps=16, codec='libx264', macro_block_size=1)
        decoded.append(video)
        del latent
        torch.cuda.empty_cache()
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 24)
    except OSError:
        font = ImageFont.load_default()
    with iio.get_writer(videos / 'comparison.mp4', fps=16, codec='libx264', macro_block_size=1) as writer:
        for i, (left, right) in enumerate(zip(*decoded)):
            canvas = Image.new('RGB', (2560, 768))
            draw = ImageDraw.Draw(canvas)
            for x, frame, name in ((0, left, 'TTT OFF'), (1280, right, 'State-TTT ON')):
                canvas.paste(Image.fromarray(frame), (x, 64))
                draw.text((x+12, 5), f'TTN step{step} | {name}', fill='white', font=font)
                draw.text((x+12, 34), f't={i/16:.2f}s | same checkpoint/inputs/noise', fill='white', font=font)
            writer.append_data(np.asarray(canvas))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', required=True, type=Path)
    parser.add_argument('--step', type=int, default=25)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--cfg-scale', type=float, default=4.5)
    args = parser.parse_args(argv)
    if args.steps < 1 or not 1 <= args.cfg_scale < float('inf'):
        parser.error('Positive solver steps and finite CFG >=1 required')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'step': args.step, 'methods': ['no-ttt', 'full'],
                'solver_steps': args.steps, 'cfg_scale': args.cfg_scale,
                'latent_frames': 61, 'raw_frames': 481, 'fps': 16, 'refiner': None,
                'precision': 'FP32 saved weights/core; BF16 autocast; no inference FP8 conversion',
                'scope': 'one static qualitative case; no future GT or video quality claim'}
    try:
        with VideoMemoryBudget(args.output):
            run, payload, path, digest = load_milestone(args.training_run, args.step)
            manifest.update(checkpoint=str(path), checkpoint_sha256=digest)
            from tools.ttn_custom_inference import load_case, make_geometry, encode_first_frame
            from worldttn.sana import build_sana, configure_cross_attention
            from worldttn.evaluation import evaluation_config, diagnostic_noise
            from worldttn.cli import seed_everything
            from worldttn.anchor import offline_state_dict
            from worldttn.checkpoint import apply_checkpoint_weights
            from diffusion.model.builder import get_tokenizer_and_text_encoder
            from train_video_scripts.train_sana_wm_stage1 import _encode_prompts

            source = run['arguments'].get('sana_config') or 'configs/worldttn/sana_teacher_single_gpu.yaml'
            config = evaluation_config(run, REPO / source)
            ttn = TTNConfig(**payload['config'])
            case, image, prompt = load_case(REPO / 'assets/worldttn/study_static/case.json')
            if case['latent_frames'] != 61 or case['camera']['trajectory'] != 'static':
                raise ValueError('Requires the original 30s static study case')
            seed_everything(case['seed'])
            batch = make_geometry(case, config.vae.vae_stride)
            batch['initial_latent'] = encode_first_frame(case, image, config, args.output)
            tokenizer, encoder = get_tokenizer_and_text_encoder(config.text_encoder.text_encoder_name, 'cpu')
            encoder.eval().requires_grad_(False)
            batch['y'], batch['mask'] = _encode_prompts([prompt], tokenizer, encoder, config, 'cpu')
            batch['uncondition'], _ = _encode_prompts([''], tokenizer, encoder, config, 'cpu')
            del tokenizer, encoder
            gc.collect()
            batch.update(width=1280, height=704, data_info={'img_hw': torch.tensor([[704, 1280]])})
            noise = diagnostic_noise((1, config.vae.vae_latent_dim, 61, 22, 40), 'cuda', case['seed']).cpu()
            torch.save(batch, args.output / 'input-bundle.pt')
            (args.output / 'prompt.txt').write_text(prompt + '\n', encoding='utf-8')
            manifest['case'] = case
            (args.output / 'summary.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

            model = build_sana(config, ttn, run['base']['source'], 'cpu', dtype=torch.float32)
            if model.base_load_report['sha256'] != run['base']['sha256'] or payload.get('weight_scope') != 'dit':
                raise ValueError('Base/full DiT checkpoint mismatch')
            expected = offline_state_dict(model, 'dit')
            if expected.keys() != payload['adapter'].keys() or any(
                    expected[k].shape != payload['adapter'][k].shape for k in expected):
                raise ValueError('Checkpoint parameter names/shapes mismatch')
            apply_checkpoint_weights(model, payload)
            del payload, expected
            configure_cross_attention(model, 'math')
            model.eval().requires_grad_(False).to('cuda')
            manifest['episodes'] = generate_pair(model, config, batch, noise, args.output,
                                                 steps=args.steps, cfg_scale=args.cfg_scale)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            decode_pair(config, args.output, args.step)
            if file_sha256(path) != digest:
                raise ValueError('Fixed checkpoint changed during inference')
            manifest['status'] = 'completed'
    except BaseException as error:
        manifest.update(status='failed', error=str(error))
        raise
    finally:
        (args.output / 'summary.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'[TTN video completed] {args.output / "videos/comparison.mp4"}', flush=True)


if __name__ == '__main__':
    main()
