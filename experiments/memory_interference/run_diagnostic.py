"""GT-clean memory interference pilot on the unmodified H100 State-TTT model."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

from memory_math import MODES, clone_tree, sigma_indices, check_clean_pair
from memory_trace import MemoryTrace


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def state_identity(runtime):
    from worldttn.evaluation import tensor_sha256
    return [tensor_sha256(runtime.world_state), tensor_sha256(runtime.transition_fast),
            runtime.commit_count, runtime.revision]


def validate_config(config):
    if config.stage != 'C' or not config.state_ttt_rank or config.eta_psi != 0:
        raise ValueError('This protocol requires Stage C direct State-TTT with eta_psi=0')
    if not config.local_update or not config.persistent_meta:
        raise ValueError('Both State-TTT branches must exist before the four-way ablation')


def source_hashes(source):
    names = ('worldttn/anchor.py', 'worldttn/state_ttt.py', 'worldttn/runtime.py',
             'worldttn/session.py', 'worldttn/training.py', 'worldttn/core.py',
             'worldttn/evaluation.py', 'worldttn/checkpoint.py')
    return {name: hashlib.sha256((source / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for name in names}


def prepare(args):
    from worldttn.core import TTNConfig, BASE_REVISION
    from worldttn.evaluation import evaluation_config, load_evaluation_cases, tensor_sha256, file_sha256
    from worldttn.sana import load_sana_config, resolve_data_paths
    from worldttn.cli import seed_everything
    seed_everything(args.seed)
    checkpoint = Path(args.checkpoint).resolve()
    initial_stat = checkpoint.stat()
    fingerprint = file_sha256(checkpoint)
    if (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns) != (initial_stat.st_size, initial_stat.st_mtime_ns):
        raise RuntimeError('Checkpoint changed during hashing; wait for a completed transfer')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if payload.get('format') != 'TTN-SANA-WM-v0.1' or payload.get('base_revision') != BASE_REVISION:
        raise ValueError('Incompatible checkpoint family')
    ttn = TTNConfig(**payload['config'])
    validate_config(ttn)
    training = payload.get('distributed', {}).get('training_config')
    if not isinstance(training, dict):
        raise ValueError('Checkpoint lacks the exact saved training configuration')
    config = evaluation_config({'training': training}, args.sana_config)
    # Transfer only filesystem locations. All trained preprocessing fields stay restored.
    local = load_sana_config(args.sana_config)
    resolve_data_paths(local, args.dataset_root)
    for name in ('data_dir', 'vae_cache_dir', 'hf_dataset_local_dir'):
        setattr(config.data, name, getattr(local.data, name))
    options = SimpleNamespace(frames=args.frames, eval_cases=1, seed=args.seed, fixed_cases=None,
        revisit_min_gap=30, revisit_distance_fraction=.02, revisit_angle_deg=5., revisit_max_pairs=5)
    cases, rejected, _ = load_evaluation_cases(config, options)
    case = cases[0]
    from diffusion.model.builder import get_tokenizer_and_text_encoder
    from train_video_scripts.train_sana_wm_stage1 import _encode_prompts
    tokenizer, encoder = get_tokenizer_and_text_encoder(config.text_encoder.text_encoder_name, 'cpu')
    encoder.eval().requires_grad_(False)
    with torch.no_grad():
        y, mask = _encode_prompts([case['prompt']], tokenizer, encoder, config, 'cpu')
    del encoder, tokenizer
    gc.collect()
    case['y'], case['mask'] = y, mask
    hashes = {name: tensor_sha256(case[name]) for name in ('reference', 'camera', 'y', 'mask')}
    if case['plucker'] is not None:
        hashes['plucker'] = tensor_sha256(case['plucker'])
    identity = dict(checkpoint=str(checkpoint), checkpoint_sha256=fingerprint, step=int(payload['step']),
        checkpoint_stat=[initial_stat.st_size, initial_stat.st_mtime_ns],
        ttn_config=ttn.to_dict(), source_hashes=source_hashes(Path(args.source)), source_hash_scheme='SHA256 LF-normalized',
        case_id=case['case_id'], prompt=case['prompt'], inputs=hashes, seed=args.seed,
        frames=args.frames, protocol='training-example GT-clean history; conditional velocity probes; no rollout quality claim',
        prefix_probes=args.probe_count, selected_chunks=args.chunks, requested_sigmas=args.sigmas,
        camera_cache_chunks=2, precision='FP32 saved model masters; BF16 autocast; FP32 S math',
        torch=torch.__version__, rejected=rejected)
    return payload, config, ttn, case, identity


@torch.no_grad()
def run_mode(args, model, config, case, mode, output):
    from worldttn.session import TTNSession
    from worldttn.training import SANAFlowLoss, chunk_ranges
    from worldttn.core import ANCHORS
    from worldttn.evaluation import tensor_sha256
    from worldttn.cli import to_device, seed_everything
    from torch.utils.data import default_collate
    seed_everything(args.seed)
    clean = case['reference'].to('cuda', dtype=torch.float32)
    camera, y = case['camera'].to('cuda'), case['y'].to('cuda')
    mask = case['mask'].to('cuda')
    info = to_device(default_collate([case['info']]), 'cuda')
    extras = {} if case['plucker'] is None else {'chunk_plucker': case['plucker'][None].to('cuda')}
    session = TTNSession(model, camera, clean.shape[-1], clean.shape[-2], extras=extras, ablation=mode)
    session.reset(1)
    flow = SANAFlowLoss(config)
    indices, sigmas = sigma_indices(flow.scheduler.sigmas, args.sigmas)
    generator = torch.Generator(device='cuda').manual_seed(args.seed + 9001)
    epsilon = torch.randn(clean.shape, generator=generator, device='cuda', dtype=torch.float32)
    result = dict(mode=mode, clean_state_hashes=[], prefix_hashes=[], memory=[], velocity=[],
                  production_indices=indices, production_sigmas=sigmas, repeatability=None)
    trace = MemoryTrace(model, ANCHORS, args.probe_count)
    ranges = chunk_ranges(args.frames)
    cache = [[None] * 10 for _ in model.blocks]
    cumulative = torch.eye(model.ttn_system.config.head_dim, device='cuda').expand_as(session.runtime.world_state).clone()
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            trace.phase = 'prefill'
            session.prefill(clean[:, :, :1], y, mask, info)
            if set(trace.references) != set(range(len(ANCHORS))):
                raise AssertionError('Observed prefix did not traverse all five anchors')
            result['prefix_hashes'] = [trace.prefix_hashes[i] for i in range(len(ANCHORS))]
            for chunk, (start, end) in enumerate(ranges, 1):
                trace.chunk, trace.phase = chunk, 'noisy'
                context = session.begin_chunk(start, end)
                with torch.autocast('cuda', enabled=False):
                    factors = context.live_factors
                    if factors is None:
                        raise AssertionError('Expected the actual live Cayley transport')
                    rotation = factors.right(torch.eye(cumulative.shape[-1], device='cuda').expand_as(cumulative))
                    torch.testing.assert_close(context.predicted, session.runtime.world_state @ rotation,
                                               rtol=1e-4, atol=2e-6)
                    cumulative = cumulative @ rotation
                trace.transport = cumulative
                if chunk in args.chunks:
                    before = state_identity(session.runtime)
                    for index, sigma in zip(indices, sigmas):
                        timestep = torch.full((1, 1, end-start), index, device='cuda', dtype=torch.long)
                        captured = []

                        def evaluate():
                            return flow(session, clean[:, :, start:end], timestep, epsilon[:, :, start:end],
                                y, context, clone_tree(cache), start, end, mask, info,
                                torch.ones((1, 1, end-start, 1, 1), device='cuda'),
                                on_prediction=lambda value: captured.append(value.detach().clone()))

                        rng_before = torch.cuda.get_rng_state()
                        loss = evaluate()
                        if len(captured) != 1:
                            raise AssertionError('Production loss called the model more than once')
                        prediction = captured.pop()
                        target = epsilon[:, :, start:end] - clean[:, :, start:end]
                        manual = (prediction.float()-target.float()).square().mean()
                        torch.testing.assert_close(loss.mean(), manual, rtol=2e-5, atol=2e-6)
                        if result['repeatability'] is None:
                            rng_after = torch.cuda.get_rng_state()
                            torch.cuda.set_rng_state(rng_before)
                            repeat_loss = evaluate()
                            torch.cuda.set_rng_state(rng_after)
                            repeated = captured.pop()
                            torch.testing.assert_close(repeated, prediction, rtol=1e-5, atol=1e-6)
                            torch.testing.assert_close(repeat_loss, loss, rtol=1e-5, atol=1e-6)
                            result['repeatability'] = dict(passed=True,
                                max_abs_difference=float((repeated-prediction).abs().max()))
                        result['velocity'].append(dict(chunk=chunk, start=start, end=end,
                            sigma=sigma, production_index=index, mse=float(loss.mean())))
                    if before != state_identity(session.runtime):
                        raise AssertionError('Noisy forward mutated persistent S/psi or committed history')
                trace.phase = 'clean'
                _, cache = session.clean_forward(clean[:, :, start:end], y, context, cache,
                    start, end, mask, info, cached_chunks=2)
                result['clean_state_hashes'].append(state_identity(session.runtime))
                result['memory'] = trace.rows
                result['calls'] = dict(trace.calls)
                atomic_json(output / (mode + '.partial.json'), result)
                print(f'[Memory diagnostic] mode={mode} chunk={chunk}/{len(ranges)} commits={session.runtime.commit_count}', flush=True)
        expected = len(ranges)*len(ANCHORS) if mode in ('full', 'no-local') else 0
        if trace.calls['clean'] != expected:
            raise AssertionError('Clean TTT call count is incorrect')
        noisy_calls = (len(args.chunks)*len(indices)+1)*len(ANCHORS) if mode in ('full', 'no-persistent') else 0
        if trace.calls['local'] != noisy_calls or result['repeatability'] is None:
            raise AssertionError('Local TTT call count or repeatability acceptance failed')
        result.update(status='completed', seconds=time.perf_counter()-start_time,
                      gpu_peak_gib=torch.cuda.max_memory_allocated()/(1 << 30))
        atomic_json(output / (mode + '.json'), result)
    finally:
        trace.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--base-weights', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--sana-config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--modes', nargs='+', choices=MODES, required=True)
    parser.add_argument('--frames', type=int, default=121)
    parser.add_argument('--chunks', nargs='+', type=int, default=[2, 6, 11, 21, 31, 40])
    parser.add_argument('--sigmas', nargs='+', type=float, default=[.2, .5, .8])
    parser.add_argument('--probe-count', type=int, default=64)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()
    args.source = str(Path(args.source).resolve())
    sys.path.insert(0, args.source)
    args.sana_config = args.sana_config or str(Path(args.source)/'configs/worldttn/sana_teacher_single_gpu.yaml')
    chunks = (args.frames-1)//3
    if args.frames < 7 or args.frames % 3 != 1 or len(set(args.chunks)) != len(args.chunks) or any(c < 2 or c > chunks for c in args.chunks):
        parser.error('Use 1+3n frames and distinct future probe chunks in [2,n]')
    if len(set(args.modes)) != len(args.modes):
        parser.error('Modes must be distinct')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Each diagnostic process must see exactly one authorized GPU')
    torch.cuda.set_per_process_memory_fraction(min(.9, (36 << 30)/torch.cuda.get_device_properties(0).total_memory))
    payload, config, ttn, case, identity = prepare(args)
    atomic_json(output/'manifest.json', identity)
    from worldttn.sana import build_sana, configure_cross_attention
    from worldttn.checkpoint import read_checkpoint, apply_checkpoint_weights
    model = build_sana(config, ttn, args.base_weights, dtype=torch.float32)
    # Reuse the production key/shape/base/config validator, rather than permissive loading.
    del payload
    payload = read_checkpoint(args.checkpoint, model)
    current_stat = Path(args.checkpoint).stat()
    if [current_stat.st_size, current_stat.st_mtime_ns] != identity['checkpoint_stat'] or int(payload['step']) != identity['step']:
        raise RuntimeError('Checkpoint changed between identity collection and model loading')
    apply_checkpoint_weights(model, payload)
    del payload
    model.requires_grad_(False).eval()
    configure_cross_attention(model, 'auto')
    gc.collect()
    records = {}
    for mode in args.modes:
        records[mode] = run_mode(args, model, config, case, mode, output)
        gc.collect()
        torch.cuda.empty_cache()
    if set(args.modes) == {'full', 'no-local'}:
        check_clean_pair(records['full'], records['no-local'])
    if set(args.modes) == {'no-persistent', 'no-ttt'}:
        check_clean_pair(records['no-persistent'], records['no-ttt'])
    atomic_json(output/'acceptance.json', dict(status='passed', modes=args.modes,
        clean_pair_isolation=True, checkpoint_sha256=identity['checkpoint_sha256'], inputs=identity['inputs']))


if __name__ == '__main__':
    main()
