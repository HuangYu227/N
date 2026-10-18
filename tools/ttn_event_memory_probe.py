"""Frozen baseline read directions and matched-budget, one-chunk interventions.

Production model/sampler code is unchanged. All patches are scoped to this
single-process diagnostic. Future GT is used only after a predicted chunk.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import ttn_event_memory_experiment as experiment
from tools.ttn_event_memory_probe_math import POLICIES, bank_fingerprint, counterfactual_reads
from worldttn.evaluation import tensor_sha256
from worldttn.event_memory import EventMemoryBank

# Sampler indices are zero-based; event-memory chunk IDs are one larger.
TARGETS = {19: (58, 61), 29: (88, 91), 36: (109, 112)}


class ProbeComplete(Exception):
    """Successful intentional stop after the requested chunk, not a failure."""


class FrozenReadProbe:
    def __init__(self, targets, *, frozen=None, policy=None, stop_index=None):
        if not targets or any(type(i) is not int or i not in TARGETS for i in targets):
            raise ValueError('unsupported diagnostic target')
        if (frozen is None) != (policy is None) or policy is not None and policy not in POLICIES:
            raise ValueError('frozen injection needs one supported policy')
        if stop_index is not None and stop_index not in targets:
            raise ValueError('stop index must be a target')
        self.targets = tuple(targets)
        self.memory_chunks = {i + 1 for i in targets}
        self.frozen, self.policy, self.stop_index = frozen, policy, stop_index
        self.queries, self.counts, self.snapshots, self.outputs = {}, {}, {}, {}
        self.bank_hashes, self.injected_keys, self.audit = {}, [], []

    @contextmanager
    def install(self, sampler_class):
        original_select = EventMemoryBank.select
        original_read = EventMemoryBank.read
        original_sample = sampler_class.sample_chunks
        owner = self

        def select(bank, batch, anchor, query, chunk_id):
            if chunk_id in owner.memory_chunks:
                key = (chunk_id, anchor, batch)
                if key in owner.queries:
                    raise AssertionError('query selection unexpectedly repeated within chunk')
                owner.queries[key] = query.detach().cpu().clone()
            return original_select(bank, batch, anchor, query, chunk_id)

        def read(bank, state, selections, anchor, alpha_s=.5, eps=1e-6):
            chunks = {selection.chunk_id for selection in selections}
            if len(chunks) != 1:
                raise AssertionError('CFG branches disagree on chunk')
            chunk = chunks.pop()
            if chunk not in owner.memory_chunks:
                return state, {'active': False, 'anchor': anchor, 'probe': 'shadow baseline'}
            count_key = (chunk, anchor)
            ordinal = owner.counts.get(count_key, 0)
            owner.counts[count_key] = ordinal + 1
            key = f'{chunk}:{anchor}:{ordinal}'
            queries = torch.stack([owner.queries[(chunk, anchor, b)] for b in range(state.shape[0])])
            first = chunk not in owner.bank_hashes
            if first:
                owner.bank_hashes[chunk] = bank_fingerprint(bank)
            if owner.frozen is None:
                # read_fn captured before patch: counterfactuals cannot recurse.
                values = counterfactual_reads(bank, state, queries, anchor, chunk,
                    alpha_s=alpha_s, eps=eps, read_fn=original_read)
                owner.snapshots[key] = {
                    'anchor': anchor, 'chunk_id': chunk, 'ordinal': ordinal,
                    'baseline_state_sha256': tensor_sha256(state),
                    'query_sha256': tensor_sha256(queries),
                    'matched_deltas': {name: value.detach().cpu().clone()
                                       for name, value in values['matched_deltas'].items()},
                    'common_norm': values['common_norm'].detach().cpu().clone(),
                    'candidate_scores': values['candidate_scores'],
                    'diagnostics': values['diagnostics'], 'reports': values['reports'],
                }
                # Identity matters: state+0 would trigger extra anchor GEMMs.
                return state, {'active': False, 'anchor': anchor, 'probe': 'counterfactuals only'}
            saved = owner.frozen['snapshots'][key]
            if first:
                if owner.bank_hashes[chunk] != owner.frozen['bank_hashes'][chunk]:
                    raise AssertionError('target archive differs from frozen baseline')
                if (tensor_sha256(state) != saved['baseline_state_sha256']
                        or tensor_sha256(queries) != saved['query_sha256']):
                    raise AssertionError('first intervention does not start from identical state/query')
            delta = saved['matched_deltas'][owner.policy].to(state.device)
            if delta.shape != state.shape or delta.dtype != state.dtype or not torch.isfinite(delta).all():
                raise AssertionError('frozen direction shape/dtype/finite mismatch')
            norms, state_norms = delta.norm(dim=(-1, -2)), state.norm(dim=(-1, -2))
            if not torch.allclose(norms.cpu(), saved['common_norm'], atol=1e-6, rtol=1e-5):
                raise AssertionError('frozen head budgets changed')
            if (norms > .05 * state_norms + 1e-6).any():
                raise AssertionError('frozen direction exceeds 5% of live state; no silent clipping')
            owner.injected_keys.append(key)
            owner.audit.append({'key': key, 'policy': owner.policy,
                'head_norms': norms.cpu().tolist(),
                'relative_head_norms': (norms / state_norms.clamp_min(eps)).cpu().tolist()})
            changed = state + delta
            if not torch.isfinite(changed).all():
                raise AssertionError('nonfinite injected state')
            return changed, {'active': bool(norms.max() > 0), 'anchor': anchor,
                             'probe': 'fixed baseline direction', 'policy': owner.policy, 'key': key}

        def sample(sampler, latents, *args, **kwargs):
            stream = original_sample(sampler, latents, *args, **kwargs)
            try:
                for item in stream:
                    index, chunk, start, end = item
                    if index in owner.targets:
                        if (start, end) != TARGETS[index]:
                            raise AssertionError('sampler chunk boundaries changed')
                        owner.outputs[index] = {'latents': chunk.detach().cpu().clone(),
                            'start': start, 'end': end,
                            'prefix_sha256': tensor_sha256(latents[:, :, :start])}
                    yield item
                    if index == owner.stop_index:
                        raise ProbeComplete()
            finally:
                stream.close()

        with patch.object(EventMemoryBank, 'select', select), patch.object(EventMemoryBank, 'read', read), \
                patch.object(sampler_class, 'sample_chunks', sample):
            yield self


def load_context(manifest):
    from worldttn.anchor import offline_state_dict
    from worldttn.checkpoint import apply_checkpoint_weights
    from worldttn.cli import seed_everything
    from worldttn.core import TTNConfig
    from worldttn.performance import ExecutionOptions, configure_execution
    from worldttn.sana import build_sana, configure_cross_attention
    experiment.check_sources(manifest)
    for path, digest in [('checkpoint', 'checkpoint_sha256'), ('input_path', 'input_file_sha256'),
                         ('run_config', 'run_config_sha256')]:
        if experiment.sha256(manifest[path]) != manifest[digest]:
            raise ValueError(f'pinned {path} changed')
    if torch.cuda.device_count() != 1:
        raise ValueError('probe worker must see exactly one allowed GPU')
    torch.cuda.set_device(0)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(32 * 2**30 / total)
    if torch.cuda.mem_get_info()[0] < 40 * 2**30:
        raise RuntimeError('probe needs at least40GiB free before model loading')
    payload = torch.load(manifest['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    identity = manifest['checkpoint_identity']
    if any(payload.get(key) != identity[key] for key in ('step', 'base_revision', 'base_sha256', 'weight_scope', 'stage')):
        raise ValueError('checkpoint identity changed')
    if payload.get('distributed', {}).get('checkpoint_id') != identity['checkpoint_id']:
        raise ValueError('checkpoint ID changed')
    ttn = TTNConfig(**payload['config'])
    if ttn.state_ttt_rank != 64 or ttn.stage != 'C':
        raise ValueError('enhanced StageC rank64 checkpoint required')
    run = json.loads(Path(manifest['run_config']).read_text())
    config = experiment.configuration(run)
    seed_everything(3407)
    model = build_sana(config, ttn, run['base']['source'], 'cpu', dtype=torch.float32)
    if model.base_load_report['sha256'] != identity['base_sha256']:
        raise ValueError('SANA base changed')
    expected = offline_state_dict(model, 'dit')
    if expected.keys() != payload['adapter'].keys() or any(expected[k].shape != payload['adapter'][k].shape for k in expected):
        raise ValueError('checkpoint model keys/shapes mismatch')
    apply_checkpoint_weights(model, payload)
    del payload, expected
    gc.collect()
    configure_execution(model, ExecutionOptions())
    configure_cross_attention(model, 'math')
    model.eval().requires_grad_(False).to('cuda')
    packet = torch.load(manifest['input_path'], map_location='cpu', weights_only=False)
    if 'clean_latents' in packet['batch']:
        raise ValueError('future GT forbidden in inference conditioning')
    row = manifest['case']
    if ({k: tensor_sha256(packet['batch'][k]) for k in row['input_sha256']} != row['input_sha256']
            or tensor_sha256(packet['noise']) != row['initial_noise_sha256']):
        raise ValueError('conditioning/noise differs from pinned protocol')
    return model, config, packet


def run_rollout(model, config, packet, manifest, probe):
    from diffusion.scheduler.self_forcing_flow_euler_sampler import SelfForcingFlowEulerCamCtrl
    from worldttn.cli import rollout, seed_everything, to_device
    from worldttn.event_memory import EventMemoryOptions
    seed_everything(manifest['case']['seed'])
    batch, noise = to_device(packet['batch'], 'cuda'), packet['noise'].to('cuda')
    options = EventMemoryOptions(mode='state-replay', **manifest['event_memory'])
    def progress(row):
        print('[Frozen probe chunk] ' + json.dumps({k: row[k] for k in ('chunk', 'start', 'end', 'seconds')}), flush=True)
    result = None
    with torch.no_grad(), probe.install(SelfForcingFlowEulerCamCtrl):
        try:
            result = rollout(model, config, batch, manifest['solver_steps'], manifest['cfg_scale'],
                             manifest['cached_blocks'], initial_noise=noise, on_chunk=progress,
                             memory_options=options)
        except ProbeComplete:
            if probe.stop_index is None or probe.stop_index not in probe.outputs:
                raise AssertionError('unexpected early stop')
    if tensor_sha256(noise) != manifest['case']['initial_noise_sha256']:
        raise AssertionError('shared noise mutated')
    if {k: tensor_sha256(batch[k]) for k in manifest['case']['input_sha256']} != manifest['case']['input_sha256']:
        raise AssertionError('conditioning mutated')
    del batch, noise
    torch.cuda.empty_cache()
    return result


def capture(args):
    experiment.prepare(args)
    root = args.output.resolve()
    experiment.write_json(root / 'capture-status.json', {'status': 'running', 'pid': os.getpid()})
    manifest = json.loads((root / 'experiment.json').read_text())
    manifest['probe_protocol'] = {
        'targets': {str(i): {'latent_start': start, 'latent_end_exclusive': end,
                            'nominal_endpoint_seconds': (end - 1) / 2} for i, (start, end) in TARGETS.items()},
        'topk': 1, 'policies': list(POLICIES),
        'budget': 'per branch/head minimum raw norm across all four policies; no amplification',
        'directions': 'frozen on shadow baseline at each anchor/solver call; not adaptive retrieval rollout',
        'GT': 'post-hoc chunk metrics only; never used for selection or direction construction',
    }
    experiment.write_json(root / 'experiment.json', manifest)
    model, config, packet = load_context(manifest)
    probe = FrozenReadProbe(tuple(TARGETS))
    from worldttn.cli import timed_cuda
    result, timing = timed_cuda(lambda: run_rollout(model, config, packet, manifest, probe))
    latents, runtime, records = result
    historical = torch.load(Path(manifest['reference_root']) / manifest['case']['name'] / 'ttn-latents.pt',
                            map_location='cpu', weights_only=True)['latents']
    if not torch.equal(latents.cpu(), historical):
        raise AssertionError('shadow baseline does not exactly reproduce original latents')
    if len(records) != 40 or runtime.commit_count != 41:
        raise AssertionError('shadow baseline incomplete')
    expected = {f'{i+1}:{anchor}:{call}' for i in TARGETS for anchor in (11, 15, 19) for call in (0, 1)}
    if set(probe.snapshots) != expected:
        raise AssertionError('unexpected eligible-call schedule; inspect before changing protocol')
    frozen = {'snapshots': probe.snapshots, 'bank_hashes': probe.bank_hashes, 'outputs': probe.outputs}
    torch.save(frozen, root / 'frozen-probe.pt')
    metadata = {key: {k: v for k, v in row.items() if k not in ('matched_deltas', 'common_norm')}
                | {'common_norm': row['common_norm'].tolist()} for key, row in probe.snapshots.items()}
    experiment.write_json(root / 'capture.json', {'status': 'completed', 'full_baseline_exact': True,
        'frozen_sha256': experiment.sha256(root / 'frozen-probe.pt'),
        'timing': timing, 'snapshots': metadata, 'bank_hashes': probe.bank_hashes})
    experiment.write_json(root / 'capture-status.json', {'status': 'completed', 'pid': os.getpid()})
    print('[Frozen probe capture passed] ' + str(root), flush=True)


def worker(args):
    root = args.output.resolve()
    manifest = json.loads((root / 'experiment.json').read_text())
    capture_result = json.loads((root / 'capture.json').read_text())
    if capture_result['status'] != 'completed' or not capture_result['full_baseline_exact']:
        raise ValueError('validated capture required')
    if experiment.sha256(root / 'frozen-probe.pt') != capture_result['frozen_sha256']:
        raise ValueError('frozen directions changed')
    if len(set(args.policies)) != len(args.policies):
        raise ValueError('duplicate policies')
    for index in TARGETS:
        for policy in args.policies:
            if (root / 'probes' / str(index) / policy).exists():
                raise ValueError('probe result exists; do not overwrite')
    frozen = torch.load(root / 'frozen-probe.pt', map_location='cpu', weights_only=True)
    model, config, packet = load_context(manifest)
    from worldttn.cli import timed_cuda
    for index in TARGETS:
        for policy in args.policies:
            folder = root / 'probes' / str(index) / policy
            folder.mkdir(parents=True, exist_ok=False)
            experiment.write_json(folder / 'status.json', {'status': 'running', 'pid': os.getpid()})
            try:
                probe = FrozenReadProbe((index,), frozen=frozen, policy=policy, stop_index=index)
                _, timing = timed_cuda(lambda: run_rollout(model, config, packet, manifest, probe))
                predicted = probe.outputs[index]
                baseline = frozen['outputs'][index]
                if predicted['prefix_sha256'] != baseline['prefix_sha256']:
                    raise AssertionError('target prefix does not exactly match frozen baseline')
                keys = {key for key in frozen['snapshots'] if key.startswith(f'{index+1}:')}
                if set(probe.injected_keys) != keys or len(probe.injected_keys) != len(keys):
                    raise AssertionError('missing, repeated or out-of-target injections')
                start, end = predicted['start'], predicted['end']
                target = packet['reference'][:, :, start:end]  # GT first used AFTER prediction
                value = predicted['latents']
                original = baseline['latents']
                mse, base_mse = float((value-target).square().mean()), float((original-target).square().mean())
                torch.save({'latents': value, 'start': start, 'end': end}, folder / 'latents.pt')
                experiment.write_json(folder / 'result.json', {'status': 'completed', 'policy': policy,
                    'sampler_chunk_index': index, 'event_chunk_id': index+1, 'start': start, 'end': end,
                    'nominal_endpoint_seconds': (end-1)/2, 'prefix_exact': True,
                    'archive_before_target_exact': True, 'first_intervention_state_query_exact': True,
                    'latent_mse': mse, 'baseline_latent_mse': base_mse,
                    'minus_baseline': mse-base_mse, 'relative_to_baseline_percent': 100*(mse/base_mse-1),
                    'difference_from_baseline_latent_rms': float((value-original).square().mean().sqrt()),
                    'timing_including_prefix': timing, 'injections': probe.audit,
                    'limitation': 'one chunk, fixed baseline directions and equal head budgets; not full adaptive rollout quality'})
                experiment.write_json(folder / 'status.json', {'status': 'completed', 'pid': os.getpid()})
                print(f'[Frozen probe completed] target={index} policy={policy} MSE={mse:.8f} baseline={base_mse:.8f}', flush=True)
            except BaseException as error:
                experiment.write_json(folder / 'status.json', {'status': 'failed', 'error': repr(error), 'pid': os.getpid()})
                raise
            gc.collect()
            torch.cuda.empty_cache()


def summarize(args):
    root = args.output.resolve()
    rows = []
    for index in TARGETS:
        for policy in POLICIES:
            row = json.loads((root / 'probes' / str(index) / policy / 'result.json').read_text())
            if row['status'] != 'completed' or not row['prefix_exact']:
                raise ValueError('incomplete paired probe')
            rows.append(row)
    experiment.write_json(root / 'summary.json', {'status': 'completed', 'results': rows,
        'scope': 'one fixed forest/seed, matched-norm baseline-frozen read directions, 3 target chunks',
        'limitation': 'local directional diagnostic; cannot establish long-rollout or perceptual quality'})
    print('[Frozen probe summary] ' + str(root / 'summary.json'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('capture', 'worker', 'summarize'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--policies', nargs='+', choices=POLICIES, default=list(POLICIES))
    args = parser.parse_args()
    if args.mode == 'capture' and args.reference_root is None:
        parser.error('capture requires --reference-root')
    # Fixed approved diagnostic protocol; no hidden hyperparameter search.
    for name, value in dict(anchors=[11, 15, 19], capacity=8, topk=1, temperature=.1,
            gain=.1, max_relative_delta=.05, retention='coverage', start_chunk=5, max_read_sigma=.5).items():
        setattr(args, name, value)
    existed = args.output.exists()
    try:
        {'capture': capture, 'worker': worker, 'summarize': summarize}[args.mode](args)
    except BaseException as error:
        if args.mode == 'capture' and not existed and args.output.is_dir():
            experiment.write_json(args.output / 'capture-status.json',
                                  {'status': 'failed', 'pid': os.getpid(), 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
