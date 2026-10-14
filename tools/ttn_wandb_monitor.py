"""Read existing TTN step logs; never import torch or touch the training process."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import time
import uuid


def numeric_leaves(value, prefix=''):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value
    elif isinstance(value, list):
        for item in value:
            yield from numeric_leaves(item, prefix)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in ('per_head', 'measurement', 'top4'):
                yield from numeric_leaves(item, f'{prefix}/{key}' if prefix else key)


def step_metrics(record):
    """Chunk/head summaries are descriptive measurements, not independent samples."""
    metrics = {'train/step': int(record['step'])}
    skipped = 0

    def scalar(name, value):
        nonlocal skipped
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(value):
                metrics[name] = value
            else:
                skipped += 1

    for key in ('loss', 'outer_grad_norm', 'seconds', 'tbptt', 'global_batch', 'world_size'):
        scalar('train/' + key, record.get(key))
    for rank in record.get('ranks', []):
        prefix = f'rank{rank["rank"]}'
        for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
            if key in rank:
                scalar(f'memory/{prefix}/{key.removesuffix("_bytes")}_gib', rank[key] / 2**30)
        rss = [r['host_rss_bytes'] for r in rank.get('memory_phases', []) if 'host_rss_bytes' in r]
        if rss:
            scalar(f'memory/{prefix}/sampled_rss_max_gib', max(rss) / 2**30)
        for section in ('sequence', 'exposure', 'activation_storage', 'storage'):
            for key, value in numeric_leaves(rank.get(section, {})):
                scalar(f'{section}/{prefix}/{key}', value)
        for section in ('anchor_gradients', 'parameter_update'):
            for key, value in numeric_leaves(rank.get(section, {})):
                scalar(f'{section}/{prefix}/{key}', value)
        if 'parameter_update' in rank:
            scalar(f'health/{prefix}/missing_core_gradients',
                   len(rank['parameter_update'].get('missing_core_gradients', [])))
        for name, frames in rank.get('prediction_diagnostics', {}).items():
            if not isinstance(frames, list):
                continue
            # Frame zero is observed and has zero velocity error by construction.
            predicted = frames if name == 'estimated_x0_adjacent_rms' else [branch[1:] for branch in frames]
            values = [v for _, v in numeric_leaves(predicted) if math.isfinite(v)]
            tail = [v for _, v in numeric_leaves([branch[-15:] for branch in frames]) if math.isfinite(v)]
            for segment, selected in (('predicted', values), ('tail15_latents', tail)):
                if selected:
                    scalar(f'training_prediction/{prefix}/{segment}/{name}', sum(selected) / len(selected))
        chunks = rank.get('chunks', [])
        for segment, selected in (('all', chunks), ('tail5_chunks', chunks[-5:])):
            samples = defaultdict(list)
            for chunk in selected:
                for anchor in chunk.get('anchors', []):
                    block = anchor['block']
                    fields = {key: anchor[key] for key in (
                        'state', 'state_pred', 'correction', 'innovation_loss', 'inner_clipped_fraction',
                        'psi', 'psi_update', 'beta', 'visual_read', 'state_spectrum', 'state_dynamics',
                        'camera_cached_tokens', 'state_ttt') if key in anchor}
                    fields['innovation_relative'] = anchor.get('per_head', {}).get('innovation_relative')
                    fields['local'] = {key: anchor.get('local', {})[key] for key in (
                        'state_ttt', 'query', 'support', 'clip_fraction', 'output_effect')
                        if key in anchor.get('local', {})}
                    for name, value in numeric_leaves(fields):
                        if math.isfinite(value):
                            samples[f'anchor{block}/{prefix}/{segment}/{name}'].append(value)
                        else:
                            skipped += 1
            for name, values in samples.items():
                scalar(name + '/mean', sum(values) / len(values))
                scalar(name + '/max', max(values))
    metrics['monitor/skipped_nonfinite_values'] = skipped
    return metrics


class StepLog:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0

    def records(self):
        if not self.path.exists():
            return
        if self.path.stat().st_size < self.offset:
            raise RuntimeError('train.jsonl was truncated; use a separate W&B run for rewritten history')
        with self.path.open('rb') as handle:
            handle.seek(self.offset)
            while line := handle.readline():
                if not line.endswith(b'\n'):
                    break  # The training writer has not finished this record yet.
                record = json.loads(line)
                self.offset = handle.tell()
                yield record


@contextmanager
def monitor_lock(path):
    with path.open('a+b') as handle:
        handle.write(b'0'); handle.flush(); handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Another monitor already owns this training directory') from error
        yield  # Closing the descriptor releases the lock even after a crash.


def monitor(directory, *, entity, project, mode='online', follow=False, poll=15, backend=None):
    directory = Path(directory).resolve(strict=True)
    if not directory.is_dir() or not (directory / 'run_config.json').is_file():
        raise ValueError('Select the running training directory containing run_config.json')
    if mode == 'online' and not os.environ.get('WANDB_API_KEY'):
        raise ValueError('Set WANDB_API_KEY only in this monitor shell; do not run shared wandb login')
    if poll <= 0:
        raise ValueError('poll must be positive')
    work = directory / 'wandb-monitor'
    work.mkdir(exist_ok=True)
    with monitor_lock(work / 'monitor.lock'):
        identity_path = work / 'identity.json'
        saved_identity = json.loads(identity_path.read_text()) if identity_path.exists() else None
        identity = saved_identity or {
            'entity': entity, 'project': project, 'mode': mode, 'id': uuid.uuid4().hex[:8]}
        if any(identity[key] != value for key, value in (('entity', entity), ('project', project), ('mode', mode))):
            initialized = identity.get('initialized') or (work / 'run-url.txt').exists()
            offline_run = any((work / 'sdk' / 'wandb').glob(f'offline-run-*-{identity["id"]}'))
            if initialized or offline_run:
                raise ValueError('Monitor entity/project/mode differs from its saved W&B identity')
            # Older launchers saved identity before init, including failed entity lookups.
            identity = {'entity': entity, 'project': project, 'mode': mode, 'id': uuid.uuid4().hex[:8]}
        # A shared shell may carry someone else's sweep/service/project settings.
        for key in tuple(os.environ):
            if key.startswith('WANDB_') and key != 'WANDB_API_KEY':
                del os.environ[key]
        # Only this child process gets isolated SDK settings/cache directories.
        for key, folder in (('WANDB_CONFIG_DIR', 'config'), ('WANDB_CACHE_DIR', 'cache'),
                            ('WANDB_DATA_DIR', 'data'), ('WANDB_DIR', 'sdk')):
            location = work / folder
            location.mkdir(exist_ok=True)
            os.environ[key] = str(location)
        if backend is None:
            import wandb as backend
        source = json.loads((directory / 'run_config.json').read_text(encoding='utf-8'))
        config = {key: source[key] for key in ('world_size', 'global_batch', 'parallel', 'training_precision')
                  if key in source}
        config['metric_scope'] = 'saved GT-clean training diagnostics; no rollout/video quality claim'
        config['aggregation'] = 'per rank; anchor mean/max across chunks/heads; observed prefill excluded'
        training = source.get('training', {})
        config['training'] = {key: training[key] for key in ('tbptt', 'seed', 'per_rank_batch', 'meta_ttt',
                                                           'compute_precision') if key in training}
        run = backend.init(entity=entity, project=project, id=identity['id'], resume='allow',
                           name=directory.name, job_type='metrics-monitor', dir=str(work / 'sdk'),
                           mode=mode, config=config, settings=backend.Settings(
                               base_url='https://api.wandb.ai', disable_git=True, disable_code=True,
                               console='off', x_disable_stats=True))
        exit_code = 0
        try:
            if saved_identity is not None and saved_identity['id'] != identity['id']:
                (work / f'identity-uninitialized-{saved_identity["id"]}.json').write_text(
                    json.dumps(saved_identity, indent=2), encoding='utf-8')
            identity['initialized'] = True
            temporary = work / 'identity.json.tmp'
            temporary.write_text(json.dumps(identity, indent=2), encoding='utf-8')
            temporary.replace(identity_path)
            run.define_metric('train/step')
            run.define_metric('*', step_metric='train/step')
            url = run.url if mode == 'online' else None
            if url:
                (work / 'run-url.txt').write_text(url + '\n', encoding='utf-8')
                print(f'[TTN W&B] {url}', flush=True)
            reader = StepLog(directory / 'train.jsonl')
            while True:
                for record in reader.records():
                    if int(record['step']) < run.step:
                        continue  # W&B restored its next history step, not a local enqueue cursor.
                    run.log(step_metrics(record), step=int(record['step']), commit=True)
                    print(f'[TTN W&B] recorded training step {record["step"]}', flush=True)
                if not follow:
                    break
                time.sleep(poll)
        except KeyboardInterrupt:
            print('[TTN W&B] monitor stopped; training process is independent', flush=True)
        except BaseException:
            exit_code = 1
            raise
        finally:
            run.finish(exit_code=exit_code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('training_directory', type=Path)
    parser.add_argument('--entity', required=True)
    parser.add_argument('--project', default='WorldTTN')
    parser.add_argument('--mode', choices=('online', 'offline'), default='online')
    parser.add_argument('--follow', action='store_true', help='Wait for new completed steps until Ctrl+C')
    parser.add_argument('--poll', type=float, default=15)
    args = parser.parse_args()
    monitor(args.training_directory, entity=args.entity, project=args.project,
            mode=args.mode, follow=args.follow, poll=args.poll)


if __name__ == '__main__':
    main()
