import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.ttn_wandb_monitor import StepLog, monitor, monitor_lock, step_metrics


def record(step=1):
    anchor = {'block': 19, 'state': {'rms': 0.5}, 'psi': {'norm': 0},
              'per_head': {'innovation_relative': [[0.6, 0.8]]},
              'state_ttt': {'effective_eta': 2., 'state_delta_relative': 0.2,
                            'per_head': {'effective_eta': [[1., 3.]]}},
              'local': {'state_ttt': {'coverage': 0.5, 'query_loss_after': 0.8,
                                    'output_effect': {'relative_delta': 0.25}}}}
    chunks = [{'chunk': i, 'anchors': [copy.deepcopy(anchor)]} for i in range(8)]
    for i, chunk in enumerate(chunks):
        chunk['anchors'][0]['state']['rms'] = i
    return {'step': step, 'loss': 0.5, 'outer_grad_norm': 3., 'seconds': 60., 'ranks': [
        {'rank': 0, 'peak_allocated_bytes': 30 * 2**30,
         'prefill': {'anchors': [{'block': 19, 'state': {'rms': 999}}]}, 'chunks': chunks,
         'memory_phases': [{'host_rss_bytes': 100 * 2**30}, {'host_rss_bytes': 110 * 2**30}],
         'prediction_diagnostics': {'per_frame_velocity_mse': [[0., 1., 2., 3.]],
                                    'estimated_x0_adjacent_rms': [[1., 2., 3.]]}}]}


def test_anchor_summaries_keep_tail_and_local_effect_separate():
    values = step_metrics(record())
    assert values['anchor19/rank0/all/state/rms/mean'] == 3.5
    assert values['anchor19/rank0/tail5_chunks/state/rms/mean'] == 5
    assert values['anchor19/rank0/all/innovation_relative/max'] == 0.8
    assert values['anchor19/rank0/all/local/state_ttt/output_effect/relative_delta/mean'] == 0.25
    assert values['anchor19/rank0/all/state_ttt/effective_eta/mean'] == 2
    assert not any('per_head' in key or 'stable_rank' in key for key in values)
    assert values['memory/rank0/peak_allocated_gib'] == 30
    assert values['memory/rank0/sampled_rss_max_gib'] == 110
    assert values['training_prediction/rank0/predicted/per_frame_velocity_mse'] == 2
    assert values['training_prediction/rank0/predicted/estimated_x0_adjacent_rms'] == 2


def test_nonfinite_values_are_flagged_and_missing_values_are_not_zero():
    sample = record()
    sample['loss'] = float('nan')
    sample['ranks'][0]['chunks'][0]['anchors'][0]['state']['rms'] = float('inf')
    metrics = step_metrics(sample)
    assert 'train/loss' not in metrics
    assert metrics['monitor/skipped_nonfinite_values'] > 0
    assert all(isinstance(value, (int, float)) for value in metrics.values())


def test_partial_jsonl_is_retried_without_losing_a_training_step(tmp_path):
    path = tmp_path / 'train.jsonl'
    reader = StepLog(path)
    assert list(reader.records()) == []
    path.write_bytes(b'{"step":1}\n{"step":')
    assert list(reader.records()) == [{'step': 1}]
    assert list(reader.records()) == []
    with path.open('ab') as handle:
        handle.write(b'2}\n')
    assert list(reader.records()) == [{'step': 2}]
    path.write_bytes(b'')
    with pytest.raises(RuntimeError, match='truncated'):
        list(reader.records())


def test_complete_invalid_jsonl_fails_visibly(tmp_path):
    path = tmp_path / 'train.jsonl'
    path.write_text('{not json}\n')
    with pytest.raises(json.JSONDecodeError):
        list(StepLog(path).records())


class FakeBackend:
    Settings = SimpleNamespace

    def __init__(self):
        self.calls, self.logged, self.finished = [], [], []
        self.next_step = 0

    def init(self, **kwargs):
        self.calls.append(kwargs)
        backend = self

        class Run:
            step = backend.next_step
            url = 'https://example.test/monitor'

            def define_metric(self, *args, **kwargs):
                pass

            def log(self, values, *, step, commit):
                assert commit
                backend.logged.append(values)
                self.step = backend.next_step = step + 1

            def finish(self, exit_code):
                backend.finished.append(exit_code)

        return Run()


def train_directory(tmp_path):
    (tmp_path / 'run_config.json').write_text(json.dumps({
        'world_size': 1, 'arguments': {'api_key': 'must-not-upload'}, 'training': {'tbptt': 0}}))
    (tmp_path / 'train.jsonl').write_text(json.dumps(record()) + '\n')
    return tmp_path


def test_online_resume_preserves_identity_without_duplicate_history_or_secrets(tmp_path, monkeypatch):
    directory = train_directory(tmp_path)
    monkeypatch.setenv('WANDB_API_KEY', 'private-test-key')
    monkeypatch.setenv('WANDB_SWEEP_ID', 'another-experiment')
    monkeypatch.setenv('WANDB_PROJECT', 'someone-elses-project')
    backend = FakeBackend()
    monitor(directory, entity='me', project='WorldTTN', backend=backend)
    with (directory / 'train.jsonl').open('a') as handle:
        handle.write(json.dumps(record(2)) + '\n')
    monitor(directory, entity='me', project='WorldTTN', backend=backend)
    assert [row['train/step'] for row in backend.logged] == [1, 2]
    assert backend.calls[0]['id'] == backend.calls[1]['id']
    assert backend.finished == [0, 0]
    assert backend.calls[0]['settings'].x_disable_stats
    assert backend.calls[0]['settings'].disable_code and backend.calls[0]['settings'].disable_git
    assert 'WANDB_SWEEP_ID' not in os.environ
    assert 'WANDB_PROJECT' not in os.environ
    assert 'private-test-key' not in json.dumps(backend.logged)
    assert 'must-not-upload' not in json.dumps(backend.calls[0]['config'])
    assert Path(directory / 'wandb-monitor/run-url.txt').is_file()
    with pytest.raises(ValueError, match='identity'):
        monitor(directory, entity='someone-else', project='WorldTTN', backend=backend)
    assert len(backend.calls) == 2


def test_follow_reads_appended_step_and_interrupts_only_monitor(tmp_path, monkeypatch):
    directory = train_directory(tmp_path)
    monkeypatch.setenv('WANDB_API_KEY', 'private-test-key')
    backend = FakeBackend()
    polls = []

    def wait(_):
        if polls:
            raise KeyboardInterrupt()
        polls.append(True)
        with (directory / 'train.jsonl').open('a') as handle:
            handle.write(json.dumps(record(2)) + '\n')

    monkeypatch.setattr('tools.ttn_wandb_monitor.time.sleep', wait)
    monitor(directory, entity='me', project='WorldTTN', backend=backend, follow=True)
    assert [row['train/step'] for row in backend.logged] == [1, 2]
    assert backend.finished == [0]


def test_explicit_key_required_and_duplicate_monitor_rejected(tmp_path, monkeypatch):
    directory = train_directory(tmp_path)
    monkeypatch.delenv('WANDB_API_KEY', raising=False)
    with pytest.raises(ValueError, match='WANDB_API_KEY'):
        monitor(directory, entity='me', project='WorldTTN', backend=FakeBackend())
    assert not (directory / 'wandb-monitor').exists()
    path = directory / 'monitor.lock'
    with monitor_lock(path):
        with pytest.raises(RuntimeError, match='Another monitor'):
            with monitor_lock(path):
                pass


def test_monitor_failure_finishes_failed_session_and_preserves_training_logs(tmp_path, monkeypatch):
    directory = train_directory(tmp_path)
    with (directory / 'train.jsonl').open('a') as handle:
        handle.write('{broken record}\n')
    original = (directory / 'train.jsonl').read_bytes()
    monkeypatch.setenv('WANDB_API_KEY', 'private-test-key')
    backend = FakeBackend()
    with pytest.raises(json.JSONDecodeError):
        monitor(directory, entity='me', project='WorldTTN', backend=backend)
    assert backend.finished == [1]
    assert (directory / 'train.jsonl').read_bytes() == original
