"""Checkpoint input tensors must die before the long training graph is built."""
import json
import weakref
from argparse import Namespace

import pytest
import torch

from test_parallel_cli import CPUFlowConfig
from test_training import TinyWorldModel, inputs
from worldttn import cli, distributed
from worldttn.checkpoint import make_optimizer, save_checkpoint
from worldttn.training import linear_flow_loss, train_clip


@pytest.mark.parametrize('resume', [False, True])
def test_cli_releases_checkpoint_weights_before_parallel_wrap(monkeypatch, tmp_path, resume):
    model = TinyWorldModel('C')
    model.base_load_report = {'sha256': None}
    checkpoint = tmp_path / 'input.pt'
    save_checkpoint(checkpoint, model, make_optimizer(model), 1)
    clean, noise, times, camera = inputs()
    bundle = tmp_path / 'batch.pt'
    torch.save({'clean_latents': clean, 'y': torch.zeros(1, 1, 2, 8),
                'camera_conditions': camera, 'width': 100, 'height': 100}, bundle)
    refs = []
    original_read, original_parallel = cli.read_checkpoint, distributed.ParallelTraining

    def tracked_read(*args, **kwargs):
        payload = original_read(*args, **kwargs)
        refs.extend(weakref.ref(tensor) for tensor in payload['adapter'].values())
        return payload

    def checked_parallel(*args, **kwargs):
        assert refs and all(ref() is None for ref in refs), 'CPU checkpoint weights retained'
        return original_parallel(*args, **kwargs)

    def update(model, config, batch, optimizer, k, parallel):
        assert all(ref() is None for ref in refs)
        return train_clip(model, clean, batch['y'], camera, optimizer, linear_flow_loss,
                          times, noise, width=100, height=100, tbptt=k, parallel=parallel)

    monkeypatch.setattr(cli, 'read_checkpoint', tracked_read)
    monkeypatch.setattr(distributed, 'ParallelTraining', checked_parallel)
    monkeypatch.setattr(cli, 'build', lambda args: (
        model, Namespace(scheduler=CPUFlowConfig()), {'learning_rate': 1e-5, 'tbptt': 2}))
    monkeypatch.setattr(cli, 'SANAFlowLoss', lambda config: linear_flow_loss)
    monkeypatch.setattr(cli, 'train_update', update)
    monkeypatch.setattr(cli, 'timed_cuda', lambda call: (call(), {
        'seconds': 0., 'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0}))
    output = tmp_path / 'train'
    args = Namespace(parallel='single', seed=3407, batch_file=str(bundle), device='cpu',
                     adapter=str(checkpoint), resume=resume, output=str(output),
                     max_steps=2 if resume else 1, save_every=1, tbptt=2)
    cli.train_command(args)
    run = json.loads((output / 'run_config.json').read_text())
    assert run['initialization']['step'] == 1
    assert run['initialization']['resume'] is resume
    records = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
    assert [row['step'] for row in records] == [2 if resume else 1]
