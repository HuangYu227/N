import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import ttn_state_ttt_video as video


def test_nvml_usage_is_for_our_pid_only_and_missing_measurements_fail():
    assert video.process_memory_mib('12, 55000\n34, 1800\n', 34) == 1800
    assert video.process_memory_mib('12, 55000\n', 34) == 0
    with pytest.raises(ValueError, match='unavailable'):
        video.process_memory_mib('34, [Not Supported]\n', 34)


def test_budget_limits_allocator_before_model_and_leaves_training_headroom(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda d: calls.append(('device', d)))
    monkeypatch.setattr(torch.cuda, 'get_allocator_backend', lambda: 'native')
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda d: SimpleNamespace(total_memory=80 * 2**30))
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (26 * 2**30, 80 * 2**30))
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction', lambda f, d: calls.append(('limit', f, d)))
    monkeypatch.setattr(video.VideoMemoryBudget, 'sample', lambda self: 800)
    budget = video.VideoMemoryBudget(tmp_path)
    budget.prepare()
    assert calls[1] == ('limit', 18000 * 2**20 / (80 * 2**30), 0)
    assert budget.allocator_mib == 18000 and budget.ceiling_mib == 23000
    assert budget.stop_mib == 21000
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (24 * 2**30, 80 * 2**30))
    with pytest.raises(RuntimeError, match='headroom'):
        budget.prepare()


def test_budget_excess_or_monitor_failure_stops_only_this_process(monkeypatch, tmp_path):
    budget = video.VideoMemoryBudget(tmp_path)
    monkeypatch.setattr(video.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        stdout=f'12, 55000\n{os.getpid()}, 21100\n'))
    exits = []
    monkeypatch.setattr(video.os, '_exit', exits.append)
    budget.poll_once()
    assert exits == [75]
    report = json.loads((tmp_path / 'memory.json').read_text())
    assert report['status'] == 'stopped' and report['sampled_process_peak_mib'] == 21100
    assert report['pid'] == os.getpid()


def test_pid_monitor_failure_fails_closed(monkeypatch, tmp_path):
    budget = video.VideoMemoryBudget(tmp_path)

    def unavailable():
        raise RuntimeError('nvidia-smi query failed')

    monkeypatch.setattr(budget, 'sample', unavailable)
    exits = []
    monkeypatch.setattr(video.os, '_exit', exits.append)
    budget.poll_once()
    assert exits == [75]


def test_non_native_allocator_is_rejected_before_inference(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda d: None)
    monkeypatch.setattr(torch.cuda, 'get_allocator_backend', lambda: 'cudaMallocAsync')
    with pytest.raises(RuntimeError, match='native'):
        video.VideoMemoryBudget(tmp_path).prepare()


def test_milestone_is_fixed_even_when_training_has_advanced(tmp_path):
    root = tmp_path / 'train'
    (root / 'models').mkdir(parents=True)
    (root / 'run_config.json').write_text(json.dumps({'base': {'sha256': 'base'}}))
    checkpoint = root / 'models/step-00000025.pt'
    torch.save({'format': 'TTN-SANA-WM-v0.1', 'base_revision': video.BASE_REVISION,
                'step': 25, 'stage': 'C', 'base_sha256': 'base', 'config': {
                    'stage': 'C', 'state_ttt_rank': 64, 'eta_psi': 0,
                    'camera_attention': 'sana'}, 'adapter': {},
                'distributed': {'checkpoint_id': 'step25'}}, checkpoint)
    (root / 'models/step-00000025.json').write_text(json.dumps({
        'status': 'verified', 'step': 25, 'checkpoint_id': 'step25'}))
    (root / 'last.pt').write_text('newer checkpoint must not be read')
    source, payload, path, digest = video.load_milestone(root, 25)
    assert path == checkpoint.resolve() and payload['step'] == 25
    assert source['base']['sha256'] == 'base' and len(digest) == 64
    with pytest.raises(FileNotFoundError):
        video.load_milestone(root, 50)


def test_ttt_pair_reuses_same_noise_and_has_correct_labels(monkeypatch, tmp_path):
    from worldttn import cli
    observed = torch.zeros(1, 2, 1, 2, 2)
    batch = {'initial_latent': observed, 'camera_conditions': torch.zeros(1, 61, 20)}
    noise = torch.randn(1, 2, 61, 2, 2)
    methods = []

    def rollout(model, config, inputs, steps, cfg, cached_blocks, *, initial_noise, ttn_ablation, on_chunk):
        methods.append((ttn_ablation, video.tensor_sha256(initial_noise)))
        result = initial_noise.clone()
        result[:, :, :1] = observed
        if ttn_ablation == 'full': result[:, :, 1:] += 0.1
        chunk = {'chunk': 0, 'start': 0, 'end': 4, 'seconds': 0}
        on_chunk(chunk)
        return result, SimpleNamespace(commit_count=21), [chunk]

    monkeypatch.setattr(cli, 'rollout', rollout)
    monkeypatch.setattr(cli, 'timed_cuda', lambda call: (call(), {'seconds': 1}))
    monkeypatch.setattr(cli, 'to_device', lambda value, device: value)
    rows = video.generate_pair(torch.nn.Identity(), object(), batch, noise, tmp_path,
                               steps=20, cfg_scale=4.5)
    assert [name for name, _ in methods] == ['no-ttt', 'full']
    assert methods[0][1] == methods[1][1]
    assert [row['method'] for row in rows] == ['no-ttt', 'full']
    a, b = [torch.load(tmp_path / (name + '.pt'), weights_only=True)['latents']
            for name in ('no-ttt', 'full')]
    assert torch.equal(a[:, :, :1], b[:, :, :1]) and not torch.equal(a[:, :, 1:], b[:, :, 1:])


def test_actual_inference_ablation_disables_local_and_clean_state_ttt():
    from test_training import TinyWorldModel, inputs
    from worldttn.session import TTNSession
    model = TinyWorldModel('C', state_ttt_rank=4, eta_psi=0,
                           local_update=True, persistent_meta=True).eval().requires_grad_(False)
    clean, noise, t, camera = inputs()
    calls, states = [], []
    hooks = [model.blocks[i].attn.state_ttt.register_forward_hook(
        lambda *args: calls.append(True)) for i in (3, 7, 11, 15, 19)]
    try:
        with torch.no_grad():
            for ablation in ('no-ttt', 'full'):
                session = TTNSession(model, camera, 100, 100, ablation=ablation)
                session.prefill(clean[:, :, :1], torch.zeros(1, 1, 2, 8))
                context = session.begin_chunk(0, 4)
                caches = [[None] * 10 for _ in model.blocks]
                session.forward(noise[:, :, :4], t[:, :, :4], torch.zeros(1, 1, 2, 8),
                                context, caches, 0, 4)
                session.clean_forward(clean[:, :, :4], torch.zeros(1, 1, 2, 8), context, caches, 0, 4)
                states.append(session.runtime.world_state.clone())
                assert len(calls) == (0 if ablation == 'no-ttt' else 10)
    finally:
        for hook in hooks:
            hook.remove()
    assert not torch.equal(states[0], states[1])
