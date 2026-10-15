import json
import os

import pytest
import torch

from tools import ttn_checkpoint_eval as evaluation


def test_rgb_metrics_identity_and_known_constant_error():
    black, white = torch.zeros(3,3,16,16), torch.ones(3,3,16,16)
    identical = evaluation.rgb_scores(black, black)
    different = evaluation.rgb_scores(black, white)
    for same, other in zip(identical, different):
        assert same == dict(rgb_mse=0, psnr_db=120, ssim=1)
        assert other['rgb_mse'] == 1 and other['psnr_db'] == 0
        assert other['ssim'] == pytest.approx(.01**2/(1+.01**2), abs=1e-6)


def test_gaussian_ssim_detects_spatial_structure_and_rejects_bad_input():
    torch.manual_seed(3)
    actual = torch.rand(2,3,24,24)
    scores = evaluation.rgb_scores(actual, actual.roll(4, dims=-1))
    assert all(s['ssim'] < .5 and s['rgb_mse'] > 0 for s in scores)
    with pytest.raises(ValueError):
        evaluation.rgb_scores(actual, actual[:1])
    with pytest.raises(ValueError):
        evaluation.rgb_scores(actual + 1, actual)


def test_three_equal_time_windows_exclude_observation_and_include_last_sample():
    windows = evaluation.time_windows([dict(error=float(i)) for i in range(60)])
    assert windows['0-10s']['error'] == 9.5
    assert windows['10-20s']['error'] == 29.5
    assert windows['20-30s']['error'] == 49.5
    with pytest.raises(ValueError):
        evaluation.time_windows([dict(error=0)] * 61)


def test_fixed_text_camera_and_noise_are_checked():
    batch = {key: torch.tensor([float(i)]) for i,key in enumerate(evaluation.INPUTS)}
    noise = torch.ones(2)
    case = dict(case_id='fixed', input_sha256=evaluation.hashes(batch),
                noise_sha256=evaluation.tensor_sha256(noise))
    evaluation.validate_pair(batch, noise, case)
    for key in ('camera_conditions', 'mask', 'y', 'uncondition'):
        with pytest.raises(ValueError):
            evaluation.validate_pair(dict(batch, **{key: batch[key]+1}), noise, case)
    with pytest.raises(ValueError):
        evaluation.validate_pair(batch, noise+1, case)


@pytest.mark.parametrize('used,free,reason', [(16000,4095,'4096'), (22500,12000,'threshold')])
def test_budget_has_margin_below_requested_25000_and_global_pressure_exits_only_self(monkeypatch, tmp_path, used, free, reason):
    budget = evaluation.Budget(tmp_path)
    assert (budget.allocator_mib, budget.stop_mib, budget.ceiling_mib) == (20000,22500,25000)
    monkeypatch.setattr(evaluation.VideoMemoryBudget, 'sample', lambda self: used)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (free*2**20,80*2**30))
    stopped = []
    monkeypatch.setattr(evaluation.os, '_exit', stopped.append)
    budget.poll_once()
    assert stopped == [75]
    report = json.loads((tmp_path/'memory.json').read_text())
    assert report['pid'] == os.getpid() and report['status'] == 'stopped'
    assert reason in report['reason']
