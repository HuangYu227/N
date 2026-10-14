"""H100 launch wiring and exit handling without starting GPU work."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def bash():
    git = shutil.which('git')
    bundled = Path(git).parent.parent / 'bin/bash.exe' if git else None
    executable = str(bundled) if bundled and bundled.is_file() else shutil.which('bash')
    if not executable:
        pytest.skip('bash is unavailable')
    return executable


def test_enhanced_and_legacy_configs_have_distinct_checkpoint_identities():
    from worldttn.core import TTNConfig
    enhanced = json.loads((ROOT / 'configs/worldttn/state_ttt_fp8.json').read_text())
    legacy = json.loads((ROOT / 'configs/worldttn/dual_psi_fullgrad_fp8.json').read_text())
    state = TTNConfig(**enhanced['ttn'])
    old = TTNConfig(**legacy['ttn'])
    assert (state.state_ttt_rank, state.state_ttt_policy, state.state_ttt_gain) == (64, 'curvature', .5)
    assert state.eta_psi == 0 and enhanced['tbptt'] == 0
    assert enhanced['training_latent_frames'] == 121 and enhanced['training_precision'] == 'fp8'
    assert old.state_ttt_rank == 0 and old.eta_psi == .01
    assert state.to_dict() != old.to_dict()


@pytest.mark.parametrize('exit_code', [0, 7])
def test_h100_launcher_uses_gpu3_and_records_real_training_exit(tmp_path, exit_code):
    data = tmp_path / 'data-root'
    for name in ('data/sekai_game_train_961frames_16fps_ovl640',
                 'data/vae_cache/LTX2VAE_diffusers_704x1280/sekai_game_train_961frames_16fps_ovl640'):
        (data / name).mkdir(parents=True)
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/usr/bin/env bash\n'
                    'if [[ "$1" == - ]]; then cat >/dev/null; exit 0; fi\n'
                    'printf "GPU=%s\\n" "$CUDA_VISIBLE_DEVICES"\n'
                    'printf "%s\\n" "$@"\n'
                    'exit "$TEST_TRAIN_EXIT"\n', encoding='utf-8')
    fake.chmod(0o755)
    output = tmp_path / 'new-run'
    result = subprocess.run([bash(), 'tools/ttn_train_state_ttt_h100.sh', '3',
                             data.as_posix(), output.as_posix()], cwd=ROOT,
                            env={**os.environ, 'PYTHON': fake.as_posix(), 'TEST_TRAIN_EXIT': str(exit_code)},
                            capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert result.returncode == exit_code, result.stderr
    assert (output / 'train.exit-code').read_text().strip() == str(exit_code)
    log = (output / 'train.log').read_text(encoding='utf-8')
    assert 'GPU=3' in log
    assert '--parallel\nsingle' in log and '--max-steps\n500' in log
    assert 'configs/worldttn/state_ttt_fp8.json' in log
    repeated = subprocess.run([bash(), 'tools/ttn_train_state_ttt_h100.sh', '3',
                               data.as_posix(), output.as_posix()], cwd=ROOT,
                              capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert repeated.returncode == 2 and 'new output directory' in repeated.stderr


def test_invalid_gpu_and_missing_resume_fail_before_python(tmp_path):
    data = tmp_path / 'data-root'
    data.mkdir()
    prefix = [bash(), 'tools/ttn_train_state_ttt_h100.sh']
    invalid = subprocess.run(prefix + ['3,4', data.as_posix(), (tmp_path / 'run').as_posix()],
                             cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert invalid.returncode == 2 and 'one physical GPU' in invalid.stderr
    missing = subprocess.run(prefix + ['3', data.as_posix(), (tmp_path / 'run').as_posix(), '--resume'],
                             cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert missing.returncode == 2 and 'last.pt' in missing.stderr
