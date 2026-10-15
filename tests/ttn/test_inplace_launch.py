import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'tools/ttn_train_inplace_l20.sh'


def bash():
    git = shutil.which('git')
    candidate = Path(git).parent.parent / 'bin/bash.exe' if git else Path('missing')
    executable = str(candidate) if candidate.exists() else shutil.which('bash')
    if executable is None:
        pytest.skip('Bash is required for launcher checks')
    return executable


def dataset(tmp_path):
    path = tmp_path / 'dataset with spaces'
    for relative in ('data/sekai_game_train_961frames_16fps_ovl640',
                     'data/vae_cache/LTX2VAE_diffusers_704x1280/sekai_game_train_961frames_16fps_ovl640'):
        (path / relative).mkdir(parents=True)
    return path


def invoke(*args, env=None):
    if env is not None and os.name == 'nt':
        # Git Bash prepends its own tools on Windows; explicitly prepend test stubs again.
        env = dict(env, BASH_ENV=str(Path(env['PYTHON']).parent / 'bash-env'))
    return subprocess.run([bash(), str(SCRIPT), *map(str, args)], cwd=ROOT,
                          env=env, capture_output=True, text=True, encoding='utf-8', timeout=30)


def test_syntax_and_fresh_command_preserve_two_gpu_fullgrad_contract(tmp_path):
    checked = subprocess.run([bash(), '-n', str(SCRIPT)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    output = tmp_path / 'new run'
    result = invoke('--dataset-root', dataset(tmp_path), '--output', output,
                    '--gpus', '0,1', '--steps', '3', '--dry-run')
    assert result.returncode == 0, result.stderr
    for expected in ('CUDA_VISIBLE_DEVICES=0\\,1', '--nproc-per-node=2', '--parallel fsdp2',
                     '--tbptt 0', '--frames 121', '--max-steps 3', '--train-precision fp8',
                     '--train-scope dit', '--backbone-lr 1e-6', '--empty-text-probability 0.1',
                     'configs/worldttn/inplace_ttt_fp8.json', 'set_per_process_memory_fraction',
                     'MemoryMax=180G MemorySwapMax=0'):
        assert expected in result.stdout
    assert '--adapter' not in result.stdout and '--resume' not in result.stdout
    assert not output.exists()


def test_resume_reuses_same_directory_and_cumulative_target(tmp_path):
    data = dataset(tmp_path)
    output = tmp_path / 'run'
    output.mkdir()
    (output / 'last.pt').touch()
    (output / 'run_config.json').write_text('{}')
    result = invoke('--dataset-root', data, '--output', output, '--steps', '500', '--resume', '--dry-run')
    assert result.returncode == 0, result.stderr
    assert '--max-steps 500' in result.stdout and '--resume' in result.stdout
    assert '--adapter' in result.stdout and 'last.pt' in result.stdout
    fresh = invoke('--dataset-root', data, '--output', output, '--dry-run')
    assert fresh.returncode == 2 and 'new output directory' in fresh.stderr


@pytest.mark.parametrize('gpus', ['0,0', '0', '0,1,2', '0;1'])
def test_invalid_gpu_selection_is_rejected_before_launch(tmp_path, gpus):
    result = invoke('--gpus', gpus, '--output', tmp_path / 'run', '--dry-run')
    assert result.returncode == 2 and 'distinct physical indices' in result.stderr


def test_mocked_launch_propagates_training_failure_and_isolates_credential(tmp_path):
    # No GPU, process group or systemd unit is created by these executable stubs.
    commands = tmp_path / 'commands'
    commands.mkdir()
    output = tmp_path / 'run'
    stubs = {
        'uname': 'echo Linux',
        'awk': 'echo 999999999',
        'systemctl': 'exit 0',
        'systemd-run': '''while (($#)); do
 case "$1" in --user|--scope|--quiet|--unit=*) shift;; -p) shift 2;; *) break;; esac
done
exec "$@"''',
        'fake-python': '''if [[ "${1:-}" == - ]]; then cat >/dev/null; exit 0; fi
if [[ "${1:-}" == -c ]]; then exit 1; fi
printf '%s\\n' "$@" > "$FAKE_OUTPUT/argv.txt"
printf '%s\\n' "${WANDB_API_KEY-unset}" > "$FAKE_OUTPUT/credential.txt"
printf '{}\\n' > "$FAKE_OUTPUT/run_config.json"
echo '[fake train] failed deliberately'
exit 7''',
    }
    for name, body in stubs.items():
        path = commands / name
        path.write_text('#!/usr/bin/env bash\n' + body + '\n', encoding='utf-8', newline='\n')
        path.chmod(0o755)
    env = dict(os.environ, PYTHON=str(commands / 'fake-python'), FAKE_OUTPUT=str(output),
               PATH=str(commands) + os.pathsep + os.environ['PATH'], WANDB_API_KEY='must-stay-monitor-only')
    (commands / 'bash-env').write_text('export PATH="$(dirname "$PYTHON"):$PATH"\n', encoding='utf-8')
    result = invoke('--dataset-root', dataset(tmp_path), '--output', output, '--steps', '3', env=env)
    assert result.returncode == 7, result.stdout + result.stderr
    assert (output / 'train.exit-code').read_text().strip() == '7'
    assert (output / 'credential.txt').read_text().strip() == 'unset'
    assert '--nproc-per-node=2' in (output / 'argv.txt').read_text(encoding='utf-8')
    assert 'must-stay-monitor-only' not in (output / 'train.log').read_text()
    assert (output / 'systemd-unit.txt').read_text().strip().endswith('.scope')
