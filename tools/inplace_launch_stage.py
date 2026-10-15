"""Durable tmux stage wrapper; manages only the command recorded for this experiment."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def write_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('cuda-test', 'capacity', 'resume-check', 'train'), required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--window', required=True)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', DISABLE_XFORMERS='1', GDN_DISABLE_COMPILE='1',
               SANA_CP_SIZE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MALLOC_ARENA_MAX='2', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONPATH=str(root)+':/home/newuser001/huangyu/WorldTLA-fp8-validation-20261013/dependencies-offline',
               PYTHON='/home/newuser001/miniconda3/envs/wm/bin/python')
    python = env['PYTHON']
    if args.stage == 'cuda-test':
        env['INPLACE_TTT_ACCEPTANCE_OUTPUT'] = str(output / 'tiny-resume')
        command = ['systemd-run', '--user', '--scope', '--quiet',
                   '--unit=worldttn-inplace-cuda-'+str(os.getpid()), '-p', 'MemoryMax=16G', '-p', 'MemorySwapMax=0',
                   python, '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   '--nproc-per-node=2', '--max-restarts=0', '--no-python', python, '-m', 'pytest',
                   '-q', '-s', '--tb=short', 'tests/ttn/test_inplace_cuda.py']
    else:
        steps = {'capacity': 3, 'resume-check': 4, 'train': 500}[args.stage]
        command = ['bash', 'tools/ttn_train_inplace_l20.sh', '--gpus', '0,1',
                   '--steps', str(steps), '--output', str(output / 'training')]
        if args.stage != 'capacity':
            command.append('--resume')
    state_path = Path('/home/newuser001/huangyu/.worldttt-experiment/current.json')
    state_path.parent.mkdir(exist_ok=True)
    state = {'mode': 'campaign', 'experiment': 'inplace-ffn-l20-from-sana-500', 'stage': args.stage,
             'status': 'starting', 'pid': os.getpid(), 'gpu': [0, 1], 'output': str(output),
             'command': command, 'code_revision': os.environ.get('TTN_SOURCE_REVISION', 'synced-working-tree'),
             'checkout_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
             'source': str(root), 'synced_files': list(json.loads((root / '.inplace-sync.json').read_text())),
             'tmux_session': 'wm', 'window': args.window, 'pane_id': os.environ.get('TMUX_PANE'),
             'start_time': datetime.now(timezone.utc).isoformat(), 'log': str(output / (args.stage+'.log')),
             'exit_code': None}
    local_state = output / (args.stage+'-state.json')
    write_json(state_path, state); write_json(local_state, state)
    with Path(state['log']).open('a', buffering=1) as log:
        log.write('[InPlace stage] '+json.dumps(state)+'\n')
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        state.update(status='running', child_pid=process.pid)
        write_json(state_path, state); write_json(local_state, state)
        print('[InPlace stage] '+json.dumps({'stage': args.stage, 'pid': os.getpid(), 'child_pid': process.pid,
                                           'log': state['log'], 'output': str(output)}), flush=True)
        for line in process.stdout:
            log.write(line)
            print(line, end='', flush=True)
        code = process.wait()
        log.write('[InPlace exit] '+str(code)+'\n')
    state.update(status='completed' if code == 0 else 'failed', exit_code=code,
                 end_time=datetime.now(timezone.utc).isoformat())
    write_json(local_state, state); write_json(state_path, state)
    print('[InPlace completed] '+json.dumps({'stage': args.stage, 'exit_code': code}), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
