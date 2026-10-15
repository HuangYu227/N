"""Explicit launcher; run only after the user confirms the H100 transfer finished."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


REGISTRY = Path('/home/newuser001/huangyu/.worldttt-experiment/current.json')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def alive(pid):
    return bool(pid and Path(f'/proc/{int(pid)}').exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--source', default='/home/newuser001/huangyu/WorldTLA-state-ttt-fp8-20261014')
    parser.add_argument('--output-parent', default='/home/newuser001/huangyu/WorldTLA-memory-diagnosis-20261015')
    args = parser.parse_args()
    # First read recorded tasks; do not discover and re-label old experiments.
    previous = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else None
    if previous:
        old = [previous, *previous.get('jobs', [])]
        for item in old:
            state = Path(item['state_file']) if item.get('state_file') else None
            row = json.loads(state.read_text()) if state and state.exists() else item
            if alive(row.get('pid')) or alive(row.get('wrapper_pid')):
                raise RuntimeError('A recorded workflow process is alive; refusing duplicate launch')
    checkpoint, source = Path(args.checkpoint).resolve(), Path(args.source).resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1_000_000_000:
        raise ValueError('The complete H100 model checkpoint is not available')
    bundle = Path(__file__).resolve().parent
    expected = json.loads((bundle/'source-hashes.json').read_text())
    for name, digest in expected.items():
        if hashlib.sha256((source/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
            raise RuntimeError('Production source differs from the reviewed diagnostic source: '+name)
    python = '/home/newuser001/miniconda3/envs/wm/bin/python'
    dataset = '/home/newuser001/huangyu/WorldTTT/datasets/sana-wm-example'
    deps = '/home/newuser001/huangyu/WorldTLA-fp8-validation-20261013/dependencies-offline'
    base = '/home/newuser001/.cache/huggingface/hub/models--Efficient-Large-Model--SANA-WM_chunk_causal/snapshots/ef7ea10f5e5213e5e1af41587961cb7293369e83/dit/sana_wm_chunk_causal_1600m_720p.safetensors'
    for value in (python, dataset, deps, base):
        if not Path(value).exists():
            raise FileNotFoundError(value)
    rows = subprocess.check_output(['nvidia-smi', '-i', '6,7', '--query-gpu=index,memory.used',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    if len(rows) != 2 or any(int(row.split(',')[1]) > 1024 for row in rows):
        raise RuntimeError('GPU6 or GPU7 is occupied; refusing to share it')
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    if available < (224 << 30):
        raise RuntimeError('Two bounded inference jobs require at least224GiB available at launch')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ').lower()
    root = Path(args.output_parent).resolve()/('pilot-'+stamp)
    root.mkdir(parents=True, exist_ok=False)
    if previous:
        atomic_json(root/'previous-registry.json', previous)
    files = [bundle/name for name in ('run_diagnostic.py', 'memory_trace.py', 'memory_math.py', 'diagnostic_guard.py')]
    synced = [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]
    synced += [{'path': str(source/name), 'sha256': hashlib.sha256((source/name).read_bytes()).hexdigest()}
               for name in expected]
    if subprocess.run(['tmux', 'has-session', '-t', 'wm'], capture_output=True).returncode:
        subprocess.run(['tmux', 'new-session', '-d', '-s', 'wm', '-n', 'diagnostics-home'], check=True)
    jobs = []
    for gpu, modes in ((6, ['full', 'no-local']), (7, ['no-persistent', 'no-ttt'])):
        work = root/f'gpu{gpu}'
        work.mkdir()
        guard = work/'diagnostic_guard.py'
        guard.write_bytes((bundle/'diagnostic_guard.py').read_bytes())
        command = [python, '-u', str(bundle/'run_diagnostic.py'), '--source', str(source),
                   '--checkpoint', str(checkpoint), '--base-weights', base, '--dataset-root', dataset,
                   '--output', str(work/'results'), '--modes', *modes]
        name = f'memory-{gpu}-{stamp}'
        spec = dict(experiment_id=name, mode='launch-only', command=command, cwd=str(source),
                    output_dir=str(work), state_file=str(work/'state.json'),
                    unit='worldtla-'+name, host_memory_max_gib=64, synced_files=synced,
                    code_revision=expected, tmux_window=name,
                    env={'CUDA_VISIBLE_DEVICES': str(gpu), 'RANK': '0', 'WORLD_SIZE': '1', 'LOCAL_RANK': '0',
                         'PYTHONPATH': os.pathsep.join((str(source), deps)), 'PYTHONUNBUFFERED': '1',
                         'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'GDN_DISABLE_COMPILE': '1',
                         'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false'})
        atomic_json(work/'launch.json', spec)
        jobs.append(dict(gpu=gpu, modes=modes, state_file=str(work/'state.json'), window=name,
                         log=str(work/'train.log'), results=str(work/'results')))
    registry = dict(experiment_id='memory-pilot-'+stamp, mode='launch-only', status='prepared',
                    root=str(root), checkpoint=str(checkpoint), jobs=jobs,
                    previous_registry=str(root/'previous-registry.json') if previous else None,
                    note='Previous stale states are preserved verbatim; their final outcomes remain unknown')
    atomic_json(root/'launch.json', registry)
    atomic_json(REGISTRY, registry)
    for job in jobs:
        work = Path(job['state_file']).parent
        pane = subprocess.check_output(['tmux', 'new-window', '-d', '-P', '-F', '#{pane_id}',
            '-t', 'wm:', '-n', job['window'], 'bash'], text=True).strip()
        job['pane_id'] = pane
        subprocess.run(['tmux', 'set-option', '-p', '-t', pane, 'remain-on-exit', 'on'], check=True)
        atomic_json(REGISTRY, registry)
        command = shlex.join(['exec', python, '-u', str(work/'diagnostic_guard.py')])
        subprocess.run(['tmux', 'send-keys', '-t', pane, '-l', command], check=True)
        subprocess.run(['tmux', 'send-keys', '-t', pane, 'Enter'], check=True)
    registry['status'] = 'launched'
    atomic_json(root/'launch.json', registry)
    atomic_json(REGISTRY, registry)
    print(json.dumps(registry, indent=2))


if __name__ == '__main__':
    main()
