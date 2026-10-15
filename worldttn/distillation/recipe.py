"""Pinned teacher identity and hardware budgets shared by both launch recipes."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch

from worldttn.core import TTNConfig
from .checkpoint import digest


def teacher_identity(path, step, sha256=None, checkpoint_id=None):
    path = Path(path).expanduser().resolve(strict=True)
    fingerprint = digest(path)
    if sha256 is not None and fingerprint != sha256:
        raise ValueError('Pinned teacher SHA256 mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    distributed = payload.get('distributed', {})
    if (payload.get('format') != 'TTN-SANA-WM-v0.1' or payload.get('step') != step
            or payload.get('train_scope') != 'dit' or payload.get('weight_scope') != 'dit'
            or not distributed.get('checkpoint_id')):
        raise ValueError(f'Expected complete step{step} DiT teacher with checkpoint ID')
    if checkpoint_id is not None and distributed['checkpoint_id'] != checkpoint_id:
        raise ValueError('Pinned teacher checkpoint ID mismatch')
    settings = TTNConfig(**payload['config'])
    if (settings.state_ttt_rank != 64 or settings.eta_psi != 0
            or settings.state_ttt_policy != 'curvature' or not settings.local_update
            or not settings.persistent_meta):
        raise ValueError('Teacher is not the enhanced rank64 StateTTT method')
    return payload, {'step': step, 'checkpoint_id': distributed['checkpoint_id'],
                     'sha256': fingerprint, 'base_sha256': payload['base_sha256']}


def allocation_budget(total_bytes, requested_gib, headroom_gib):
    if not all(math.isfinite(v) and v > 0 for v in (requested_gib, headroom_gib)):
        raise ValueError('GPU budget and headroom must be positive finite GiB')
    result = min(requested_gib * 2**30, total_bytes - headroom_gib * 2**30)
    if result <= 0:
        raise ValueError('Device has no memory after reserving headroom')
    return int(result)


def code_identity(root):
    root = Path(root)
    files = []
    for folder in ('worldttn', 'diffusion/model', 'longlive2', 'tools/distillation', 'configs/distillation'):
        files.extend(p for p in (root/folder).rglob('*')
                     if p.suffix in ('.py', '.yaml', '.sh') and '__pycache__' not in p.parts)
    files.append(root/'tools/download.py')
    hashes = {p.relative_to(root).as_posix(): digest(p) for p in sorted(files)}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'sha256': fingerprint, 'commit': commit, 'files': hashes}


def memory_report(device):
    values = {}
    status = Path('/proc/self/status')
    if status.exists():
        for name, label in (('VmRSS', 'host_rss_gib'), ('VmHWM', 'host_peak_rss_gib')):
            line = next(row for row in status.read_text().splitlines() if row.startswith(name+':'))
            values[label] = int(line.split()[1])/2**20
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith(('MemAvailable:', 'SwapFree:')):
                values['host_'+line.split(':')[0].lower()+'_gib'] = int(line.split()[1])/2**20
    if torch.device(device).type == 'cuda':
        free, total = torch.cuda.mem_get_info(device)
        values.update(gpu_allocated_gib=torch.cuda.memory_allocated(device)/2**30,
                      gpu_reserved_gib=torch.cuda.memory_reserved(device)/2**30,
                      gpu_peak_gib=torch.cuda.max_memory_allocated(device)/2**30,
                      gpu_peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                      gpu_device_free_gib=free/2**30, gpu_device_used_gib=(total-free)/2**30)
    return values
