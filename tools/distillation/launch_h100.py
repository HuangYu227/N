"""Run step300 DMD in the caller's tmux pane; no SSH or other jobs are touched."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from omegaconf import OmegaConf
import torch

from worldttn.checkpoint_integrity import atomic_json
from worldttn.distillation.checkpoint import digest
from worldttn.distillation.recipe import allocation_budget, code_identity, teacher_identity

ROOT = Path(__file__).resolve().parents[2]
H100_WORK = Path('/mnt/einsia/aws01-nvme/einsia-shared/homes/gaomingju/workspace/yuhuang')
OLD_RUN = H100_WORK/'WorldTTN-dual-psi-fp8/output/state-ttt-h100-gpu5-20261014T084949Z'


def find_teacher(folder):
    candidates = sorted(set(Path(folder).glob('last-resume-00000300-*/model.pt')) |
                        {Path(folder)/'step-00000300.pt', Path(folder)/'last.pt'})
    for candidate in candidates:
        if candidate.is_file():
            payload = torch.load(candidate, map_location='cpu', weights_only=False, mmap=True)
            if payload.get('step') == 300:
                return candidate
    raise ValueError('No saved step300 model found; pass --teacher /absolute/path/to/step300/model.pt')


def pin_teacher(source, destination):
    """Copy an open inode, so atomic updates of a source last.pt cannot change it."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.copying')
    if destination.exists() or temporary.exists():
        raise FileExistsError('Teacher destination already exists: '+str(destination))
    fingerprint = hashlib.sha256()
    try:
        with Path(source).open('rb') as reader, temporary.open('xb') as writer:
            before = os.fstat(reader.fileno())
            for block in iter(lambda: reader.read(8 << 20), b''):
                writer.write(block); fingerprint.update(block)
            writer.flush(); os.fsync(writer.fileno())
            after = os.fstat(reader.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Teacher was modified in place while being copied')
        payload, identity = teacher_identity(temporary, 300, fingerprint.hexdigest())
        del payload
        os.replace(temporary, destination)
        destination.chmod(0o444)
        return identity
    finally:
        if temporary.exists(): temporary.unlink()


def resources(config, ram_limit=None):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    devices = {parts[0]: parts[1:] for row in rows.splitlines()
               if (parts := [part.strip() for part in row.split(',')])}
    selected = devices.get(str(config.physical_gpus))
    if not selected or 'H100' not in selected[1]:
        raise RuntimeError('Physical GPU5 must be an H100')
    budget = allocation_budget(int(selected[2])*2**20, config.gpu_budget_gib, config.gpu_headroom_gib)
    if int(selected[3])*2**20 < budget+4*2**30:
        raise RuntimeError('GPU5 needs the allocator budget plus 4GiB free; do not overlap the original training')
    mem = {line.split(':')[0]: int(line.split()[1])*1024
           for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    available = mem['MemAvailable']/2**30
    limit = min(360, int(available-64)) if ram_limit is None else ram_limit
    if limit < 128 or available < limit+64:
        raise RuntimeError('Need at least 128GiB task RAM plus 64GiB available system headroom')
    # Fail before preparing a 10GB snapshot if the host cannot enforce the scope.
    subprocess.run(['systemctl', '--user', 'show-environment'], stdout=subprocess.DEVNULL, check=True)
    if shutil.which('systemd-run') is None:
        raise RuntimeError('systemd-run --user is required to enforce RAM/swap limits')
    return {'gpu_uuid': selected[0], 'gpu_name': selected[1], 'gpu_free_gib': int(selected[3])/1024,
            'gpu_budget_bytes': budget, 'memory_scope_gib': limit,
            'available_memory_gib_before': available, 'swap_bytes': 0}


def prepare(args):
    config = OmegaConf.load(args.config or ROOT/'configs/distillation/longlive_ttn_step300_h100_8step.yaml')
    if (config.teacher_step!=300 or config.world_size!=1 or str(config.physical_gpus)!='5'
            or config.gradient_accumulation!=2 or config.student_steps not in (4,8)):
        raise ValueError('Expected step300, GPU5, one rank, accumulation2, four/eight-step recipe')
    allocation = resources(config, args.ram_limit_gib)
    # Import dependencies before preparing files; never modify the shared environment.
    import peft, diffusers, lpips, wandb  # noqa: F401
    from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard  # noqa: F401
    from worldttn.sana import load_sana_config
    from sana.tools import resolve_hf_path
    from tools.metrics.sana_wm.eval_unified import _get_lpips_model
    _get_lpips_model('cpu')  # Resolve metric weights before spending 25 updates.
    dataset = Path(args.dataset_root or config.dataset_root or H100_WORK/'datasets/sana-wm-example').expanduser().resolve(strict=True)
    for relative in ('data/sekai_game_train_961frames_16fps_ovl640',
                     'data/vae_cache/LTX2VAE_diffusers_704x1280/sekai_game_train_961frames_16fps_ovl640'):
        if not (dataset/relative).is_dir(): raise ValueError('Missing dataset directory: '+relative)
    supplied=args.teacher or config.teacher
    source = Path(supplied).expanduser().resolve(strict=True) if supplied else find_teacher(args.teacher_run)
    sana = load_sana_config(str(ROOT/config.sana_config))
    supplied=args.base_weights or config.base_weights
    base = Path(supplied).expanduser().resolve(strict=True) if supplied else Path(resolve_hf_path(sana.model.load_from)).resolve(strict=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run = Path(args.output).expanduser().resolve() if args.output else ROOT/'output'/f'step300-h100-gpu5-{stamp}'
    if run.exists(): raise FileExistsError('Fresh run needs a new output directory; use --resume for recovery')
    if shutil.disk_usage(run.parent if run.parent.exists() else ROOT).free < 30*2**30:
        raise RuntimeError('Need at least 30GiB free for the pinned teacher and adapter recovery bundles')
    run.mkdir(parents=True)
    teacher = pin_teacher(source, run/'teacher/model.pt')
    if digest(base) != teacher['base_sha256']: raise ValueError('SANA base SHA256 differs from teacher')
    config.teacher = str(run/'teacher/model.pt')
    config.teacher_sha256 = teacher['sha256']
    config.teacher_checkpoint_id = teacher['checkpoint_id']
    config.base_weights = str(base)
    config.dataset_root = str(dataset)
    config.sana_config = str(ROOT/config.sana_config)
    config.capacity_proof = str(run/'capacity/capacity-passed.json')
    config.gpu_uuid = allocation['gpu_uuid']
    code = code_identity(ROOT)
    config.code_sha256 = code['sha256']
    path = run/'config.yaml'
    OmegaConf.save(config, path)
    atomic_json({'config': str(path), 'config_sha256': digest(path), 'teacher': teacher,
                 'teacher_source': str(source), 'code': code, 'resources': allocation,
                 'python': sys.executable, 'gpu': [5], 'world_size': 1,
                 'tmux': os.environ.get('TMUX'), 'created_at': stamp}, run/'launch.json')
    print('Prepared RUN='+str(run), flush=True)
    gc.collect()
    return run


def execute(run, resume=False, ram_limit=None):
    run = Path(run).resolve(strict=True)
    receipt = json.loads((run/'launch.json').read_text())
    config = OmegaConf.load(run/'config.yaml')
    if digest(run/'config.yaml') != receipt['config_sha256']:
        raise ValueError('Prepared config changed')
    if code_identity(ROOT)['sha256'] != config.code_sha256:
        raise ValueError('Code changed; use the recorded checkout before resuming')
    if resume:
        previous = run/'status.json'
        if previous.exists() and json.loads(previous.read_text())['status'] not in ('failed','interrupted','completed'):
            raise ValueError('Run has no terminal status; check the original tmux process before resuming')
        allocation = resources(config, ram_limit or receipt['resources']['memory_scope_gib'])
        if allocation['gpu_uuid'] != receipt['resources']['gpu_uuid']:
            raise ValueError('GPU identity changed; capacity proof belongs to another device')
    else:
        allocation = receipt['resources']
    lock = run/'runner.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    child = None
    active_unit = None
    def status(state, phase, code=None):
        atomic_json({'status':state,'phase':phase,'pid':os.getpid(),'exit_code':code,
                     'updated_at':datetime.now(timezone.utc).isoformat(),'output':str(run)},run/'status.json')
    def stop(signum, frame):
        raise KeyboardInterrupt
    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    phase = 'preflight'
    try:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='5', MASTER_ADDR='127.0.0.1', OMP_NUM_THREADS='2',
                   GDN_DISABLE_COMPILE='1', DISABLE_XFORMERS='1',
                   NCCL_SOCKET_IFNAME='lo', GLOO_SOCKET_IFNAME='lo',
                   WANDB_DIR=str(run), WANDB_CONFIG_DIR=str(run/'wandb-config'),
                   WANDB_CACHE_DIR=str(run/'wandb-cache'), PYTHONUNBUFFERED='1')
        for name in ('WANDB_CONFIG_DIR','WANDB_CACHE_DIR'): Path(env[name]).mkdir(exist_ok=True)
        proof = Path(config.capacity_proof)
        phases = ['training'] if resume and proof.is_file() else ['cuda-tests','capacity','training']
        for phase in phases:
            status('running',phase)
            folder = run/phase
            folder.mkdir(exist_ok=True)
            active_unit='ttn-dmd-'+hashlib.sha256(str(run).encode()).hexdigest()[:12]+f'-{phase}-{os.getpid()}.scope'
            command = ['systemd-run','--user','--scope','--quiet',
                       '--unit='+active_unit,
                       '-p',f'MemoryMax={allocation["memory_scope_gib"]}G','-p','MemorySwapMax=0',
                       sys.executable,'-u','-m','torch.distributed.run',
                       '--nnodes=1','--nproc_per_node=1','--rdzv_backend=c10d',
                       '--rdzv_endpoint=127.0.0.1:0','--rdzv_id='+run.name+'-'+phase]
            if phase=='cuda-tests':
                command += [
                    '-m','pytest','-q','tests/ttn/test_distillation_fsdp.py',
                    'tests/ttn/test_distillation_gdn_lora.py',
                    '--basetemp',str(folder/'pytest')]
            else:
                command += ['-m','longlive2.train','--config_path',str(run/'config.yaml'),
                            '--logdir',str(folder)]
            if phase == 'capacity': command.append('--capacity-only')
            atomic_json({'command':command,'phase':phase}, folder/'command.json')
            print(f'[TTN DMD] {phase}: log={run/(phase+".log")}', flush=True)
            with (run/(phase+'.log')).open('a', buffering=1) as stream:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                result = child.wait(); child = None
            (run/(phase+'.exit-code')).write_text(str(result)+'\n')
            if result:
                status('failed',phase,result)
                print(f'[TTN DMD] {phase} failed, exit={result}; formal training will not advance', flush=True)
                return result
            if phase=='capacity':
                recorded=json.loads((folder/'run-config.json').read_text())['identity']
                if not proof.is_file() or json.loads(proof.read_text())!=recorded:
                    raise RuntimeError('Capacity exited without a matching 121-latent acceptance receipt')
        status('completed','training',0)
        return 0
    except KeyboardInterrupt:
        if child is not None:
            subprocess.run(['systemctl','--user','stop',active_unit],check=False,timeout=20,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            if child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL); child.wait()
        status('interrupted',phase,130)
        return 130
    except BaseException:
        status('failed',phase,1)
        raise
    finally:
        for sig, handler in old_handlers.items(): signal.signal(sig,handler)
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,help='Prepared gated stage4 recipe; defaults to step300 eight-step')
    parser.add_argument('--teacher', type=Path, help='Exact step300 checkpoint; copied before training')
    parser.add_argument('--teacher-run', type=Path, default=OLD_RUN, help='Search this original run for step300')
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--base-weights', type=Path, help='Existing SANA safetensors; defaults to its HF cache')
    parser.add_argument('--output', type=Path, help='New run directory')
    parser.add_argument('--resume', type=Path, help='Resume an existing run with the recorded config and teacher')
    parser.add_argument('--ram-limit-gib', type=int, help='Task RAM scope; defaults to min(360, available-64)')
    args = parser.parse_args()
    if args.resume and (args.teacher or args.output or args.base_weights or args.config or args.dataset_root):
        parser.error('--resume uses its recorded teacher/config; do not supply fresh-run overrides')
    os.chdir(ROOT)
    run = args.resume if args.resume else prepare(args)
    raise SystemExit(execute(run,resume=bool(args.resume),ram_limit=args.ram_limit_gib))


if __name__ == '__main__': main()
