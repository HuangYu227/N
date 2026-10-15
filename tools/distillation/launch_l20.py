"""Reserve only L20 physical GPU 6/7 for a guarded capacity -> 500 update run."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import shlex
import subprocess

ROOT=Path('/home/newuser001/huangyu/WorldTLA-longlive-distill')
STATE=Path('/home/newuser001/huangyu/.worldttt-experiment/current.json')
CONFIG=ROOT/'configs/distillation/longlive_ttn_step200_8step.yaml'


def output(command):
    return subprocess.check_output(command,text=True).strip()


def atomic(path,value):
    path.parent.mkdir(exist_ok=True,parents=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    os.replace(temporary,path)


def check(config):
    if not config.is_file() or not (ROOT/'tools/distillation/run_l20.sh').is_file():
        raise RuntimeError('Distillation code/config is incomplete')
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if ':' in line:
            name,value=line.split(':',1)
            mem[name]=int(value.strip().split()[0])*1024
    if mem['MemAvailable'] < (360+64)*(1<<30):
        raise RuntimeError('Need 360GiB memory scope plus 64GiB system headroom')
    rows=output(['nvidia-smi','--query-gpu=index,uuid,memory.free,compute_mode',
                 '--format=csv,noheader,nounits']).splitlines()
    devices={int(v[0]):v[1:] for line in rows if (v:=[s.strip() for s in line.split(',')])}
    for number in (6,7):
        if number not in devices or int(devices[number][1])<43*1024:
            raise RuntimeError(f'Physical GPU {number} has insufficient unallocated memory')
    if STATE.exists():
        old=json.loads(STATE.read_text())
        if old.get('status') in ('capacity','training'):
            raise RuntimeError('Recorded experiment is still active')
        for job in old.get('jobs',[]):
            file=Path(job.get('state_file',''))
            if not file.is_file() or json.loads(file.read_text()).get('status') not in ('completed','failed'):
                raise RuntimeError('Previous experiment has no terminal receipt')
    if 'wm' not in output(['tmux','list-sessions','-F','#{session_name}']).splitlines():
        raise RuntimeError('Expected existing wm tmux session')
    return mem,devices


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=CONFIG)
    args=parser.parse_args()
    config=args.config.expanduser().resolve()
    from omegaconf import OmegaConf
    settings=OmegaConf.load(config)
    if settings.backend!='ttn_state_ttt' or settings.student_steps not in (4,8):
        raise ValueError('Expected the pinned TTN four- or eight-step recipe')
    mem,devices=check(config)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=ROOT/'output'/f'longlive-state-ttt-step200-{settings.student_steps}step-{stamp}'
    run.mkdir(parents=True,exist_ok=False)
    if STATE.exists(): atomic(run/'previous-current.json',json.loads(STATE.read_text()))
    name='ttn-dmd-'+stamp[4:]
    if name in output(['tmux','list-windows','-t','wm','-F','#{window_name}']).splitlines():
        raise RuntimeError('Tmux window already exists')
    command='bash '+ ' '.join(shlex.quote(str(item)) for item in
                            (ROOT/'tools/distillation/run_l20.sh',config,run))
    pane=output(['tmux','new-window','-d','-t','wm','-n',name,'-P','-F','#{pane_id}',command])
    pid=int(output(['tmux','display-message','-p','-t',pane,'#{pane_pid}']))
    tracked=[]
    for folder in ('longlive2','worldttn/distillation','tools/distillation','configs/distillation'):
        tracked.extend((ROOT/folder).rglob('*.py'))
        tracked.extend((ROOT/folder).rglob('*.yaml'))
        tracked.extend((ROOT/folder).rglob('*.sh'))
    tracked.append(ROOT/'tools/download.py')
    tracked.extend(ROOT/path for path in ('diffusion/model/nets/sana_gdn_camctrl_blocks.py',
                                          'diffusion/model/ops/fused_streaming.py'))
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in tracked if '__pycache__' not in p.parts}
    record={'experiment_id':run.name,'mode':'campaign','status':'capacity',
            'repo':str(ROOT),'output':str(run),'config':str(config),
            'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
            'command':command,'gpu':[6,7],'gpu_uuid':[devices[k][0] for k in (6,7)],
            'memory_scope_gib':360,'swap_bytes':0,
            'available_memory_gib_before':mem['MemAvailable']/(1<<30),
            'tmux_session':'wm','tmux_window':name,'tmux_pane_id':pane,'pid':pid,
            'log':str(run/'capacity.log'),'code_revision':'content-sha256:'+
                hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
            'synced_files':files,'start_time':stamp}
    atomic(STATE,record)
    atomic(run/'launch.json',record)
    print(json.dumps({'status':'EXPERIMENT_STARTED','output':str(run),'pane':pane,
                      'pid':pid,'memory_gib':record['available_memory_gib_before']},ensure_ascii=False))


if __name__=='__main__': main()
