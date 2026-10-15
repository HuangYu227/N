#!/usr/bin/env bash
set -euo pipefail
REPO=/home/newuser001/huangyu/WorldTLA-longlive-distill
CONFIG="$1"
RUN="$2"
WM_PYTHON=/home/newuser001/miniconda3/envs/wm/bin/python
cd "$REPO"
mkdir -p "$RUN"
write_status() {
  "$WM_PYTHON" - "$RUN" "$1" "$2" <<'PY'
import json,os,sys,time
from pathlib import Path
run=Path(sys.argv[1]);path=run/'status.json'
state={'status':sys.argv[2],'exit_code':int(sys.argv[3]),'pid':os.getppid(),
       'updated_at':time.time(),'output':str(run)}
tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));os.replace(tmp,path)
current=Path('/home/newuser001/huangyu/.worldttt-experiment/current.json')
if current.is_file():
    record=json.loads(current.read_text())
    if record.get('output')==str(run):
        record.update(status=state['status'],exit_code=state['exit_code'])
        tmp=current.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));os.replace(tmp,current)
PY
}
write_status capacity -1
export CUDA_VISIBLE_DEVICES=6,7
export MASTER_ADDR=127.0.0.1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export OMP_NUM_THREADS=1
export GDN_DISABLE_COMPILE=1
export DISABLE_XFORMERS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$REPO:/home/newuser001/huangyu/WorldTLA-fp8-validation-20261013/dependencies-offline${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DIR="$RUN"
export WANDB_CONFIG_DIR="$RUN/wandb-config"
export WANDB_CACHE_DIR="$RUN/wandb-cache"
mkdir -p "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR"
PORT="$($WM_PYTHON - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0))
    print(sock.getsockname()[1])
PY
)"
set +e
systemd-run --user --scope -p MemoryMax=360G -p MemorySwapMax=0 \
  "$WM_PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port="$PORT" -m longlive2.train \
  --config_path "$CONFIG" --logdir "$RUN" --capacity-only \
  > "$RUN/capacity.log" 2>&1
CAPACITY_EXIT=$?
set -e
printf '%s\n' "$CAPACITY_EXIT" > "$RUN/capacity.exit"
if (( CAPACITY_EXIT != 0 )); then
  write_status failed "$CAPACITY_EXIT"
  printf '[TTN DMD] capacity failed, exit=%d\n' "$CAPACITY_EXIT"
  exit "$CAPACITY_EXIT"
fi
write_status training -1
PORT="$($WM_PYTHON - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0))
    print(sock.getsockname()[1])
PY
)"
set +e
systemd-run --user --scope -p MemoryMax=360G -p MemorySwapMax=0 \
  "$WM_PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port="$PORT" -m longlive2.train \
  --config_path "$CONFIG" --logdir "$RUN" \
  > "$RUN/train.log" 2>&1
TRAIN_EXIT=$?
set -e
printf '%s\n' "$TRAIN_EXIT" > "$RUN/train.exit"
if (( TRAIN_EXIT == 0 )); then write_status completed 0; else write_status failed "$TRAIN_EXIT"; fi
printf '[TTN DMD] formal training exit=%d\n' "$TRAIN_EXIT"
exit "$TRAIN_EXIT"
