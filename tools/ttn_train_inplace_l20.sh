#!/usr/bin/env bash
# One bounded, two-L20 job. Run inside an existing tmux pane.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
gpus=0,1
steps=500
output=''
resume=0
dry_run=0
dataset_root="${DATASET_ROOT:-/home/newuser001/huangyu/WorldTTT/datasets/sana-wm-example}"
while (($#)); do
  case "$1" in
    --gpus|--steps|--output|--dataset-root)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --gpus) gpus="$2";; --steps) steps="$2";;
        --output) output="$2";; --dataset-root) dataset_root="$2";;
      esac
      shift 2;;
    --resume) resume=1; shift;;
    --dry-run) dry_run=1; shift;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
[[ "$gpus" =~ ^([0-9]+),([0-9]+)$ && "${BASH_REMATCH[1]}" != "${BASH_REMATCH[2]}" ]] || {
  echo '--gpus requires two distinct physical indices, e.g. 0,1.' >&2; exit 2;
}
[[ "$steps" =~ ^[1-9][0-9]*$ && "$steps" -le 500 && -n "$output" ]] || {
  echo 'Usage: bash tools/ttn_train_inplace_l20.sh --output DIR [--gpus 0,1] [--steps 3|4|500] [--resume]' >&2; exit 2;
}
[[ -z "${SLURM_PROCID:-}" ]] || { echo 'Use the Slurm launcher inside a Slurm allocation.' >&2; exit 2; }
dataset_root="$(realpath -e "$dataset_root")"
output="$(realpath -m "$output")"
[[ -d "$dataset_root/data/sekai_game_train_961frames_16fps_ovl640" &&
   -d "$dataset_root/data/vae_cache/LTX2VAE_diffusers_704x1280/sekai_game_train_961frames_16fps_ovl640" ]] || {
  echo 'Dataset root must contain raw ZIPs and the existing LTX-2 VAE cache.' >&2; exit 2;
}
args=(train --config configs/worldttn/inplace_ttt_fp8.json --dataset-root "$dataset_root"
      --output "$output" --parallel fsdp2 --train-scope dit --optimizer-policy origin
      --backbone-lr 1e-6 --train-precision fp8 --activation-offload cpu-pageable
      --ttn-core-backend reference --ttn-psi-backend reference --cross-attn-backend math
      --text-encoder-device cpu --tbptt 0 --frames 121 --max-steps "$steps"
      --save-every 25 --seed 3407 --empty-text-probability 0.1 --memory-trace)
if ((resume)); then
  [[ -f "$output/last.pt" && -f "$output/run_config.json" ]] || {
    echo 'Resume requires this run directory with last.pt and run_config.json.' >&2; exit 2;
  }
  args+=(--adapter "$output/last.pt" --resume)
else
  [[ ! -e "$output" ]] || { echo 'Fresh training requires a new output directory.' >&2; exit 2; }
fi
[[ -z "${BASE_WEIGHTS:-}" ]] || args+=(--base-weights "$BASE_WEIGHTS")
python_bin="${PYTHON:-python}"
# The allocator limit applies in each torchrun worker, not only its parent.
worker='import torch; torch.cuda.set_device(int(__import__("os").environ["LOCAL_RANK"])); torch.cuda.set_per_process_memory_fraction(43*2**30/torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory); from worldttn.cli import main; main()'
command=("$python_bin" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2
         --max-restarts=0 --no-python "$python_bin" -u -c "$worker" "${args[@]}")
if ((dry_run)); then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpus"
  printf '%q ' "${command[@]}"
  printf '\nRAM scope: MemoryMax=180G MemorySwapMax=0; preflight=212GiB; stop below 32GiB host MemAvailable.\n'
  exit 0
fi
[[ "$(uname -s)" == Linux ]] || { echo 'This launcher requires Linux and a user systemd manager.' >&2; exit 2; }
systemctl --user show-environment >/dev/null
export CUDA_VISIBLE_DEVICES="$gpus" DISABLE_XFORMERS=1 SANA_CP_SIZE=1 GDN_DISABLE_COMPILE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TTN_ACTIVATION_GPU_CACHE_GIB=0 TTN_ACTIVATION_GPU_ALLOCATION_LIMIT_GIB=40
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset RANK LOCAL_RANK WORLD_SIZE NODE_RANK LOCAL_WORLD_SIZE
# Do not give training or another account a W&B credential inherited by this shell.
monitor_api_key="${WANDB_API_KEY:-}"
unset WANDB_API_KEY
"$python_bin" - <<'PY'
from pathlib import Path
import torch
from worldttn.fp8 import _load_backend
mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
available = int(mem['MemAvailable'].split()[0]) * 1024
if available < 212 * 2**30:
    raise RuntimeError(f'Host MemAvailable {available/2**30:.2f} GiB is below 212 GiB; not started')
if torch.cuda.device_count() != 2:
    raise RuntimeError('Exactly two selected GPUs must be visible')
for index in range(2):
    name = torch.cuda.get_device_name(index)
    free, total = torch.cuda.mem_get_info(index)
    if 'L20' not in name or free < 43 * 2**30:
        raise RuntimeError(f'GPU {index}: {name}, free {free/2**30:.2f} GiB; requires a free L20 with >=43 GiB')
    print(f'[TTN preflight] logical GPU {index}: {name}, free={free/2**30:.2f} GiB', flush=True)
print(f'[TTN preflight] MemAvailable={available/2**30:.2f} GiB; TorchAO={_load_backend().version}', flush=True)
PY
mkdir -p "$output"
rm -f -- "$output/train.exit-code"
unit="worldttn-inplace-${UID}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
printf '%s.scope\n' "$unit" > "$output/systemd-unit.txt"
printf '[TTN launch] physical GPUs=%s; cumulative target=%s; resume=%s; output=%s\n' "$gpus" "$steps" "$resume" "$output"
train_pid=''; guard_pid=''; monitor_pid=''
cleanup() {
  [[ -z "$guard_pid" ]] || kill "$guard_pid" 2>/dev/null || true
  if [[ -n "$train_pid" ]] && kill -0 "$train_pid" 2>/dev/null; then
    systemctl --user stop "$unit.scope" >/dev/null 2>&1 || true
    wait "$train_pid" 2>/dev/null || true
  fi
  [[ -f "$output/train.exit-code" ]] || printf '130\n' > "$output/train.exit-code"
  if [[ -n "$monitor_pid" ]]; then
    for _ in {1..20}; do kill -0 "$monitor_pid" 2>/dev/null || break; sleep 1; done
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
(
  set +e
  systemd-run --user --scope --quiet --unit="$unit" -p MemoryMax=180G -p MemorySwapMax=0 \
    "${command[@]}" 2>&1 | tee -a "$output/train.log"
  codes=("${PIPESTATUS[@]}")
  code="${codes[0]}"
  [[ "$code" != 0 ]] || code="${codes[1]}"
  printf '%s\n' "$code" > "$output/train.exit-code"
  exit "$code"
) &
train_pid=$!
(
  while kill -0 "$train_pid" 2>/dev/null; do
    available_kib="$(awk '$1=="MemAvailable:" {print $2}' /proc/meminfo)"
    if [[ "$available_kib" =~ ^[0-9]+$ && "$available_kib" -lt 33554432 ]]; then
      printf '[TTN RAM guard] Host MemAvailable=%s KiB; stopping only %s.scope\n' "$available_kib" "$unit" | tee -a "$output/ram-guard.log" >&2
      systemctl --user stop "$unit.scope"
      break
    fi
    sleep 5
  done
) &
guard_pid=$!
(
  while [[ ! -f "$output/run_config.json" ]] && kill -0 "$train_pid" 2>/dev/null; do sleep 2; done
  [[ -f "$output/run_config.json" ]] || exit 0
  monitor_python="${WANDB_PYTHON:-$python_bin}"
  if ! "$monitor_python" -c 'import wandb' >/dev/null 2>&1; then
    printf 'W&B SDK unavailable in monitor Python; train.jsonl remains the metrics source.\n' >> "$output/wandb-status.txt"
    exit 0
  fi
  exec env WANDB_API_KEY="$monitor_api_key" "$monitor_python" -u tools/ttn_wandb_monitor.py "$output" \
    --entity 3193513478- --project WorldTTN --mode auto --follow --poll 5 --stop-file "$output/train.exit-code"
) > "$output/wandb-monitor.log" 2>&1 &
monitor_pid=$!
set +e
wait "$train_pid"
code=$?
set -e
printf '[TTN launch] exit=%s; see %s/train.log\n' "$code" "$output"
exit "$code"
