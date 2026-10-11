#!/usr/bin/env bash
# Run in the user's tmux session; no daemon, remote login or implicit environment switch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CUDA_VISIBLE_DEVICES:?Set the allocated physical GPU indices explicitly}"
: "${DATASET_ROOT:?Set the directory containing the SANA data/ and VAE cache tree}"
[[ -d "$DATASET_ROOT" ]] || { echo "Missing dataset root: $DATASET_ROOT" >&2; exit 2; }
for path in "${DATA_DIR:-}" "${VAE_CACHE_DIR:-}"; do
  [[ -z "$path" || -d "$path" ]] || { echo "Missing data/cache directory: $path" >&2; exit 2; }
done
case "${BASE_WEIGHTS:-}" in
  ''|hf://*) ;;
  *) [[ -f "$BASE_WEIGHTS" ]] || { echo "Missing base weights: $BASE_WEIGHTS" >&2; exit 2; } ;;
esac
ACTIVATION_OFFLOAD="${ACTIVATION_OFFLOAD:-cpu-pageable}"
case "$ACTIVATION_OFFLOAD" in
  none|cpu|cpu-pageable) ;;
  *) echo "Invalid ACTIVATION_OFFLOAD: $ACTIVATION_OFFLOAD" >&2; exit 2 ;;
esac
PYTHON="${PYTHON:-python}"
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
workers="${#devices[@]}"
(( workers >= 2 )) || { echo 'FSDP2 needs at least two allocated GPUs.' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES
export DISABLE_XFORMERS=1 SANA_CP_SIZE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export GDN_DISABLE_COMPILE="${GDN_DISABLE_COMPILE:-1}"
"$PYTHON" - <<'PY'
import os
import torch
from torch.distributed.fsdp import fully_shard
ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(ids) == len(set(ids)), 'Duplicate GPU indices'
assert torch.cuda.device_count() == len(ids), 'Visible GPU count differs from allocation'
print('Torch:', torch.__version__, 'GPU indices:', ids, flush=True)
PY
if [[ "${RESUME:-0}" == 1 ]]; then
  : "${OUTPUT:?Resume requires the existing output directory}"
  [[ -f "$OUTPUT/last.pt" ]] || { echo 'Resume checkpoint missing.' >&2; exit 2; }
else
  OUTPUT="${OUTPUT:-output/worldttn/frame-fullgrad-$(date -u +%Y%m%dT%H%M%SZ)}"
  [[ ! -e "$OUTPUT" ]] || { echo 'Use a fresh OUTPUT, or RESUME=1 with this protocol checkpoint.' >&2; exit 2; }
fi
mkdir -p "$OUTPUT"
args=(train --config configs/worldttn/frame_fullgrad.json --stage C --train-scope dit
      --optimizer-policy origin --backbone-lr 1e-6 --cross-attn-backend math
      --parallel fsdp2 --tbptt 0 --max-steps "${MAX_STEPS:-500}" --save-every "${SAVE_EVERY:-25}"
      --seed "${SEED:-3407}" --text-encoder-device cpu --dataset-root "$DATASET_ROOT"
      --activation-offload "$ACTIVATION_OFFLOAD" --activation-gpu-budget-gib "${GPU_ACTIVATION_GIB:-0}"
      --memory-trace --output "$OUTPUT")
[[ -z "${DATA_DIR:-}" ]] || args+=(--data-dir "$DATA_DIR")
[[ -z "${VAE_CACHE_DIR:-}" ]] || args+=(--vae-cache-dir "$VAE_CACHE_DIR")
[[ -z "${BASE_WEIGHTS:-}" ]] || args+=(--base-weights "$BASE_WEIGHTS")
if [[ "${RESUME:-0}" == 1 ]]; then args+=(--resume --adapter "$OUTPUT/last.pt"); fi
printf 'Protocol: frame-noisy-fullgrad-v1; workers=%s; target=%s; output=%s\n' "$workers" "${MAX_STEPS:-500}" "$OUTPUT"
printf '%q ' "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$workers" -m worldttn.cli "${args[@]}" > "$OUTPUT/command.txt"
printf '\n' >> "$OUTPUT/command.txt"
"$PYTHON" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$workers" \
  -m worldttn.cli "${args[@]}" 2>&1 | tee -a "$OUTPUT/train.log"
