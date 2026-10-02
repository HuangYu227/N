#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${SLURM_PROCID:-}" ]]; then
  echo 'Slurm tasks must use tools/ttn_slurm_train.sbatch (one process per node).' >&2
  exit 2
fi
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export DISABLE_XFORMERS=1
export SANA_CP_SIZE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
stage="${STAGE:-A}"
args=(train --parallel "${PARALLEL:-fsdp2}" --stage "$stage"
      --tbptt "${TBPTT:-2}" --seed "${SEED:-3407}" --max-steps "${MAX_STEPS:-100}"
      --save-every "${SAVE_EVERY:-50}" --text-encoder-device "${TEXT_ENCODER_DEVICE:-cpu}"
      --output "${OUTPUT:-output/worldttn/l20-${stage}-$(date -u +%Y%m%dT%H%M%SZ)}")
if [[ -n "${BATCH_FILE:-}" ]]; then
  args+=(--batch-file "$BATCH_FILE")
else
  args+=(--dataset-root "${DATASET_ROOT:-/home/newuser001/huangyu/WorldTTT/datasets/sana-wm-example}")
fi
[[ -z "${DATA_DIR:-}" ]] || args+=(--data-dir "$DATA_DIR")
[[ -z "${VAE_CACHE_DIR:-}" ]] || args+=(--vae-cache-dir "$VAE_CACHE_DIR")
[[ -z "${SANA_CONFIG:-}" ]] || args+=(--sana-config "$SANA_CONFIG")
[[ -z "${BASE_WEIGHTS:-}" ]] || args+=(--base-weights "$BASE_WEIGHTS")
[[ -z "${ADAPTER:-}" ]] || args+=(--adapter "$ADAPTER")
if [[ "${RESUME:-0}" == 1 ]]; then
  [[ -n "${ADAPTER:-}" ]] || { echo 'RESUME=1 requires ADAPTER' >&2; exit 2; }
  args+=(--resume)
fi
exec "${PYTHON:-python}" -m accelerate.commands.launch \
  --config_file configs/worldttn/l20_2gpu.yaml -m worldttn.cli "${args[@]}" "$@"
