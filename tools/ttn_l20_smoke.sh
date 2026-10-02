#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${SLURM_PROCID:-}" ]]; then
  echo 'Slurm tasks must use tools/ttn_slurm_train.sbatch with COMMAND=distributed-smoke.' >&2
  exit 2
fi
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export DISABLE_XFORMERS=1
export SANA_CP_SIZE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
args=(distributed-smoke --parallel "${PARALLEL:-fsdp2}" --tbptt "${TBPTT:-1}"
      --frames 13 --steps "${STEPS:-4}" --seed "${SEED:-3407}"
      --output "${OUTPUT:-output/worldttn/l20-smoke-$(date -u +%Y%m%dT%H%M%SZ)}")
[[ -z "${BASE_WEIGHTS:-}" ]] || args+=(--base-weights "$BASE_WEIGHTS")
[[ -z "${BATCH_FILE:-}" ]] || args+=(--batch-file "$BATCH_FILE")
[[ -z "${SANA_CONFIG:-}" ]] || args+=(--sana-config "$SANA_CONFIG")
exec "${PYTHON:-python}" -m accelerate.commands.launch \
  --config_file configs/worldttn/l20_2gpu.yaml -m worldttn.cli "${args[@]}" "$@"
