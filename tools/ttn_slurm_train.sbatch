#!/usr/bin/env bash
#SBATCH --job-name=ttn-ddp
#SBATCH --partition=short
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

# Submit from the shared checkout root with an activated environment, or set
# PROJECT_ROOT and PYTHON to shared absolute paths. Override resources with
# sbatch --nodes=... --time=... . The script needs no Accelerate configuration.
# COMMAND=distributed-check sbatch tools/ttn_slurm_train.sbatch
# COMMAND=distributed-smoke TBPTT=1 sbatch tools/ttn_slurm_train.sbatch
# DATASET_ROOT=/shared/data/sana-wm-example STAGE=A sbatch tools/ttn_slurm_train.sbatch
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SLURM_NTASKS:?Slurm task count is required}"
project_root="$(cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Shared project root is required}}" && pwd)"
[[ -f "$project_root/worldttn/cli.py" ]] || { echo "Not a TTN checkout: $project_root" >&2; exit 2; }
cd "$project_root"

if [[ -z "${MASTER_ADDR:-}" ]]; then
  node_list="${SLURM_STEP_NODELIST:-${SLURM_JOB_NODELIST:?Slurm node list is required}}"
  hostnames="$(scontrol show hostnames "$node_list")"
  MASTER_ADDR="${hostnames%%$'\n'*}"
  [[ -n "$MASTER_ADDR" ]] || { echo 'scontrol returned an empty node list' >&2; exit 2; }
fi
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-$((15000 + SLURM_JOB_ID % 40000))}"
# Each new allocation gets fresh global ranks from srun, not inherited launch variables.
unset RANK LOCAL_RANK WORLD_SIZE NODE_RANK LOCAL_WORLD_SIZE
export DISABLE_XFORMERS=1
export SANA_CP_SIZE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Keep Slurm's GPU mask and NCCL's network selection; user NIC overrides propagate.
command="${COMMAND:-train}"
export OUTPUT="${OUTPUT:-$project_root/output/worldttn/slurm-${SLURM_JOB_ID}-${STAGE:-A}}"
args=("$command" --parallel ddp --distributed-timeout "${DISTRIBUTED_TIMEOUT:-600}"
      --seed "${SEED:-3407}" --output "$OUTPUT")
case "$command" in
  train)
    args+=(--stage "${STAGE:-A}" --tbptt "${TBPTT:-2}" --max-steps "${MAX_STEPS:-100}"
           --save-every "${SAVE_EVERY:-50}" --text-encoder-device "${TEXT_ENCODER_DEVICE:-cpu}")
    if [[ -n "${BATCH_FILE:-}" ]]; then
      args+=(--batch-file "$BATCH_FILE")
    else
      args+=(--dataset-root "${DATASET_ROOT:?Set shared DATASET_ROOT or BATCH_FILE}")
    fi
    ;;
  distributed-smoke)
    read -r -a stages <<< "${STAGES:-A B C}"
    args+=(--stages "${stages[@]}" --tbptt "${TBPTT:-1}" --frames "${FRAMES:-13}"
           --steps "${STEPS:-4}" --latent-height "${LATENT_HEIGHT:-22}" --latent-width "${LATENT_WIDTH:-40}")
    [[ -z "${BATCH_FILE:-}" ]] || args+=(--batch-file "$BATCH_FILE")
    ;;
  distributed-check) ;;
  *) echo "Unsupported COMMAND: $command" >&2; exit 2 ;;
esac
[[ -z "${CONFIG:-}" ]] || args+=(--config "$CONFIG")
[[ -z "${DATA_DIR:-}" ]] || args+=(--data-dir "$DATA_DIR")
[[ -z "${VAE_CACHE_DIR:-}" ]] || args+=(--vae-cache-dir "$VAE_CACHE_DIR")
[[ -z "${SANA_CONFIG:-}" ]] || args+=(--sana-config "$SANA_CONFIG")
[[ -z "${BASE_WEIGHTS:-}" ]] || args+=(--base-weights "$BASE_WEIGHTS")
[[ -z "${ADAPTER:-}" ]] || args+=(--adapter "$ADAPTER")
if [[ "${RESUME:-0}" == 1 ]]; then
  [[ -n "${ADAPTER:-}" ]] || { echo 'RESUME=1 requires ADAPTER' >&2; exit 2; }
  [[ "$command" == train ]] || { echo 'RESUME=1 is only supported for train' >&2; exit 2; }
  args+=(--resume)
fi
exec srun --mpi=none --kill-on-bad-exit=1 --ntasks="$SLURM_NTASKS" --ntasks-per-node=1 \
  --gres=gpu:1 --chdir="$project_root" "${PYTHON:-python}" -u -m worldttn.cli "${args[@]}" "$@"
