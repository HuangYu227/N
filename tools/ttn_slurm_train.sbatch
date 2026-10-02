#!/usr/bin/env bash
#SBATCH --job-name=ttn-ddp
#SBATCH --partition=short
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

# Activate the existing $ROOT/envs/worldttn prefix environment and explicitly
# export PYTHON=$ROOT/envs/worldttn/bin/python. Override resources with
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
source "$project_root/tools/ttn_cache_env.sh"
export PYTHON="${PYTHON:?Set PYTHON to $ROOT/envs/worldttn/bin/python}"
[[ "$PYTHON" == "$ROOT/envs/worldttn/bin/python" && -x "$PYTHON" ]] || {
  echo "PYTHON must be the existing executable $ROOT/envs/worldttn/bin/python; got $PYTHON" >&2
  exit 2
}
printf '[TTN cache] ROOT=%s PIP_CACHE_DIR=%s HF_HOME=%s TORCH_HOME=%s PYTHON=%s\n' \
  "$ROOT" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$PYTHON"

# The batch shell has no current training step. A STEP_NODELIST inherited
# from the submitting interactive session can belong to a different job.
node_list="${SLURM_JOB_NODELIST:?Slurm allocation node list is required}"
if [[ -z "${MASTER_ADDR:-}" ]]; then
  hostnames="$(scontrol show hostnames "$node_list")"
  master_node="${hostnames%%$'\n'*}"
  [[ -n "$master_node" ]] || { echo 'scontrol returned an empty node list' >&2; exit 2; }
  MASTER_ADDR="$("$PYTHON" -c 'import socket, sys; print(socket.gethostbyname(sys.argv[1]))' "$master_node")"
fi
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-$((15000 + SLURM_JOB_ID % 40000))}"
# Each new allocation gets fresh global ranks from srun, not inherited launch variables.
unset RANK LOCAL_RANK WORLD_SIZE NODE_RANK LOCAL_WORLD_SIZE
export DISABLE_XFORMERS=1
export SANA_CP_SIZE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
# Keep Slurm's GPU mask and NCCL's network selection; user NIC overrides propagate.
command="${COMMAND:-train}"
default_timeout=600
[[ "$command" != distributed-check ]] || default_timeout=120
printf '[TTN Slurm] job=%s nodes=%s master_node=%s MASTER_ADDR=%s MASTER_PORT=%s command=%s\n' \
  "$SLURM_JOB_ID" "$node_list" "${master_node:-explicit}" "$MASTER_ADDR" "$MASTER_PORT" "$command"
export OUTPUT="${OUTPUT:-$project_root/output/worldttn/slurm-${SLURM_JOB_ID}-${STAGE:-A}}"
args=("$command" --parallel ddp --distributed-timeout "${DISTRIBUTED_TIMEOUT:-$default_timeout}"
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
if [[ "$command" == train || "$command" == distributed-smoke ]]; then
  args+=(--activation-offload "${ACTIVATION_OFFLOAD:-cpu}")
  [[ "${MEMORY_TRACE:-1}" == 0 ]] || args+=(--memory-trace)
fi
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
# Set local TMPDIR and compilation caches inside each worker, after the shared
# cache helper above. Do not initialize a single directory on the batch node.
exec srun --mpi=none --kill-on-bad-exit=1 --distribution=block --ntasks="$SLURM_NTASKS" --ntasks-per-node=1 \
  --gres=gpu:1 --chdir="$project_root" bash "$project_root/tools/ttn_slurm_worker.sh" "${args[@]}" "$@"
