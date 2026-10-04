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
# Full C + DiT: STAGE=C TRAIN_SCOPE=dit PARALLEL=fsdp2 BACKBONE_LR=1e-6 MAX_STEPS=1
# sbatch --mem=256G tools/ttn_slurm_train.sbatch (first verify one real update).
# Single-card, one-update CUDA diagnosis (keep the existing prefix environment):
# COMMAND=diagnose-update STAGE=A CUDA_TRACE=1 GDN_DISABLE_COMPILE=1 CUDA_LAUNCH_BLOCKING=1 sbatch --nodes=1 --ntasks=1 --ntasks-per-node=1 tools/ttn_slurm_train.sbatch
# Text SDPA backend A/B: add CROSS_ATTN_BACKEND=math, then flash after math passes.
# This scopes backend selection to text cross-attention and preserves tensor layouts/offload.
# For synthetic diagnose-update only: DIAGNOSTIC_UNMASK_ALL_VALID=1 lets FLASH
# receive None after each text mask is checked to be all ones; padding is rejected.
# Geometry/rotary backward A/B (same model, four-node DDP and CPU offload):
# GDN_DISABLE_COMPILE=1 CUDA_LAUNCH_BLOCKING=1 COMMAND=distributed-smoke STAGES=A TBPTT=1 sbatch tools/ttn_slurm_train.sbatch
# GDN_DISABLE_COMPILE=0 CUDA_LAUNCH_BLOCKING=1 COMMAND=distributed-smoke STAGES=A TBPTT=1 sbatch tools/ttn_slurm_train.sbatch
# After eager succeeds, test only the two complex/RoPE wrappers eagerly:
# GDN_DISABLE_COMPILE=0 GDN_DISABLE_COMPLEX_COMPILE=1 COMMAND=distributed-smoke STAGES=A TBPTT=1 sbatch tools/ttn_slurm_train.sbatch
# These import-time switches never change the underlying Tensor mathematics;
# they do not disable handwritten Triton/FLA kernels. Run the comparisons serially.
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
parallel="${PARALLEL:-ddp}"
# Joint DiT needs sharded parameters/gradients/AdamW states on 40GB cards.
# Existing frozen-backbone TTN jobs retain DDP unless explicitly overridden.
if [[ "$command" == train && "${TRAIN_SCOPE:-ttn}" == dit && -z "${PARALLEL:-}" ]]; then parallel=fsdp2; fi
case "$parallel" in ddp|fsdp2|single) ;; *) echo "Unsupported PARALLEL: $parallel" >&2; exit 2 ;; esac
# A one-card diagnostic must not create a process group or wrap the model in DDP.
if [[ "$command" == diagnose-update && "$SLURM_NTASKS" == 1 ]]; then parallel=single; fi
default_timeout=600
[[ "$command" != distributed-check ]] || default_timeout=120
printf '[TTN Slurm] job=%s nodes=%s master_node=%s MASTER_ADDR=%s MASTER_PORT=%s command=%s parallel=%s\n' \
  "$SLURM_JOB_ID" "$node_list" "${master_node:-explicit}" "$MASTER_ADDR" "$MASTER_PORT" "$command" "$parallel"
export OUTPUT="${OUTPUT:-$project_root/output/worldttn/slurm-${SLURM_JOB_ID}-${STAGE:-A}}"
args=("$command" --parallel "$parallel" --distributed-timeout "${DISTRIBUTED_TIMEOUT:-$default_timeout}"
      --seed "${SEED:-3407}" --output "$OUTPUT")
case "$command" in
  train)
    args+=(--stage "${STAGE:-A}" --tbptt "${TBPTT:-2}" --max-steps "${MAX_STEPS:-100}"
           --save-every "${SAVE_EVERY:-50}" --text-encoder-device "${TEXT_ENCODER_DEVICE:-cpu}"
           --train-scope "${TRAIN_SCOPE:-ttn}" --backbone-lr "${BACKBONE_LR:-1e-6}")
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
  diagnose-update)
    args+=(--stage "${STAGE:-A}" --tbptt "${TBPTT:-1}" --frames "${FRAMES:-13}"
           --latent-height "${LATENT_HEIGHT:-22}" --latent-width "${LATENT_WIDTH:-40}")
    [[ -z "${BATCH_FILE:-}" ]] || args+=(--batch-file "$BATCH_FILE")
    ;;
  *) echo "Unsupported COMMAND: $command" >&2; exit 2 ;;
esac
if [[ "$command" == train || "$command" == distributed-smoke || "$command" == diagnose-update ]]; then
  args+=(--activation-offload "${ACTIVATION_OFFLOAD:-cpu}" --cross-attn-backend "${CROSS_ATTN_BACKEND:-auto}")
  [[ "${MEMORY_TRACE:-1}" == 0 ]] || args+=(--memory-trace)
  [[ "${CUDA_TRACE:-0}" == 0 ]] || args+=(--cuda-trace)
fi
[[ "${DIAGNOSTIC_UNMASK_ALL_VALID:-0}" == 0 ]] || args+=(--diagnostic-unmask-all-valid)
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
if [[ "${UNFREEZE:-0}" == 1 ]]; then
  [[ "$command" == train && "${TRAIN_SCOPE:-ttn}" == dit && -n "${ADAPTER:-}" && "${RESUME:-0}" != 1 ]] || {
    echo 'UNFREEZE=1 requires train, TRAIN_SCOPE=dit, ADAPTER, and no RESUME=1' >&2; exit 2;
  }
  args+=(--unfreeze)
fi
# Set local TMPDIR and compilation caches inside each worker, after the shared
# cache helper above. Do not initialize a single directory on the batch node.
exec srun --mpi=none --kill-on-bad-exit=1 --distribution=block --ntasks="$SLURM_NTASKS" --ntasks-per-node=1 \
  --gres=gpu:1 --chdir="$project_root" bash "$project_root/tools/ttn_slurm_worker.sh" "${args[@]}" "$@"
