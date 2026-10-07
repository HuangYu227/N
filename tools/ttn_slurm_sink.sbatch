#!/usr/bin/env bash
#SBATCH --job-name=ttn-sink
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-ttn-sink-%j.out
set -euo pipefail
: "${ROOT:?set ROOT}"
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
: "${SINK_OUTPUT:?submit with tools.ttn_submit_sink}"
[[ "${SLURM_NTASKS:-0}" == 1 && "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}" == 1 ]] || {
    echo 'Sink inference requires one node/task/GPU' >&2; exit 2;
}
cd "$PROJECT_ROOT"
source tools/ttn_cache_env.sh
export PYTHON="$ROOT/envs/worldttn/bin/python"
export TTN_ENTRY_MODULE=tools.ttn_submit_sink
export DISABLE_XFORMERS=1 SANA_CP_SIZE=1 SANA_WM_STAGE1_KV_SAVE_STRIDE=1
export GDN_DISABLE_COMPILE="${GDN_DISABLE_COMPILE:-1}" OMP_NUM_THREADS=8
unset RANK LOCAL_RANK WORLD_SIZE NODE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
[[ -x "$PYTHON" && -f "$SINK_OUTPUT/plan.json" ]]
task_index="${SLURM_ARRAY_TASK_ID:-0}"
finish() {
    code=$?
    if [[ "$code" != 0 ]]; then
        "$PYTHON" -u -m tools.ttn_submit_sink --output "$SINK_OUTPUT" \
            --mark-failed "$task_index" --failure-exit-code "$code" || true
        if [[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
            scancel --state=PENDING "$SLURM_ARRAY_JOB_ID" || true
        fi
    fi
    exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
srun --mpi=none --kill-on-bad-exit=1 --nodes=1 --ntasks=1 --gres=gpu:1 \
    bash tools/ttn_slurm_worker.sh --output "$SINK_OUTPUT" --run-variant "$task_index"
