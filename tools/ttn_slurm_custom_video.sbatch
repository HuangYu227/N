#!/usr/bin/env bash
#SBATCH --job-name=ttn-custom-video
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-ttn-custom-video-%j.out
set -euo pipefail
: "${ROOT:?set ROOT}"
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
: "${CUSTOM_TRAINING_RUN:?set CUSTOM_TRAINING_RUN to the completed step100 snapshot}"
: "${CUSTOM_OUTPUT:?set CUSTOM_OUTPUT}"
[[ "${SLURM_NTASKS:-0}" == 1 && "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}" == 1 ]] || {
    echo 'Custom paired inference requires one node/task/GPU' >&2; exit 2;
}
cd "$PROJECT_ROOT"
source tools/ttn_cache_env.sh
export PYTHON="$ROOT/envs/worldttn/bin/python"
export TTN_ENTRY_MODULE=tools.ttn_custom_inference
export DISABLE_XFORMERS=1 SANA_CP_SIZE=1 SANA_WM_STAGE1_KV_SAVE_STRIDE=1
export GDN_DISABLE_COMPILE=1 OMP_NUM_THREADS=8
unset RANK LOCAL_RANK WORLD_SIZE NODE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
[[ -x "$PYTHON" && -f "$CUSTOM_TRAINING_RUN/run_config.json" && -f "$CUSTOM_TRAINING_RUN/last.pt" ]]
srun --mpi=none --kill-on-bad-exit=1 --nodes=1 --ntasks=1 --gres=gpu:1 \
    bash tools/ttn_slurm_worker.sh --training-run "$CUSTOM_TRAINING_RUN" \
    --case "${CUSTOM_CASE:-assets/worldttn/study_static/case.json}" --output "$CUSTOM_OUTPUT" "$@"
