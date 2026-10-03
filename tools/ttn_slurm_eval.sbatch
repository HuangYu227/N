#!/usr/bin/env bash
#SBATCH --job-name=ttn-eval
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

# Sequential SANA/TTN inference on one GPU, with the same inputs/noise. No DDP
# or training update is needed. Begin with 61 latents; 121 is the full example
# horizon and may need a longer allocation. STEPS=20 is an explicit diagnostic
# schedule, not the 4-step engineering smoke nor a claim of optimal quality.
# TRAINING_RUN=/shared/WorldTTN/output/worldttn/slurm-423058-A \
#   sbatch tools/ttn_slurm_eval.sbatch
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${TRAINING_RUN:?Set TRAINING_RUN to the completed training output directory}"
[[ "${SLURM_NTASKS:-0}" == 1 && "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}" == 1 ]] || {
  echo 'Paired inference needs exactly one node/task/GPU; run methods sequentially' >&2; exit 2;
}
project_root="$(cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Shared project root is required}}" && pwd)"
[[ -f "$project_root/worldttn/evaluation.py" ]] || { echo "Not a TTN evaluation checkout: $project_root" >&2; exit 2; }
cd "$project_root"
source "$project_root/tools/ttn_cache_env.sh"
export PYTHON="${PYTHON:?Set PYTHON to $ROOT/envs/worldttn/bin/python}"
[[ "$PYTHON" == "$ROOT/envs/worldttn/bin/python" && -x "$PYTHON" ]] || {
  echo "PYTHON must be the existing executable $ROOT/envs/worldttn/bin/python; got $PYTHON" >&2; exit 2;
}
[[ -f "$TRAINING_RUN/run_config.json" && -f "$TRAINING_RUN/train.jsonl" ]] || {
  echo "Missing completed training records: $TRAINING_RUN" >&2; exit 2;
}
unset RANK LOCAL_RANK WORLD_SIZE NODE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
export DISABLE_XFORMERS=1 SANA_CP_SIZE=1 SANA_WM_STAGE1_KV_SAVE_STRIDE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export GDN_DISABLE_COMPILE="${GDN_DISABLE_COMPILE:-1}"
export GDN_DISABLE_COMPLEX_COMPILE="${GDN_DISABLE_COMPLEX_COMPILE:-0}"
export DPM_TQDM=True
output="${OUTPUT:-$project_root/output/worldttn/eval-${SLURM_JOB_ID}}"
args=(evaluate --parallel single --training-run "$TRAINING_RUN" --output "$output"
      --frames "${FRAMES:-61}" --steps "${STEPS:-20}" --eval-cases "${EVAL_CASES:-1}"
      --seed "${SEED:-3407}" --cfg-scale "${CFG_SCALE:-4.5}" --cached-blocks "${CACHED_BLOCKS:-2}"
      --cross-attn-backend "${CROSS_ATTN_BACKEND:-math}")
[[ -z "${ADAPTER:-}" ]] || args+=(--adapter "$ADAPTER")
[[ -z "${BASE_WEIGHTS:-}" ]] || args+=(--base-weights "$BASE_WEIGHTS")
[[ -z "${SANA_CONFIG:-}" ]] || args+=(--sana-config "$SANA_CONFIG")
[[ -z "${CONFIG:-}" ]] || args+=(--config "$CONFIG")
printf '[TTN eval Slurm] job=%s TRAINING_RUN=%s frames=%s steps=%s OUTPUT=%s PYTHON=%s\n' \
  "$SLURM_JOB_ID" "$TRAINING_RUN" "${FRAMES:-61}" "${STEPS:-20}" "$output" "$PYTHON"
# The same worker as training creates private node-local /tmp compiler caches.
exec srun --mpi=none --kill-on-bad-exit=1 --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --gres=gpu:1 --chdir="$project_root" bash "$project_root/tools/ttn_slurm_worker.sh" "${args[@]}" "$@"
