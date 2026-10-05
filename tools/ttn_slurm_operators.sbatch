#!/usr/bin/env bash
#SBATCH --job-name=ttn-operators
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
: "${ROOT:?Set shared LTU ROOT}"
: "${PROJECT_ROOT:?Set independent WorldTTN-perf checkout}"
project_root="$(cd "$PROJECT_ROOT" && pwd)"
[[ "$project_root" != "$ROOT/WorldTTN" ]] || { echo 'Use independent checkout' >&2; exit 2; }
cd "$project_root"
source tools/ttn_cache_env.sh
export PYTHON="$ROOT/envs/worldttn/bin/python" TTN_ENTRY_MODULE=worldttn.benchmark
export OUTPUT="${OUTPUT:-$project_root/output/operators-${SLURM_JOB_ID}}"
# Worker initializes node/job-local temporary caches before importing Inductor.
exec srun --mpi=none --kill-on-bad-exit=1 --nodes=1 --ntasks=1 --gres=gpu:1 --chdir="$project_root" \
  bash "$project_root/tools/ttn_slurm_worker.sh" operators --output "$OUTPUT" \
  --precision "${PRECISION_PROTOCOL:-strict}" "$@"
