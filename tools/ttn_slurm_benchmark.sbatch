#!/usr/bin/env bash
#SBATCH --job-name=ttn-perf
#SBATCH --partition=short
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
: "${ROOT:?Set shared LTU ROOT}"
: "${PROJECT_ROOT:?Set the independent WorldTTN-perf checkout}"
: "${BENCHMARK_SNAPSHOT:?Set an immutable completed training snapshot}"
project_root="$(cd "$PROJECT_ROOT" && pwd)"
[[ "$project_root" != "$ROOT/WorldTTN" ]] || { echo 'Do not benchmark in the live training checkout' >&2; exit 2; }
[[ "$(git -C "$project_root" branch --show-current)" == codex/ttn-fp32-acceleration ]] || {
  echo 'Expected codex/ttn-fp32-acceleration checkout' >&2; exit 2;
}
[[ -f "$BENCHMARK_SNAPSHOT/last.pt" && -f "$BENCHMARK_SNAPSHOT/benchmark_snapshot.json" ]] || {
  echo 'Use tools/ttn_benchmark_snapshot.py to pin weights AND optimizer/RNG/cursor shards' >&2; exit 2;
}
export PROJECT_ROOT="$project_root" COMMAND=train STAGE=C PARALLEL=fsdp2 RESUME=1
export ADAPTER="$BENCHMARK_SNAPSHOT/last.pt"
export PYTHON="$ROOT/envs/worldttn/bin/python"
export TRAIN_SCOPE="${TRAIN_SCOPE:-ttn-visual}" TBPTT="${TBPTT:-2}"
export CROSS_ATTN_BACKEND="${CROSS_ATTN_BACKEND:-math}" ACTIVATION_OFFLOAD="${ACTIVATION_OFFLOAD:-cpu}"
export GDN_DISABLE_COMPILE="${GDN_DISABLE_COMPILE:-1}"
export TTN_CORE_BACKEND="${TTN_CORE_BACKEND:-reference}" TTN_PSI_BACKEND="${TTN_PSI_BACKEND:-reference}"
export OUTPUT="${OUTPUT:-$ROOT/WorldTTN-perf/output/benchmark-${SLURM_JOB_ID}}"
[[ "$OUTPUT" != "$BENCHMARK_SNAPSHOT" && ! -e "$OUTPUT" ]] || { echo 'Use a new private OUTPUT' >&2; exit 2; }
extra=()
case "${BENCHMARK_MODE:-throughput}" in
  throughput)
    unset TORCH_LOGS
    export TTN_PROFILE=0 TTN_LAYOUT_AUDIT=0 CUDA_LAUNCH_BLOCKING=0
    extra+=(--ttn-benchmark-stable-steps "${STABLE_STEPS:-10}") ;;
  profile)
    export TTN_PROFILE=1 TTN_LAYOUT_AUDIT=0 MAX_STEPS=1
    # Stable count stays zero: profiled samples cannot promote a backend.
    extra+=(--ttn-profiler-trace "$OUTPUT/profiler") ;;
  layouts)
    export TTN_PROFILE=0 TTN_LAYOUT_AUDIT=1 TORCH_LOGS=recompiles
    # Use an explicit absolute endpoint for a resumed snapshot.
    : "${MAX_STEPS:?Set MAX_STEPS to snapshot step + 3 for layout validation}" ;;
  *) echo 'BENCHMARK_MODE must be throughput, profile or layouts' >&2; exit 2 ;;
esac
# No pip/conda mutations, no branch switch, no automatic job submission chain.
exec bash "$project_root/tools/ttn_slurm_train.sbatch" "${extra[@]}" "$@"
