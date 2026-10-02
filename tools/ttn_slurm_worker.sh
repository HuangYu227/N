#!/usr/bin/env bash
# Run on EACH allocated node. Only temporary compilation files belong in /tmp;
# the batch launcher has already set shared permanent caches and checked PYTHON.
set -euo pipefail
: "${SLURM_JOB_ID:?Run this worker through the Slurm batch launcher}"
: "${SLURM_PROCID:?Slurm global rank is required}"
: "${PYTHON:?The batch launcher must provide the existing prefix interpreter}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$SLURM_PROCID" =~ ^[0-9]+$ ]] || {
  echo 'Invalid Slurm job ID or global rank' >&2; exit 2;
}

# An explicit /tmp template ignores an inherited NFS TMPDIR. mktemp makes this
# private and unique even if the same job/rank is launched again on this node.
scratch_prefix="/tmp/worldttn-${SLURM_JOB_ID}-rank${SLURM_PROCID}."
scratch="$(mktemp -d "${scratch_prefix}XXXXXX")"
[[ "$scratch" == "$scratch_prefix"* && -n "${scratch#"$scratch_prefix"}" \
   && "${scratch#"$scratch_prefix"}" != */* && -d "$scratch" && ! -L "$scratch" ]] || {
  echo "Refusing unexpected compilation directory: $scratch" >&2; exit 2;
}
child_pid=""
cleanup() {
  if [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  # Delete only the freshly allocated private directory, after Python exits.
  [[ "$scratch" == "$scratch_prefix"* && "${scratch#"$scratch_prefix"}" != */* \
     && -d "$scratch" && ! -L "$scratch" ]] && rm -rf -- "$scratch"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

export TMPDIR="$scratch/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export TORCHINDUCTOR_CACHE_DIR="$scratch/inductor"
export TRITON_CACHE_DIR="$scratch/triton"
export TORCH_EXTENSIONS_DIR="$scratch/extensions"
export CUDA_CACHE_PATH="$scratch/cuda"
export PYTHONPYCACHEPREFIX="$scratch/pycache"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
# Read before SANA's import-time decorators and CUDA initialization. Defaults
# preserve compilation and asynchronous execution; debugging is opt-in.
export GDN_DISABLE_COMPILE="${GDN_DISABLE_COMPILE:-0}"
export GDN_DISABLE_COMPLEX_COMPILE="${GDN_DISABLE_COMPLEX_COMPILE:-0}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$PYTHONPYCACHEPREFIX"
printf '[TTN compile] host=%s rank=%s PYTHON=%s TMPDIR=%s TORCHINDUCTOR_CACHE_DIR=%s TRITON_CACHE_DIR=%s TORCH_EXTENSIONS_DIR=%s CUDA_CACHE_PATH=%s PYTHONPYCACHEPREFIX=%s TORCHINDUCTOR_COMPILE_THREADS=%s GDN_DISABLE_COMPILE=%s GDN_DISABLE_COMPLEX_COMPILE=%s CUDA_LAUNCH_BLOCKING=%s\n' \
  "$(hostname)" "$SLURM_PROCID" "$PYTHON" "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" \
  "$PYTHONPYCACHEPREFIX" "$TORCHINDUCTOR_COMPILE_THREADS" \
  "$GDN_DISABLE_COMPILE" "$GDN_DISABLE_COMPLEX_COMPILE" "$CUDA_LAUNCH_BLOCKING"

# Keep the launcher alive to clean up on success, failure or Slurm TERM, and
# propagate Python's status so --kill-on-bad-exit still terminates peer ranks.
"$PYTHON" -u -m worldttn.cli "$@" &
child_pid=$!
set +e
wait "$child_pid"
status=$?
child_pid=""
exit "$status"
