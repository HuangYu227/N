#!/usr/bin/env bash
#SBATCH --job-name=ttn-meta-tests
#SBATCH --partition=short
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
: "${ROOT:?Set shared LTU ROOT}"
: "${PROJECT_ROOT:?Set the isolated dual-psi checkout}"
cd "$PROJECT_ROOT"
source tools/ttn_cache_env.sh
export PYTHON="$ROOT/envs/worldttn/bin/python" TTN_ENTRY_MODULE=pytest SANA_CP_SIZE=1
export META_TEST_OUTPUT="$PROJECT_ROOT/output/meta-tests-${SLURM_JOB_ID}"
mkdir -p "$META_TEST_OUTPUT"
unset RANK LOCAL_RANK WORLD_SIZE NODE_RANK LOCAL_WORLD_SIZE
hosts="$(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")"
export MASTER_ADDR="$("$PYTHON" -c 'import socket,sys; print(socket.gethostbyname(sys.argv[1]))' "${hosts%%$'\n'*}")"
export MASTER_PORT="$((15000 + SLURM_JOB_ID % 40000))"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" NCCL_DEBUG=WARN NCCL_SOCKET_FAMILY=AF_INET
if [[ "${PROXIMAL_MEMORY_TESTS:-0}" == 1 ]]; then
  case "${SLURM_NTASKS:?}" in
    1) tests=(tests/ttn/test_proximal.py tests/ttn/test_memory_cache.py tests/ttn/test_proximal_runtime.py
              tests/ttn/test_proximal_evaluation.py tests/ttn/test_proximal_device.py tests/ttn/test_sana_camera_restore.py
              tests/ttn/test_meta_training.py::test_production_flow_records_exact_scheduler_sigma_separately_from_mapped_timestep
              tests/ttn/test_activation_checkpoint.py tests/ttn/test_mixed_offload.py) ;;
    2|3|4) tests=(
              'tests/ttn/test_proximal_cuda.py::test_proximal_fsdp_cache_only_future_credit[live-cache]'
              'tests/ttn/test_proximal_cuda.py::test_proximal_fsdp_cache_only_future_credit[detached-cache-control]'
              tests/ttn/test_proximal_cuda.py::test_proximal_fsdp_offload_and_exact_resume
              tests/ttn/test_proximal_cuda.py::test_proximal_fsdp_mixed_offload_matches_cpu) ;;
    *) echo 'Proximal acceptance requires one, two, three or four nodes/tasks' >&2; exit 2 ;;
  esac
else
case "${SLURM_NTASKS:?}" in
  1) tests=(tests/ttn/test_meta_core.py tests/ttn/test_sana_camera_restore.py tests/ttn/test_meta_sampler.py) ;;
  4) tests=(tests/ttn/test_meta_cuda.py) ;;
  *) echo 'Meta acceptance requires one or four nodes/tasks' >&2; exit 2 ;;
esac
if [[ "${FULL_MEMORY_TESTS:-0}" == 1 ]]; then
  "$PYTHON" -c 'from diffusers import FlowMatchEulerDiscreteScheduler'
  if [[ "$SLURM_NTASKS" == 4 ]]; then
    tests=(tests/ttn/test_full_training.py::test_full_memory_four_gpu_future_credit_and_resume)
  else
    tests=(tests/ttn/test_meta_core.py tests/ttn/test_full_training.py tests/ttn/test_activation_checkpoint.py
           tests/ttn/test_training.py::test_save_on_cpu_reduces_live_cuda_saved_activations_without_changing_gradients
           -k 'not four_gpu')
  fi
fi
fi
if [[ "${PROXIMAL_MEMORY_TESTS:-0}" == 1 && "$SLURM_NTASKS" != 1 ]]; then
  # A rank-local pytest failure must not let that rank enter another NCCL test.
  suite_output="$META_TEST_OUTPUT"
  suite_port="$MASTER_PORT"
  for index in "${!tests[@]}"; do
    export MASTER_PORT="$((suite_port + index))" META_TEST_OUTPUT="$suite_output/case-$index"
    mkdir -p "$META_TEST_OUTPUT"
    printf '[TTN acceptance] start case=%s test=%s port=%s logs=%s/rank-%%t.out\n' \
      "$index" "${tests[index]}" "$MASTER_PORT" "$META_TEST_OUTPUT"
    if srun --mpi=none --kill-on-bad-exit=1 --ntasks="$SLURM_NTASKS" --ntasks-per-node=1 --gres=gpu:1 \
        --chdir="$PROJECT_ROOT" --output="$META_TEST_OUTPUT/rank-%t.out" \
        bash tools/ttn_slurm_worker.sh -x -vv -s "${tests[index]}" "$@"; then
      printf '[TTN acceptance] completed case=%s\n' "$index"
    else
      status=$?
      printf '[TTN acceptance] failed case=%s exit=%s logs=%s\n' "$index" "$status" "$META_TEST_OUTPUT" >&2
      for log in "$META_TEST_OUTPUT"/rank-*.out; do
        [[ -f "$log" ]] || continue
        printf '[TTN acceptance] log=%s\n' "$log" >&2
        tail -n 60 "$log" >&2
      done
      exit "$status"
    fi
  done
  exit 0
fi
exec srun --mpi=none --kill-on-bad-exit=1 --ntasks="$SLURM_NTASKS" --ntasks-per-node=1 --gres=gpu:1 \
  --chdir="$PROJECT_ROOT" bash tools/ttn_slurm_worker.sh -q -s "${tests[@]}" "$@"
