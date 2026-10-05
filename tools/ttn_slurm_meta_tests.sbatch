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
case "${SLURM_NTASKS:?}" in
  1) tests=(tests/ttn/test_meta_core.py tests/ttn/test_sana_camera_restore.py tests/ttn/test_meta_sampler.py) ;;
  4) tests=(tests/ttn/test_meta_cuda.py) ;;
  *) echo 'Meta acceptance requires one or four nodes/tasks' >&2; exit 2 ;;
esac
exec srun --mpi=none --kill-on-bad-exit=1 --ntasks="$SLURM_NTASKS" --ntasks-per-node=1 --gres=gpu:1 \
  --chdir="$PROJECT_ROOT" bash tools/ttn_slurm_worker.sh -q -s "${tests[@]}" "$@"
