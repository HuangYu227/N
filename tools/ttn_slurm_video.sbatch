#!/usr/bin/env bash
#SBATCH --job-name=ttn-video
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-ttn-video-%j.out
set -euo pipefail
: "${ROOT:?set ROOT}"
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
: "${VIDEO_EVALUATION:?set VIDEO_EVALUATION}"
: "${VIDEO_OUTPUT:?set VIDEO_OUTPUT}"
cd "$PROJECT_ROOT"
source tools/ttn_cache_env.sh
export DISABLE_XFORMERS=1 SANA_CP_SIZE=1 PYTHONUNBUFFERED=1
srun "$ROOT/envs/worldttn/bin/python" -m tools.ttn_decode_comparison \
    --evaluation "$VIDEO_EVALUATION" --output "$VIDEO_OUTPUT" --chunks 5
