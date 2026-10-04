#!/usr/bin/env bash
#SBATCH --job-name=ttn-mechanism
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%1
#SBATCH --output=slurm-%x-%A_%a.out

# Four sequential single-GPU jobs keep each experiment inside the time limit.
# No distributed training allocation or optimizer is involved.
set -euo pipefail
: "${MECHANISM_OUTPUT:?Use tools/ttn_submit_mechanism.py}"
: "${PYTHON:?Use the absolute Python path from the existing prefix environment}"
case "${SLURM_ARRAY_TASK_ID:?Submit as the four-task array}" in
  0) variant=full;       export TTN_ABLATION=full HISTORY_SOURCE=generated EVAL_METHODS='sana ttn' ;;
  1) variant=no-ttt;     export TTN_ABLATION=no-ttt HISTORY_SOURCE=generated EVAL_METHODS=ttn ;;
  2) variant=identity;   export TTN_ABLATION=identity HISTORY_SOURCE=generated EVAL_METHODS=ttn ;;
  3) variant=gt-history; export TTN_ABLATION=full HISTORY_SOURCE=gt EVAL_METHODS='sana ttn' ;;
  *) echo 'Expected array task 0..3' >&2; exit 2 ;;
esac
export COMMAND=evaluate FRAMES=61 STATE_DIAGNOSTICS=1
export OUTPUT="$MECHANISM_OUTPUT/$variant"
project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$project_root"
finish() {
  code=$?
  status=completed
  [[ "$code" == 0 ]] || status=failed
  "$PYTHON" -u -m worldttn.mechanism_evaluation "$MECHANISM_OUTPUT" --variant "$variant" --status "$status" || {
    [[ "$code" != 0 ]] || code=1
  }
  exit "$code"
}
trap finish EXIT
"$PYTHON" -u -m worldttn.mechanism_evaluation "$MECHANISM_OUTPUT" --variant "$variant" --status running
printf '[TTN mechanism job] variant=%s ablation=%s history=%s methods=%s output=%s\n' \
  "$variant" "$TTN_ABLATION" "$HISTORY_SOURCE" "$EVAL_METHODS" "$OUTPUT"
bash "$project_root/tools/ttn_slurm_eval.sbatch"
