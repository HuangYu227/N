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

# Sequential single-GPU jobs keep each experiment inside the time limit.
# No distributed training allocation or optimizer is involved.
set -euo pipefail
: "${MECHANISM_OUTPUT:?Use tools/ttn_submit_mechanism.py}"
: "${PYTHON:?Use the absolute Python path from the existing prefix environment}"
: "${SLURM_ARRAY_TASK_ID:?Submit as the mechanism array}"
dispatch=""
if [[ -f "$MECHANISM_OUTPUT/plan.json" ]]; then
  dispatch="$("$PYTHON" - "$MECHANISM_OUTPUT/plan.json" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
from pathlib import Path
plan = json.loads(Path(sys.argv[1]).read_text())
index = int(sys.argv[2])
names = plan["variants"]
if not 0 <= index < len(names): raise ValueError("array task is outside the planned variants")
if "interventions" in plan:
    name = names[index]
    ablation, history, methods = plan["interventions"][name]
    fields = [name, ablation, history, " ".join(methods)]
    if any(not isinstance(v, str) or not v or any(c in v for c in "\t\r\n") for v in fields):
        raise ValueError("invalid planned intervention")
    print("\t".join(fields))
PY
)"
fi
if [[ -n "$dispatch" ]]; then
  IFS=$'\t' read -r variant TTN_ABLATION HISTORY_SOURCE EVAL_METHODS <<< "$dispatch"
  export TTN_ABLATION HISTORY_SOURCE EVAL_METHODS
else
  # Legacy plans and direct submissions retain their original array mapping.
  case "$SLURM_ARRAY_TASK_ID" in
  0) variant=full;       export TTN_ABLATION=full HISTORY_SOURCE=generated EVAL_METHODS='sana ttn' ;;
  1) variant=no-ttt;     export TTN_ABLATION=no-ttt HISTORY_SOURCE=generated EVAL_METHODS=ttn ;;
  2) variant=identity;   export TTN_ABLATION=identity HISTORY_SOURCE=generated EVAL_METHODS=ttn ;;
  3) variant=gt-history; export TTN_ABLATION=full HISTORY_SOURCE=gt EVAL_METHODS='sana ttn' ;;
  4) variant=no-local; export TTN_ABLATION=no-local HISTORY_SOURCE=generated EVAL_METHODS=ttn ;;
  5) variant=no-persistent; export TTN_ABLATION=no-persistent HISTORY_SOURCE=generated EVAL_METHODS=ttn ;;
  *) echo 'Expected array task 0..5' >&2; exit 2 ;;
  esac
fi
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
