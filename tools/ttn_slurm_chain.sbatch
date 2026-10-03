#!/usr/bin/env bash
# Resources and absolute paths are provided by ttn_slurm_chain.py.
set -euo pipefail
: "${PYTHON:?Use the existing prefix environment}"
: "${ROOT:?Shared permanent root is required}"
[[ "$PYTHON" == "$ROOT/envs/worldttn/bin/python" && -x "$PYTHON" ]] || {
  echo "Expected the existing prefix Python: $ROOT/envs/worldttn/bin/python" >&2; exit 2;
}
exec "$PYTHON" -u "${PROJECT_ROOT:?}/tools/ttn_slurm_chain.py" worker "$1" "$2"
