#!/usr/bin/env bash
# Resources and absolute paths are provided by ttn_slurm_chain.py.
set -euo pipefail
: "${PYTHON:?Use the existing prefix environment}"
: "${ROOT:?Shared permanent root is required}"
[[ "$PYTHON" == "$ROOT/envs/worldttn/bin/python" && -x "$PYTHON" ]] || {
  echo "Expected the existing prefix Python: $ROOT/envs/worldttn/bin/python" >&2; exit 2;
}
# Module execution puts this checkout before stale installed/PYTHONPATH packages.
cd "${PROJECT_ROOT:?}"
exec "$PYTHON" -u -m tools.ttn_slurm_chain worker "$1" "$2"
