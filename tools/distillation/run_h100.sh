#!/usr/bin/env bash
# Execute inside your own tmux pane with the existing H100 environment active.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec "${PYTHON:-python}" -u -m tools.distillation.launch_h100 "$@"
