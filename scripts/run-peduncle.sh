#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$TASK_DIR/src"
export PYTHONNOUSERSITE=1
exec "$TASK_DIR/../.venv-vista/bin/python" "$TASK_DIR/tools/validation/verify_peduncle.py" "$@"
