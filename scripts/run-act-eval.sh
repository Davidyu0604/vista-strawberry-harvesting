#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$TASK_DIR/src"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2
exec "$TASK_DIR/../.venv-vista/bin/python" -u "$TASK_DIR/src/evaluate_act.py" "$@"
