#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$TASK_DIR/src:$TASK_DIR/.act-deps"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$TASK_DIR/.hf-cache"
ACT_PYTHON="${STRAWBERRY_ACT_PYTHON:-$TASK_DIR/../act-so101/.venv/bin/python}"
if [[ ! -x "$ACT_PYTHON" && -z "${STRAWBERRY_ACT_PYTHON:-}" ]]; then
  ACT_PYTHON="$TASK_DIR/../hardware-bringup/.venv-calibration/bin/python"
fi
if [[ ! -x "$ACT_PYTHON" ]]; then
  echo 'ACT Python is missing; set STRAWBERRY_ACT_PYTHON to a compatible interpreter.' >&2
  exit 1
fi
exec "$ACT_PYTHON" "$@"
