#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ACT_TRAIN_PYTHON="$TASK_DIR/../.venv-act-train/bin/python"
export PYTHONPATH="$TASK_DIR/.act-deps"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$TASK_DIR/.hf-cache"
export TORCH_HOME="$TASK_DIR/.torch-cache"
export OMP_NUM_THREADS=4
if [[ ! -x "$ACT_TRAIN_PYTHON" ]]; then
  echo 'Missing .venv-act-train. See reports/04_训练验证/fresh60-training-v1/README.md.' >&2
  exit 1
fi
exec "$ACT_TRAIN_PYTHON" "$TASK_DIR/train_act.py" "$@"
