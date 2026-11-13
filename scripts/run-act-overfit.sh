#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$TASK_DIR/src:$TASK_DIR/.act-deps"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$TASK_DIR/.hf-cache" TORCH_HOME="$TASK_DIR/.torch-cache"
export OMP_NUM_THREADS=4
exec "$TASK_DIR/../.venv-act-train/bin/python" -u "$TASK_DIR/src/train_act_overfit.py" "$@"
