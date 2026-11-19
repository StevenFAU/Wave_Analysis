#!/bin/bash
# Runs INSIDE the container started by deploy/dgx/breakers.sh. MODE=features: decode each
# VWBT_9000 clip once and cache frozen features (scripts/breaker_features.py); MODE=heads:
# train and score the heads on that cache (scripts/train_breaker_heads.py fit, then score).
# Variables: RUN, MODE, COMMIT, ENV_DIR, CV2_DIR, SRC, EXTRA_ARGS, NUM_WORKERS.
set -euo pipefail
echo "run $RUN: $MODE, commit $COMMIT, on $(hostname) at $(date -u +%FT%TZ)"
cd "/data/repo/$COMMIT"
export PYTHONPATH="$PWD/src:$PWD/scripts:$CV2_DIR"
"$ENV_DIR/bin/python" - <<'PY'
import torch
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}")
PY
mkdir -p "/data/runs/$RUN"
# EXTRA_ARGS holds zero or more arguments, so it is split on purpose.
# shellcheck disable=SC2086
case "$MODE" in
features)
    "$ENV_DIR/bin/python" -c "import cv2; print('OpenCV', cv2.__version__)"
    "$ENV_DIR/bin/python" scripts/breaker_features.py --source "$SRC" --out "/data/runs/$RUN/features" \
        --device cuda --workers "$NUM_WORKERS" --tmp /scratch $EXTRA_ARGS
    ;;
heads)
    "$ENV_DIR/bin/python" scripts/train_breaker_heads.py fit --source "$SRC" --out "/data/runs/$RUN/heads" \
        --device cuda $EXTRA_ARGS
    "$ENV_DIR/bin/python" scripts/train_breaker_heads.py score --run "/data/runs/$RUN/heads"
    ;;
*)
    echo "MODE must be features or heads" >&2
    exit 1
    ;;
esac
