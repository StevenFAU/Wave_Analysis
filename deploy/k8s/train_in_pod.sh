#!/bin/bash
# Runs INSIDE the training pod (deploy/k8s/train-job.yaml), not on the desktop.
# The variables come from the job: RUN, SPLIT, COMMIT, ENV_DIR, PKG, CONFIG,
# EXTRA_ARGS, NUM_WORKERS, TORCH_HOME.
set -euo pipefail
echo "run $RUN: split $SPLIT, commit $COMMIT, on $(hostname) at $(date -u +%FT%TZ)"

# Many small files on network storage are slow to read; memory is not. The
# package is one tar on the volume, unpacked into /scratch (memory) and checked.
mkdir -p /scratch/pkg
tar -xf "/data/packages/$PKG.tar" -C /scratch/pkg
(cd "/scratch/pkg/$PKG" && sha256sum -c --quiet SHA256SUMS)
echo "package $PKG unpacked and intact"

cd "/data/repo/$COMMIT"
export PYTHONPATH="$PWD/src"           # the code of this commit; the environment holds only libraries
"$ENV_DIR/bin/python" - <<'PY'
import torch
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}, "
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
PY

# EXTRA_ARGS holds zero or more arguments, so it is split on purpose.
# shellcheck disable=SC2086
exec "$ENV_DIR/bin/python" scripts/train_sea_state.py \
    --config "$CONFIG" --package "/scratch/pkg/$PKG" --split "$SPLIT" \
    --amp bf16 --num-workers "$NUM_WORKERS" --out "/data/runs/$RUN" $EXTRA_ARGS
