#!/bin/bash
# Runs INSIDE the container started by deploy/dgx/measure.sh: frozen buoy probes, one
# fold after another. /data holds the code, package, environment, checkpoints and runs;
# /scratch is in memory. Variables: RUN, COMMIT, ENV_DIR, PKG, JOBS (<fold>:<checkpoint
# run> ...), EXTRA_ARGS, NUM_WORKERS.
set -euo pipefail
echo "run $RUN: probes $JOBS, commit $COMMIT, on $(hostname) at $(date -u +%FT%TZ)"

mkdir -p /scratch/pkg
tar -xf "/data/packages/$PKG.tar" -C /scratch/pkg
(cd "/scratch/pkg/$PKG" && sha256sum -c --quiet SHA256SUMS)
echo "package $PKG unpacked and intact"

cd "/data/repo/$COMMIT"
export PYTHONPATH="$PWD/src"
"$ENV_DIR/bin/python" - <<'PY'
import torch
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}")
PY

mkdir -p "/data/runs/$RUN"
for job in $JOBS; do
    fold="${job%%:*}"
    checkpoint="${job#*:}"
    start=$(date +%s)
    echo "fold $fold: checkpoint $checkpoint, start $(date -u +%FT%TZ)"
    # EXTRA_ARGS holds zero or more arguments, so it is split on purpose.
    # shellcheck disable=SC2086
    "$ENV_DIR/bin/python" scripts/measurement_buoy_probe.py \
        --package "/scratch/pkg/$PKG" --fold "$fold" --checkpoint-run "/data/runs/$checkpoint" \
        --device cuda --batch 32 --workers "$NUM_WORKERS" --out "/data/runs/$RUN/fold$fold" $EXTRA_ARGS
    echo "fold $fold done in $(($(date +%s) - start)) s"
done
