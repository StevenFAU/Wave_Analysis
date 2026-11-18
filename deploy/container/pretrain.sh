#!/bin/bash
# Runs INSIDE a container started by deploy/dgx/pretrain.sh, not on the desktop.
# MODE=pretrain: DINOv3's trainer (torchrun, one GPU) on the package's unlabelled views,
#   output in /data/pretrain/$RUN (it resumes from its last checkpoint there if restarted);
# MODE=probe: scripts/dinov3_probe.py over that run's teacher checkpoints.
# /data holds the code, packages, environments and outputs; /scratch is in memory.
# Variables from the caller: RUN, MODE, COMMIT, ENV_DIR, DINOV3_DIR, EXTRA_DIR, PKG,
# CONFIG, NUM_WORKERS, EXTRA_ARGS, and INIT (pretrain).
set -euo pipefail
echo "$MODE $RUN: commit $COMMIT, DINOv3 $(cat "$DINOV3_DIR/.git/HEAD"), on $(hostname) at $(date -u +%FT%TZ)"

mkdir -p /scratch/pkg
tar -xf "/data/packages/$PKG.tar" -C /scratch/pkg
(cd "/scratch/pkg/$PKG" && sha256sum -c --quiet SHA256SUMS)
echo "package $PKG unpacked and intact"

cd "/data/repo/$COMMIT"
# the code of this commit, DINOv3's checkout and its extra libraries; torch etc. from the environment
export PYTHONPATH="$PWD/src:$DINOV3_DIR:$EXTRA_DIR"
# torch.compile's kernels: not on /scratch, which Docker mounts noexec (Triton loads them as libraries)
export TORCHINDUCTOR_CACHE_DIR=/data/cache/inductor TRITON_CACHE_DIR=/data/cache/triton
"$ENV_DIR/bin/python" - <<'PY'
import torch
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}, "
      f"{torch.cuda.mem_get_info()[0] / 1e9:.0f} of {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB free")
PY

# EXTRA_ARGS holds zero or more arguments, so it is split on purpose.
# shellcheck disable=SC2086
if [[ "$MODE" == pretrain ]]; then
    exec "$ENV_DIR/bin/python" -m torch.distributed.run --standalone --nproc_per_node=1 \
        scripts/dinov3_pretrain.py --config-file "$CONFIG" --output-dir "/data/pretrain/$RUN" \
        "train.dataset_path=BuoyViews:root=/scratch/pkg/$PKG:split=train" \
        "student.resume_from_teacher_chkpt=$INIT" "train.num_workers=$NUM_WORKERS" $EXTRA_ARGS
else
    exec "$ENV_DIR/bin/python" scripts/dinov3_probe.py --package "/scratch/pkg/$PKG" --config "$CONFIG" \
        --run-dir "/data/pretrain/$RUN" --out "/data/pretrain/$RUN/probe" --save-timm \
        --num-workers "$NUM_WORKERS" $EXTRA_ARGS
fi
