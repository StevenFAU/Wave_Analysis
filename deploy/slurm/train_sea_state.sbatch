#!/bin/bash
# Train and evaluate the sea-state classifier on one GPU (docs/operations/cluster.md).
#
#   sbatch deploy/slurm/train_sea_state.sbatch                          # year split, full run
#   sbatch --export=ALL,SPLIT=day deploy/slurm/train_sea_state.sbatch   # held-out days
#   sbatch --export=ALL,LIMIT=64,EPOCHS=1 deploy/slurm/train_sea_state.sbatch   # 5-minute smoke test
#
# Before the first job, replace the four CHANGE_ME values below with what the
# cluster's administrators give you (or pass them on the command line, e.g.
# `sbatch --partition=gpu --account=abc123 ...`, which overrides these lines).
# sbatch refuses the job while a CHANGE_ME is left, which is intended.
#
#SBATCH --job-name=sea-state
#SBATCH --partition=CHANGE_ME           # the GPU partition (question 2)
#SBATCH --account=CHANGE_ME             # your allocation, if the cluster uses accounts (question 2)
#SBATCH --gres=CHANGE_ME                # e.g. gpu:1 or gpu:h200:1, as the cluster names it (questions 2, 6)
#SBATCH --time=CHANGE_ME                # e.g. 02:00:00; a full run takes well under an hour (question 2)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16              # image decoding workers
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out         # relative to the folder you ran sbatch in (the repository)
#SBATCH --error=logs/%x-%j.out

set -euo pipefail

# ---- settings (override with --export=ALL,NAME=value) ----------------------
REPO="${REPO:-$SLURM_SUBMIT_DIR}"                       # this repository on the cluster
PACKAGE="${PACKAGE:-/scratch/$USER/currituck_sea_state_v0}"   # where you copied the package (question 4)
CONFIG="${CONFIG:-configs/experiments/currituck_sea_state_v0.yaml}"
SPLIT="${SPLIT:-year}"                                  # year or day
INPUT="${INPUT:-}"                                      # empty: the config's (sea_crop); or full_frame
BACKBONE="${BACKBONE:-}"                                # empty: the config's (convnext_tiny); or resnet50
EPOCHS="${EPOCHS:-}"
LIMIT="${LIMIT:-}"                                      # stills per partition, for a smoke test
STAGE="${STAGE:-1}"                                     # 1: copy the package to the node's local disk first
RUN="${RUN:-$REPO/runs/${SPLIT}_${INPUT:-sea_crop}_${SLURM_JOB_ID}}"
export TORCH_HOME="${TORCH_HOME:-$REPO/.cache/torch}"   # ImageNet weights, downloaded on the login node
# -----------------------------------------------------------------------------

cd "$REPO"
echo "job $SLURM_JOB_ID on $(hostname) at $(date -u +%FT%TZ); commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Many small files read from a shared file system are slow; the node's own
# disk is not. $TMPDIR is the usual job-local folder (ask if it is something else).
if [[ "$STAGE" == 1 && -n "${TMPDIR:-}" && -d "${TMPDIR:-}" ]]; then
    echo "copying the package to $TMPDIR ..."
    rsync -a "$PACKAGE/" "$TMPDIR/package/"
    PACKAGE="$TMPDIR/package"
fi

ARGS=(--config "$CONFIG" --package "$PACKAGE" --split "$SPLIT" --out "$RUN"
      --num-workers "$((SLURM_CPUS_PER_TASK - 1))")
[[ -n "$INPUT" ]] && ARGS+=(--input "$INPUT")
[[ -n "$BACKBONE" ]] && ARGS+=(--backbone "$BACKBONE")
[[ -n "$EPOCHS" ]] && ARGS+=(--epochs "$EPOCHS")
[[ -n "$LIMIT" ]] && ARGS+=(--limit "$LIMIT")

.venv/bin/python scripts/train_sea_state.py "${ARGS[@]}"
echo "finished at $(date -u +%FT%TZ); results in $RUN"
