#!/bin/bash
# The three runs of the sea-state experiment on the desktop GPU, one after the
# other (docs/operations/cluster.md, "Training on the desktop GPU"). Start it as a
# user service, so it survives closing the terminal or the Claude app:
#
#   systemd-run --user --collect --unit wave-analysis-train -p Nice=5 \
#       --working-directory="$PWD" bash deploy/local/train_sea_state.sh
#
# Precision fp32 (bf16 is five times slower on this card, which has no bf16
# hardware), batches of 64 computed in pieces that fit in 16 GB; the update is
# the full batch's. RUNS=split:input:piece ... picks other runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PACKAGE=${PACKAGE:-data/processed/packages/currituck_sea_state_v0}
CONFIG=configs/experiments/currituck_sea_state_v0.yaml
RUNS=${RUNS:-"year:sea_crop:32 day:sea_crop:32 year:full_frame:8"}
stamp=$(date -u +%Y%m%dT%H%MZ)

for run in $RUNS; do
    IFS=: read -r split input piece <<<"$run"
    out="runs/desktop_${split}_${input}_${stamp}"
    echo "$(date -u +%FT%TZ) start $out"
    .venv-rocm/bin/python scripts/train_sea_state.py --config "$CONFIG" --package "$PACKAGE" \
        --split "$split" --input "$input" --amp none --micro-batch "$piece" --out "$out" \
        2> >(grep -v '^(null): No such file or directory$' >&2)
    echo "$(date -u +%FT%TZ) finished $out"
done
