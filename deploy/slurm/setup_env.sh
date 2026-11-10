#!/bin/bash
# One-time setup of the training environment on a cluster's LOGIN node
# (docs/operations/cluster.md, step 3). Run from the repository folder:
#
#   bash deploy/slurm/setup_env.sh
#
# It needs internet on the login node (question 5). It installs nothing outside
# this folder and ~/.local/bin (uv). Training itself never runs here.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# 1. uv, the package manager this repository is locked with (one file in ~/.local/bin).
if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/uv" ]]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# 2. The exact versions of uv.lock, with the ml extra: torch and torchvision as
#    published on PyPI (the Linux wheels include CUDA). No development tools.
uv sync --frozen --no-dev --extra ml

# 3. The ImageNet weights, downloaded once into .cache/torch: compute nodes
#    often have no internet.
export TORCH_HOME="$PWD/.cache/torch"
.venv/bin/python - <<'PY'
import os, torch, torchvision.models as m
m.convnext_tiny(weights="DEFAULT"); m.resnet50(weights="DEFAULT")
print(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}; weights in {os.environ['TORCH_HOME']}")
PY
mkdir -p logs runs
echo "Environment ready. On a GPU node, 'nvidia-smi' must show a 'CUDA Version' at least the one above."
