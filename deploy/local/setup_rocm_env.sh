#!/bin/bash
# Training environment for the desktop's AMD GPU (Radeon RX 6800 class, gfx1030,
# 16 GB), next to the project's own .venv, which stays CPU-only and untouched.
# docs/operations/cluster.md, "Training on the desktop GPU (ROCm)". Run from the
# repository folder:
#
#   bash deploy/local/setup_rocm_env.sh
#
# It installs into .venv-rocm only: the versions of uv.lock for everything but
# torch and torchvision, which come from PyTorch's ROCm index (about 6.5 GB to
# download). No sudo: the GPU is reachable through the logged-in user's access
# to /dev/kfd and /dev/dri/renderD128.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TORCH="2.14.0+rocm7.2"
TORCHVISION="0.29.0+rocm7.2"
INDEX="https://download.pytorch.org/whl/rocm7.2"
ENV=.venv-rocm

python_version=$(.venv/bin/python -c 'import platform; print(platform.python_version())')
[[ -x "$ENV/bin/python" ]] || uv venv --python "$python_version" "$ENV"

# 1. Everything the training script needs, at uv.lock's versions, without
#    torch, torchvision and their CUDA libraries.
reqs=$(mktemp)
trap 'rm -f "$reqs"' EXIT
uv export --frozen --no-dev --extra ml --no-hashes --no-emit-project \
    --no-emit-package torch --no-emit-package torchvision \
    | grep -vE '^(cuda-|nvidia-|triton==)' >"$reqs"
uv pip install --python "$ENV/bin/python" -r "$reqs"

# 2. torch and torchvision built for ROCm (they bring their own ROCm libraries).
uv pip install --python "$ENV/bin/python" --index-url "$INDEX" \
    "torch==$TORCH" "torchvision==$TORCHVISION"

# 3. The project itself, editable, without touching the versions above.
uv pip install --python "$ENV/bin/python" --no-deps -e .

"$ENV/bin/python" - <<'PY'
import torch
print(f"torch {torch.__version__}, HIP {torch.version.hip}")
ok = torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0) if ok else "NOT AVAILABLE")
raise SystemExit(0 if ok else 1)
PY
