#!/bin/bash
# Runs INSIDE a container (deploy/dgx/run.sh setup), not on the desktop: builds
# the Python environment of uv.lock into $ENV_DIR on /data, once per lock. The
# environment holds only the libraries (PyTorch with its CUDA 13.0, numpy, ...);
# the code comes from /data/repo/<commit> at run time. Needs internet (PyPI).
# The variables come from the caller: COMMIT, ENV_DIR.
set -euo pipefail
python3 -c "import urllib.request as u; u.urlopen('https://pypi.org/simple/', timeout=15)" ||
    { echo "the container cannot reach PyPI: upload a wheelhouse instead" >&2; exit 1; }
cd "/data/repo/$COMMIT"
UV_PROJECT_ENVIRONMENT="$ENV_DIR" UV_CACHE_DIR=/data/.uv-cache UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy \
    /data/bin/uv sync --frozen --no-dev --extra ml --no-install-project --python /usr/local/bin/python3.12
"$ENV_DIR/bin/python" -c "import torch, torchvision; print('environment ready: torch', torch.__version__, 'CUDA', torch.version.cuda)"
