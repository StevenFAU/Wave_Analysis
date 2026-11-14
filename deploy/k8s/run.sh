#!/usr/bin/env bash
# Train on a Kubernetes cluster with NVIDIA GPUs (FAU's DGX cluster, through
# Rancher), driven from the desktop. docs/operations/cluster.md, "Kubernetes".
#
#   export KUBECONFIG=~/.kube/fau-dgx.yaml NS=<your namespace>     # the VPN must be up
#   deploy/k8s/run.sh check       # who you are, what you may do, the GPU machines
#   deploy/k8s/run.sh up          # the storage volume and the helper pod
#   deploy/k8s/run.sh push        # code (this commit), package, ImageNet weights, uv
#   deploy/k8s/run.sh setup       # the Python environment of uv.lock, on the volume (once per lock)
#   deploy/k8s/run.sh train station --limit 64 --epochs 1   # a smoke test (minutes)
#   deploy/k8s/run.sh train station                         # the run; any train_sea_state.py options after the split
#   deploy/k8s/run.sh status | logs <run> | fetch <run> | down
#
# The cluster runs the committed code: `push` refuses a working tree with
# uncommitted changes, and each run records the commit it ran.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

NS="${NS:?set NS to your namespace, e.g. export NS=cohens2025-wave}"
K=(kubectl -n "$NS")
HERE=deploy/k8s
HELPER=wave-analysis-helper
PKG="${PKG:-ndbc_buoycam_sea_state_v0}"
CONFIG="${CONFIG:-configs/experiments/ndbc_buoycam_sea_state_v0.yaml}"
COMMIT="$(git rev-parse HEAD)"
LOCK="$(sha256sum uv.lock | cut -c1-12)"
ENV_DIR="/data/envs/$LOCK"

h() { "${K[@]}" exec -i "$HELPER" -- "$@"; }    # run a command in the helper pod
say() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }

cmd="${1:-}"
shift || true
case "$cmd" in
check)
    kubectl auth whoami | sed -n '1,2p'
    for r in pods jobs persistentvolumeclaims; do
        say "may create $r in $NS: $("${K[@]}" auth can-i create "$r" 2>&1 | tail -1)"
    done
    kubectl get nodes -l nvidia.com/gpu.product -o \
        custom-columns='NODE:.metadata.name,GPU:.metadata.labels.nvidia\.com/gpu\.product,GPUS:.status.allocatable.nvidia\.com/gpu,TAINTS:.spec.taints[*].key'
    ;;
up)
    "${K[@]}" apply -f "$HERE/storage.yaml"
    "${K[@]}" apply -f "$HERE/helper.yaml"
    say "waiting for the helper pod ..."
    "${K[@]}" wait --for=condition=Ready "pod/$HELPER" --timeout=600s
    ;;
push)
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "commit first: the cluster runs a commit, not a working tree" >&2
        exit 1
    fi
    h mkdir -p /data/repo /data/packages /data/bin /data/cache/torch/hub/checkpoints /data/runs /data/envs
    # 1. the code of this commit, with the commit written beside it (no .git on the cluster)
    if h test -f "/data/repo/$COMMIT/GIT_COMMIT"; then
        say "code $COMMIT already there"
    else
        git archive --format=tar --prefix="$COMMIT/" HEAD | h tar -x -C /data/repo
        echo "$COMMIT" | h sh -c "cat > /data/repo/$COMMIT/GIT_COMMIT"
        say "code $COMMIT copied"
    fi
    # 2. uv, to build the environment from uv.lock
    if ! h test -x /data/bin/uv; then
        h sh -c 'cat > /data/bin/uv.part && chmod 755 /data/bin/uv.part && mv /data/bin/uv.part /data/bin/uv' <"$(command -v uv)"
    fi
    say "uv $(h /data/bin/uv --version | awk '{print $2}')"
    # 3. the ImageNet weights the config's backbone starts from
    for f in "$HOME"/.cache/torch/hub/checkpoints/convnext_tiny-*.pth; do
        dest="/data/cache/torch/hub/checkpoints/$(basename "$f")"
        h test -f "$dest" || h sh -c "cat > $dest.part && mv $dest.part $dest" <"$f"
        say "weights $(basename "$f")"
    done
    # 4. the package, as one tar, checked by its SHA-256 at both ends
    src="data/processed/packages/$PKG"
    tarf="data/interim/k8s/$PKG.tar"
    mkdir -p "$(dirname "$tarf")"
    if [[ ! -f "$tarf" || "$src/SHA256SUMS" -nt "$tarf" ]]; then
        say "packing $src ..."
        tar -C data/processed/packages -cf "$tarf.part" "$PKG" && mv "$tarf.part" "$tarf"
        sha256sum "$tarf" | cut -d' ' -f1 >"$tarf.sha256"
    fi
    want="$(cat "$tarf.sha256")"
    have="$(h sh -c "sha256sum /data/packages/$PKG.tar 2>/dev/null | cut -d' ' -f1" || true)"
    if [[ "$have" == "$want" ]]; then
        say "package $PKG already there"
    else
        say "uploading $PKG ($(du -h "$tarf" | cut -f1)) ..."
        dd if="$tarf" bs=4M status=progress | h sh -c "cat > /data/packages/$PKG.tar.part"
        have="$(h sha256sum "/data/packages/$PKG.tar.part" | cut -d' ' -f1)"
        [[ "$have" == "$want" ]] || { echo "upload damaged: $have != $want" >&2; exit 1; }
        h mv "/data/packages/$PKG.tar.part" "/data/packages/$PKG.tar"
        say "package $PKG uploaded and checked"
    fi
    ;;
setup)
    # Needs internet from the helper pod (PyPI). The environment holds only the
    # libraries of uv.lock (PyTorch with CUDA 13.0, numpy, ...); the code comes
    # from /data/repo/<commit> at run time.
    h python3 -c "import urllib.request as u; u.urlopen('https://pypi.org/simple/', timeout=15)" ||
        { echo "the pod cannot reach PyPI: ask the admins, or build a wheelhouse instead" >&2; exit 1; }
    h env UV_PROJECT_ENVIRONMENT="$ENV_DIR" UV_CACHE_DIR=/data/.uv-cache UV_PYTHON_DOWNLOADS=never \
        UV_LINK_MODE=copy sh -c "cd /data/repo/$COMMIT && /data/bin/uv sync --frozen --no-dev --extra ml \
        --no-install-project --python /usr/local/bin/python3.12"
    h "$ENV_DIR/bin/python" -c "import torch, torchvision; print('environment ready: torch', torch.__version__, 'CUDA', torch.version.cuda)"
    ;;
train)
    split="${1:?which split? station or time}"
    shift
    extra="$*"
    [[ "$split" =~ ^[a-z]+$ ]] || { echo "bad split '$split'" >&2; exit 1; }
    [[ "$extra" != *"|"* ]] || { echo "no '|' in the options" >&2; exit 1; }
    h test -f "/data/repo/$COMMIT/GIT_COMMIT" || { echo "commit $COMMIT is not on the cluster: run.sh push" >&2; exit 1; }
    h test -x "$ENV_DIR/bin/python" || { echo "no environment for this uv.lock: run.sh setup" >&2; exit 1; }
    run="buoy-$split-$(date -u +%Y%m%dt%H%M)"
    [[ "$extra" == *"--class-weights none"* ]] && run="$run-unweighted"
    [[ "$extra" == *"--limit"* ]] && run="$run-smoke"
    sed -e "s|__RUN__|$run|g" -e "s|__SPLIT__|$split|g" -e "s|__COMMIT__|$COMMIT|g" \
        -e "s|__ENV_DIR__|$ENV_DIR|g" -e "s|__PKG__|$PKG|g" -e "s|__CONFIG__|$CONFIG|g" \
        -e "s|__EXTRA__|$extra|g" "$HERE/train-job.yaml" | "${K[@]}" apply -f -
    say "submitted run $run; follow it with: deploy/k8s/run.sh logs $run"
    ;;
status)
    "${K[@]}" get jobs,pods -l app=wave-analysis -o wide
    ;;
logs)
    run="${1:?which run?}"
    "${K[@]}" logs -f "job/wave-$run"
    ;;
fetch)
    run="${1:?which run?}"
    mkdir -p runs
    h tar -C /data/runs -cf - "$run" | tar -x -C runs
    say "results in runs/$run"
    ;;
down)
    "${K[@]}" delete pod "$HELPER" --ignore-not-found
    say "helper removed; the volume and its files stay (delete it with: kubectl -n $NS delete pvc wave-analysis)"
    ;;
*)
    sed -n '2,16p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
