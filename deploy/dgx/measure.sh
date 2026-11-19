#!/usr/bin/env bash
# Frozen buoy probes (scripts/measurement_buoy_probe.py) on the shared GPU server, the
# way deploy/dgx/run.sh trains: the server runs a pushed commit, the package is a tar on
# the server, and each call starts one detached container labelled app=wave-analysis.
# The probe takes its own flags, not the trainer's --config and --split, hence this
# launcher beside run.sh (which stays as it is).
#
#   export GPU_HOST=<ssh-alias>                 # the server name in ~/.ssh/config
#   PKG=ndbc_buoycam_slope_v1 BASE_PKG=ndbc_buoycam_heights_v2cv deploy/dgx/run.sh push
#                                               # code + package (images reused from v2cv)
#   GPU=<index> deploy/dgx/measure.sh probe <tag> <fold>:<checkpoint run> [...]
#                                               # the folds run one after another
#   EXTRA="--water-only" GPU=<index> deploy/dgx/measure.sh probe ...
#   deploy/dgx/run.sh status | logs <run> [-f] | fetch <run> | stop <run>
#
# GPU must be given: check free memory first (run.sh check). The checkpoint runs are
# the server's own (BASE/runs); the probe checks each one's fold, test strips and hash.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GPU_HOST="${GPU_HOST:?set GPU_HOST to the GPU server name in ~/.ssh/config}"
IMAGE=python:3.12.16-slim-trixie
PKG="${PKG:-ndbc_buoycam_slope_v1}"
COMMIT="$(git rev-parse HEAD)"
LOCK="$(sha256sum uv.lock | cut -c1-12)"

r() { ssh -T -o BatchMode=yes "$GPU_HOST" "$@"; }
say() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
r true 2>/dev/null || { echo "no open login to $GPU_HOST: run 'ssh $GPU_HOST' in a terminal and leave it open" >&2; exit 1; }
BASE="${BASE:-/raid/scratch/$(r id -un)/wave-analysis}"

cmd="${1:-}"
shift || true
case "$cmd" in
probe)
    tag="${1:?a tag for the run name, e.g. dinov3-s0-fold0}"
    shift
    (($#)) || { echo "give at least one <fold>:<checkpoint run>" >&2; exit 1; }
    [[ "$tag" =~ ^[a-z0-9-]+$ ]] || { echo "tag: a-z, 0-9 and - only" >&2; exit 1; }
    for job in "$@"; do
        [[ "$job" =~ ^([0-4]):(buoy-fold([0-4])-[a-z0-9-]+)$ ]] && [[ "${BASH_REMATCH[1]}" == "${BASH_REMATCH[3]}" ]] ||
            { echo "bad job '$job': <fold>:buoy-fold<fold>-..." >&2; exit 1; }
        r test -f "$BASE/runs/${job#*:}/best.pt" || { echo "no checkpoint $BASE/runs/${job#*:}" >&2; exit 1; }
    done
    extra="${EXTRA:-}"
    [[ "$extra" =~ ^[A-Za-z0-9\ ._=/:-]*$ ]] || { echo "EXTRA may hold only letters, digits, spaces and ._=/:-" >&2; exit 1; }
    gpu="${GPU:?set GPU to a GPU with enough free memory (run.sh check)}"
    [[ "$gpu" =~ ^[0-7]$ ]] || { echo "bad GPU '$gpu'" >&2; exit 1; }
    r test -f "$BASE/repo/$COMMIT/GIT_COMMIT" || { echo "commit $COMMIT is not on the server: run.sh push" >&2; exit 1; }
    r test -f "$BASE/envs/$LOCK/pyvenv.cfg" || { echo "no environment for this uv.lock: run.sh setup" >&2; exit 1; }
    r test -f "$BASE/packages/$PKG.tar" || { echo "no package $PKG on the server: run.sh push" >&2; exit 1; }
    run="buoy-probe-$(date -u +%Y%m%dt%H%M)-$tag"
    r "docker ps -a --format '{{.Names}}' | grep -qx wave-$run" && { echo "run $run exists; wait a minute" >&2; exit 1; }
    r bash -s <<EOF
set -e
docker run -d --rm --name wave-$run --label app=wave-analysis --label run=$run \
    --gpus device=$gpu --cpus 16 --memory 64g --shm-size 16g \
    --mount type=tmpfs,destination=/scratch,tmpfs-size=8g \
    --user \$(id -u):\$(id -g) -e HOME=/scratch -v $BASE:/data \
    -e RUN=$run -e COMMIT=$COMMIT -e ENV_DIR=/data/envs/$LOCK -e PKG=$PKG -e "JOBS=$*" \
    -e "EXTRA_ARGS=$extra" -e NUM_WORKERS=12 -e TORCH_HOME=/data/cache/torch -e HF_HOME=/data/cache/hf \
    -e PYTHONUNBUFFERED=1 -e TORCH_DISABLE_NATIVE_JIT=1 \
    $IMAGE bash -c 'bash /data/repo/$COMMIT/deploy/container/measure.sh >/data/logs/$run.log 2>&1; s=\$?;
        echo "finished with exit code \$s at \$(date -u +%FT%TZ)" >>/data/logs/$run.log; exit \$s' >/dev/null
EOF
    say "started $run on GPU $gpu ($*); follow it with: deploy/dgx/run.sh logs $run -f"
    ;;
*)
    sed -n '2,18p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
