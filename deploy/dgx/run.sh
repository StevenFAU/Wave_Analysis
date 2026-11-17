#!/usr/bin/env bash
# Train on a shared GPU server over SSH, in a Docker container: the way FAU's
# CAAI lab uses its H200 server (docs/operations/cluster.md, "GPU server over
# SSH"). Driven from the desktop; each run trains detached on the server.
#
#   ssh <host>                  # log in once (passwords) and leave it open: later
#                               # connections reuse it (ControlMaster, ~/.ssh/config)
#   export GPU_HOST=<host>      # the server name in ~/.ssh/config
#   deploy/dgx/run.sh check     # Docker, free disk, how busy each GPU is
#   deploy/dgx/run.sh push      # code (this commit), package, ImageNet weights, uv
#   deploy/dgx/run.sh setup     # the Python environment of uv.lock (once per lock)
#   deploy/dgx/run.sh train station --limit 64 --epochs 1   # a smoke test (minutes)
#   deploy/dgx/run.sh train station                         # the run; train_sea_state.py options after the split
#   deploy/dgx/run.sh status | logs <run> [-f] | fetch <run> | stop <run>
#
# One GPU per run: GPU=<index> picks it, otherwise the least busy GPU with at
# least MIN_FREE_GB (default 40) of free memory. The server is shared: this
# script only touches its own containers (label app=wave-analysis) and BASE.
# The server runs the committed code: `push` refuses a working tree with
# uncommitted changes, and each run records the commit it ran.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GPU_HOST="${GPU_HOST:?set GPU_HOST to the GPU server name in ~/.ssh/config}"
IMAGE=python:3.12.16-slim-trixie   # the environment's Python; PyTorch's wheels bring CUDA 13.0
PKG="${PKG:-ndbc_buoycam_sea_state_v0}"
CONFIG="${CONFIG:-configs/experiments/ndbc_buoycam_sea_state_v0.yaml}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"
COMMIT="$(git rev-parse HEAD)"
LOCK="$(sha256sum uv.lock | cut -c1-12)"

r() { ssh -T -o BatchMode=yes "$GPU_HOST" "$@"; }    # a command on the server, through the open login
say() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
r true 2>/dev/null || { echo "no open login to $GPU_HOST: run 'ssh $GPU_HOST' in a terminal and leave it open" >&2; exit 1; }
BASE="${BASE:-/raid/scratch/$(r id -un)/wave-analysis}"   # on the server; /data inside the containers

gpu_table() {  # per GPU over 10 s: index, mean busy %, free GB
    r 'timeout 10 nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -l 1 || true' |
        awk -F', *' '{u[$1] += $2; f[$1] = ($4 - $3) / 1024; n[$1]++}
            END {for (g in n) printf "%s %.0f %.0f\n", g, u[g] / n[g], f[g]}' | sort -n
}
run_name() { [[ "${1:-}" =~ ^buoy-[a-z0-9-]+$ ]] || { echo "which run? (see: run.sh status)" >&2; exit 1; }; }

cmd="${1:-}"
shift || true
case "$cmd" in
check)
    r bash -s <<EOF
echo "server \$(hostname), user \$(id -un)"
id -nG | tr ' ' '\n' | grep -qx docker && echo "docker group: yes" || echo "docker group: NO (ask the administrators)"
echo "docker \$(docker version --format '{{.Server.Version}}'), runtimes: \$(docker info --format '{{range \$k, \$v := .Runtimes}}{{\$k}} {{end}}')"
mkdir -p $BASE && df -h --output=avail,pcent $BASE | tail -1 | awk '{print "free where BASE is: " \$1 " (disk " \$2 " full)"}'
echo "our containers running: \$(docker ps -q --filter label=app=wave-analysis | wc -l)"
EOF
    say "GPUs over 10 s (index, busy %, free GB):"
    gpu_table
    ;;
push)
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "commit first: the server runs a commit, not a working tree" >&2
        exit 1
    fi
    r mkdir -p "$BASE"/{repo,packages,bin,cache/torch/hub/checkpoints,runs,logs,envs}
    # 1. the code of this commit, with the commit written beside it (no .git on the server)
    if r test -f "$BASE/repo/$COMMIT/GIT_COMMIT"; then
        say "code $COMMIT already there"
    else
        git archive --format=tar --prefix="$COMMIT/" HEAD | r tar -x -C "$BASE/repo"
        echo "$COMMIT" | r "cat > $BASE/repo/$COMMIT/GIT_COMMIT"
        say "code $COMMIT copied"
    fi
    # 2. uv, to build the environment from uv.lock
    if ! r test -x "$BASE/bin/uv"; then
        r "cat > $BASE/bin/uv.part && chmod 755 $BASE/bin/uv.part && mv $BASE/bin/uv.part $BASE/bin/uv" <"$(command -v uv)"
    fi
    say "uv $(r "$BASE/bin/uv" --version | awk '{print $2}')"
    # 3. the ImageNet weights the config's backbone starts from
    for f in "$HOME"/.cache/torch/hub/checkpoints/convnext_tiny-*.pth; do
        dest="$BASE/cache/torch/hub/checkpoints/$(basename "$f")"
        r test -f "$dest" || r "cat > $dest.part && mv $dest.part $dest" <"$f"
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
    have="$(r "sha256sum $BASE/packages/$PKG.tar 2>/dev/null | cut -d' ' -f1" || true)"
    if [[ "$have" == "$want" ]]; then
        say "package $PKG already there"
    else
        say "uploading $PKG ($(du -h "$tarf" | cut -f1)) ..."
        dd if="$tarf" bs=4M status=progress | r "cat > $BASE/packages/$PKG.tar.part"
        have="$(r sha256sum "$BASE/packages/$PKG.tar.part" | cut -d' ' -f1)"
        [[ "$have" == "$want" ]] || { echo "upload damaged: $have != $want" >&2; exit 1; }
        r mv "$BASE/packages/$PKG.tar.part" "$BASE/packages/$PKG.tar"
        say "package $PKG uploaded and checked"
    fi
    ;;
setup)
    r test -f "$BASE/repo/$COMMIT/GIT_COMMIT" || { echo "commit $COMMIT is not on the server: run.sh push" >&2; exit 1; }
    r bash -s <<EOF
set -e
docker image inspect $IMAGE >/dev/null 2>&1 || docker pull -q $IMAGE
docker run --rm --user \$(id -u):\$(id -g) -e HOME=/tmp -v $BASE:/data \
    -e COMMIT=$COMMIT -e ENV_DIR=/data/envs/$LOCK \
    $IMAGE bash /data/repo/$COMMIT/deploy/container/setup.sh
EOF
    ;;
train)
    split="${1:?which split? station or time}"
    shift
    extra="$*"
    [[ "$split" =~ ^[a-z]+$ ]] || { echo "bad split '$split'" >&2; exit 1; }
    [[ "$extra" =~ ^[A-Za-z0-9\ ._=/-]*$ ]] || { echo "options may hold only letters, digits, spaces and ._=/-" >&2; exit 1; }
    r test -f "$BASE/repo/$COMMIT/GIT_COMMIT" || { echo "commit $COMMIT is not on the server: run.sh push" >&2; exit 1; }
    r test -f "$BASE/envs/$LOCK/pyvenv.cfg" || { echo "no environment for this uv.lock: run.sh setup" >&2; exit 1; }
    if [[ -n "${GPU:-}" ]]; then
        gpu="$GPU"
    else
        table="$(gpu_table)"
        gpu="$(awk -v min="$MIN_FREE_GB" '$3 >= min' <<<"$table" | sort -k2,2n | head -1 | cut -d' ' -f1)"
        [[ -n "$gpu" ]] || { echo "no GPU has $MIN_FREE_GB GB free now (index, busy %, free GB):" >&2; echo "$table" >&2; exit 1; }
        say "GPU $gpu: $(awk -v g="$gpu" '$1 == g {print $2 "% busy, " $3 " GB free"}' <<<"$table") over the last 10 s"
    fi
    [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "bad GPU '$gpu'" >&2; exit 1; }
    run="buoy-$split-$(date -u +%Y%m%dt%H%M)"
    [[ "$extra" == *"--class-weights none"* ]] && run="$run-unweighted"
    [[ "$extra" == *"--limit"* ]] && run="$run-smoke"
    # Detached, so it runs on after this connection closes. /scratch (memory)
    # takes the unpacked package; the log goes to $BASE/logs/<run>.log.
    r bash -s <<EOF
set -e
docker image inspect $IMAGE >/dev/null 2>&1 || docker pull -q $IMAGE >/dev/null
docker run -d --rm --name wave-$run --label app=wave-analysis --label run=$run \
    --gpus device=$gpu --cpus 24 --memory 96g --shm-size 32g \
    --mount type=tmpfs,destination=/scratch,tmpfs-size=8g \
    --user \$(id -u):\$(id -g) -e HOME=/scratch -v $BASE:/data \
    -e RUN=$run -e SPLIT=$split -e COMMIT=$COMMIT -e ENV_DIR=/data/envs/$LOCK -e PKG=$PKG \
    -e CONFIG=$CONFIG -e "EXTRA_ARGS=$extra" -e NUM_WORKERS=22 -e TORCH_HOME=/data/cache/torch \
    -e PYTHONUNBUFFERED=1 \
    $IMAGE bash -c 'bash /data/repo/$COMMIT/deploy/container/train.sh >/data/logs/$run.log 2>&1; s=\$?;
        echo "finished with exit code \$s at \$(date -u +%FT%TZ)" >>/data/logs/$run.log; exit \$s' >/dev/null
EOF
    say "started run $run on GPU $gpu; follow it with: deploy/dgx/run.sh logs $run -f"
    ;;
status)
    r bash -s <<EOF
docker ps --filter label=app=wave-analysis --format 'running: {{.Names}} ({{.Status}})'
cd $BASE/logs 2>/dev/null || exit 0
for f in \$(ls -t *.log 2>/dev/null | head -8); do echo "\${f%.log}: \$(tail -1 \$f | cut -c1-160)"; done
EOF
    ;;
logs)
    run_name "${1:-}"
    r tail -n 40 ${2:+-f} "$BASE/logs/$1.log"
    ;;
fetch)
    run_name "${1:-}"
    mkdir -p runs
    r tar -C "$BASE/runs" -cf - "$1" | tar -x -C runs
    r cat "$BASE/logs/$1.log" >"runs/$1/train.log"
    say "results in runs/$1"
    ;;
stop)
    run_name "${1:-}"
    r docker stop "wave-$1"
    ;;
*)
    sed -n '2,20p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
