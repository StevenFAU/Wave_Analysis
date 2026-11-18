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
#                               # (BASE_PKG=<package on the server>: reuse its images)
#   deploy/dgx/run.sh setup     # the Python environment of uv.lock (once per lock)
#   deploy/dgx/run.sh train station --limit 64 --epochs 1   # a smoke test (minutes)
#   deploy/dgx/run.sh train station                         # the run; train_sea_state.py options after the split
#   deploy/dgx/run.sh sync      # a package that grows (video frames): only new files travel
#   deploy/dgx/run.sh status | logs <run> [-f] | fetch <run> | stop <run>
#
# One GPU per run: GPU=<index> picks it, otherwise the least busy GPU with at
# least MIN_FREE_GB (default 40) of free memory. TRAIN_SCRIPT picks the trainer
# (default scripts/train_sea_state.py), TAG adds a word to the run's name and KIND
# (default buoy; video for the shore cameras) starts it. The server is shared: this
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
run_name() { [[ "${1:-}" =~ ^(buoy|video)-[a-z0-9-]+$ ]] || { echo "which run? (see: run.sh status)" >&2; exit 1; }; }

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
    # Data manifests (download logs) may change while a download runs; code may not.
    if ! git diff --quiet -- . ":(exclude)data/manifests" || ! git diff --cached --quiet -- . ":(exclude)data/manifests"; then
        echo "commit first: the server runs a commit, not a working tree" >&2
        exit 1
    fi
    # Uploads can be slow (about 2 Mbit/s through FAU's bastion on 2026-11-17) and
    # break: rsync resumes a broken transfer (--partial-dir) and gives a file its
    # final name only once it is complete. Note that stopping this script does not
    # stop a transfer already handed to the shared login (it holds the open file).
    put() { rsync -t --partial-dir=.rsync-partial --info=progress2 -e "ssh -T -o BatchMode=yes" "$@"; }
    r mkdir -p "$BASE"/{repo,packages,bin,cache/torch/hub/checkpoints,runs,logs,envs}
    # 1. the code of this commit (about 5 MB), unpacked beside its final place and
    #    renamed with the commit written in it (no .git on the server)
    if r test -f "$BASE/repo/$COMMIT/GIT_COMMIT"; then
        say "code $COMMIT already there"
    else
        r rm -rf "$BASE/repo/$COMMIT.part"
        git archive --format=tar --prefix="$COMMIT.part/" HEAD | r tar -x -C "$BASE/repo"
        r "echo $COMMIT > $BASE/repo/$COMMIT.part/GIT_COMMIT && mv $BASE/repo/$COMMIT.part $BASE/repo/$COMMIT"
        say "code $COMMIT copied"
    fi
    # 2. uv, to build the environment from uv.lock
    put --chmod=F755 "$(command -v uv)" "$GPU_HOST:$BASE/bin/uv"
    say "uv $(r "$BASE/bin/uv" --version | awk '{print $2}')"
    # 3. the ImageNet weights the config's backbone starts from
    put "$HOME"/.cache/torch/hub/checkpoints/convnext_tiny-*.pth "$GPU_HOST:$BASE/cache/torch/hub/checkpoints/"
    say "weights there"
    # 4. the package, as one tar. Its content is named by the SHA-256 of its
    #    SHA256SUMS file (every file's checksum), recorded beside the tar on the
    #    server once that tar has been checked, so a later push skips it.
    src="data/processed/packages/$PKG"
    tarf="data/interim/k8s/$PKG.tar"
    mkdir -p "$(dirname "$tarf")"
    content="$(sha256sum "$src/SHA256SUMS" | cut -d' ' -f1)"
    if [[ "$(r cat "$BASE/packages/$PKG.content" 2>/dev/null || true)" == "$content" ]]; then
        say "package $PKG already there (content $content)"
    elif [[ ! -f "$tarf" || "$src/SHA256SUMS" -nt "$tarf" ]] && ! {
        say "packing $src ..."
        tar -C data/processed/packages -cf "$tarf.part" "$PKG" && mv "$tarf.part" "$tarf" &&
            sha256sum "$tarf" | cut -d' ' -f1 >"$tarf.sha256"
    }; then
        echo "packing $src failed" >&2
        exit 1
    elif [[ -n "${BASE_PKG:-}" ]] && r test -f "$BASE/packages/$BASE_PKG.tar" &&
        python3 - "$src/SHA256SUMS" "data/processed/packages/$BASE_PKG/SHA256SUMS" <<'PY'
import sys
new = [line.split(None, 1) for line in open(sys.argv[1]) if " images/" in line]
old = {path.strip(): sha for sha, path in (line.split(None, 1) for line in open(sys.argv[2]))}
missing = [p for sha, p in new if old.get(p.strip()) != sha]
sys.exit(1 if missing or not new else 0)
PY
    then
        # Every image of PKG is in BASE_PKG, byte for byte: send only the other
        # files and build the package on the server from BASE_PKG's images.
        say "package $PKG: its images are those of $BASE_PKG; sending the other files only"
        tar -C "$src" --exclude=./images -cf - . |
            r "rm -rf $BASE/packages/.build-$PKG && mkdir -p $BASE/packages/.build-$PKG/$PKG && tar -x -C $BASE/packages/.build-$PKG/$PKG"
        r bash -s <<EOF
set -euo pipefail
cd $BASE/packages/.build-$PKG
tar -xf $BASE/packages/$BASE_PKG.tar $BASE_PKG/images
cd $BASE_PKG && grep ' images/' ../$PKG/SHA256SUMS | cut -c67- | tar -cf - -T - | tar -x -C ../$PKG
cd ../$PKG && sha256sum -c --quiet SHA256SUMS
cd .. && tar -cf $BASE/packages/$PKG.tar.part $PKG && mv $BASE/packages/$PKG.tar.part $BASE/packages/$PKG.tar
echo $content > $BASE/packages/$PKG.content
cd $BASE/packages && rm -rf .build-$PKG
EOF
    else
        say "package $PKG ($(du -h "$tarf" | cut -f1)) ..."
        put "$tarf" "$GPU_HOST:$BASE/packages/$PKG.tar"
        want="$(cat "$tarf.sha256")"
        have="$(r sha256sum "$BASE/packages/$PKG.tar" | cut -d' ' -f1)"
        if [[ "$have" != "$want" ]]; then
            r rm -f "$BASE/packages/$PKG.tar"
            echo "package damaged on the server ($have != $want): removed; run push again" >&2
            exit 1
        fi
        echo "$content" | r "cat > $BASE/packages/$PKG.content"
    fi
    say "package $PKG there and checked"
    ;;
sync)
    # A package that grows while its source downloads (the WebCOOS video frames):
    # rsync its folder, so only new files travel, then tar and check it on the server
    # and record its content, so that push finds it there.
    src="data/processed/packages/$PKG"
    [[ -f "$src/SHA256SUMS" ]] || { echo "no $src/SHA256SUMS: build the package first" >&2; exit 1; }
    content="$(sha256sum "$src/SHA256SUMS" | cut -d' ' -f1)"
    r mkdir -p "$BASE/packages/.src"
    rsync -rt --delete --partial-dir=.rsync-partial --info=progress2 -e "ssh -T -o BatchMode=yes" \
        "$src/" "$GPU_HOST:$BASE/packages/.src/$PKG/"
    r bash -s <<EOF
set -euo pipefail
cd $BASE/packages/.src/$PKG && sha256sum -c --quiet SHA256SUMS
cd .. && tar -cf $BASE/packages/$PKG.tar.part $PKG && mv $BASE/packages/$PKG.tar.part $BASE/packages/$PKG.tar
echo $content > $BASE/packages/$PKG.content
EOF
    say "package $PKG synced and checked (content $content)"
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
    [[ "$split" =~ ^[a-z][a-z0-9]*$ ]] || { echo "bad split '$split'" >&2; exit 1; }
    [[ "$extra" =~ ^[A-Za-z0-9\ ._=/:-]*$ ]] || { echo "options may hold only letters, digits, spaces and ._=/:-" >&2; exit 1; }
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
    [[ "${KIND:-buoy}" =~ ^(buoy|video)$ ]] || { echo "KIND is buoy or video" >&2; exit 1; }
    run="${KIND:-buoy}-$split-$(date -u +%Y%m%dt%H%M)"
    [[ "$extra" == *"--class-weights none"* ]] && run="$run-unweighted"
    [[ "$extra" =~ --seed\ ([0-9]+) ]] && run="$run-seed${BASH_REMATCH[1]}"
    [[ "${TAG:-}" =~ ^[a-z0-9-]*$ ]] || { echo "TAG may hold only a-z, 0-9 and -" >&2; exit 1; }
    [[ "${TRAIN_SCRIPT:-scripts/train_sea_state.py}" =~ ^scripts/[a-z0-9_]+\.py$ ]] || { echo "bad TRAIN_SCRIPT" >&2; exit 1; }
    [[ -n "${TAG:-}" ]] && run="$run-$TAG"
    [[ "$extra" == *"--limit"* ]] && run="$run-smoke"
    r "docker ps -a --format '{{.Names}}' | grep -qx wave-$run" && { echo "run $run is already running; wait a minute" >&2; exit 1; }
    # Detached, so it runs on after this connection closes. /scratch (memory)
    # takes the unpacked package; the log goes to $BASE/logs/<run>.log.
    # TORCH_DISABLE_NATIVE_JIT: PyTorch 2.14 sends some products (a one-query
    # attention's backward) to Triton kernels that need a C compiler, which the slim
    # image lacks; off, they use the ordinary CUDA kernels.
    r bash -s <<EOF
set -e
docker image inspect $IMAGE >/dev/null 2>&1 || docker pull -q $IMAGE >/dev/null
docker run -d --rm --name wave-$run --label app=wave-analysis --label run=$run \
    --gpus device=$gpu --cpus 24 --memory 96g --shm-size 32g \
    --mount type=tmpfs,destination=/scratch,tmpfs-size=8g \
    --user \$(id -u):\$(id -g) -e HOME=/scratch -v $BASE:/data \
    -e RUN=$run -e SPLIT=$split -e COMMIT=$COMMIT -e ENV_DIR=/data/envs/$LOCK -e PKG=$PKG \
    -e CONFIG=$CONFIG -e "EXTRA_ARGS=$extra" -e NUM_WORKERS=22 -e TORCH_HOME=/data/cache/torch -e HF_HOME=/data/cache/hf \
    -e PYTHONUNBUFFERED=1 -e TORCH_DISABLE_NATIVE_JIT=1 -e TRAIN_SCRIPT=${TRAIN_SCRIPT:-scripts/train_sea_state.py} \
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
