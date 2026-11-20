#!/usr/bin/env bash
# Continued self-supervised pretraining of DINOv3 on the lab GPU server, beside run.sh
# (docs/experiments/domain_pretraining.md). DINOv3's own launcher (dinov3.run.submit)
# assumes Slurm, which the server lacks: this runs its trainer with torchrun on one GPU
# in a Docker container, the way run.sh trains. Same login, BASE, code and packages.
#
#   export GPU_HOST=<host>               # as for run.sh, with the login open
#   deploy/dgx/run.sh push               # this commit's code (and the package) on the server
#   deploy/dgx/pretrain.sh setup         # DINOv3 at the pinned commit, its extra libraries, the image
#   deploy/dgx/pretrain.sh weights       # timm's public ViT-B/16 -> the trainer's start checkpoint, checked
#   GPU=<i> deploy/dgx/pretrain.sh start smoke      # about 15 min: short stages, frequent checkpoints
#   GPU=<i> deploy/dgx/pretrain.sh start full       # the run, detached; trainer overrides may follow
#   GPU=<i> RUN=<run> deploy/dgx/pretrain.sh start full   # restart a stopped run: it resumes
#   GPU=<i> deploy/dgx/pretrain.sh probe <run> [probe options]   # validation ridge probe (scripts/dinov3_probe.py)
#   deploy/dgx/pretrain.sh status | logs <run> [-f] | stop <run> | fetch <run>
#
# GPU=<index> is required (the server is shared; check run.sh check first). CPUS sets
# the container's CPUs (default 32; DINOv3's ten crops per view are made on the CPU),
# MEM the pretraining container's memory limit (default 160g; the first full run was
# killed at it after 10,120 steps, so the restart has more and glibc's malloc arenas
# capped, a usual cause of data-loader workers that grow).
# The image is the full python:3.12.16 (not run.sh's slim one): torch.compile's Triton
# kernels need a C compiler, and the Python is the same, so the uv.lock environment
# built by run.sh setup runs in it. Only this project's containers (label
# app=wave-analysis) and BASE are touched.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GPU_HOST="${GPU_HOST:?set GPU_HOST to the GPU server name in ~/.ssh/config}"
IMAGE=python:3.12.16-trixie
DINOV3_REPO=https://github.com/facebookresearch/dinov3.git
DINOV3_COMMIT=81aae9ad87be9a5e19f5043ac38957449af44b15   # branch FINO, 2026-06-12 (metadata-guided recipes)
CONFIG=configs/pretraining/dinov3_vitb16_buoycam_fino.yaml
PKG="${PKG:-ndbc_buoycam_sea_state_v0}"          # pretraining images (labels are not read)
PROBE_PKG="${PROBE_PKG:-ndbc_buoycam_heights_v2}" # the same images with height labels, for the probe
PROBE_CONFIG=configs/experiments/ndbc_buoycam_heights_v2.yaml
CPUS="${CPUS:-32}"
MEM="${MEM:-160g}"
COMMIT="$(git rev-parse HEAD)"
LOCK="$(sha256sum uv.lock | cut -c1-12)"
EXTRA_ID="$(sha256sum deploy/dgx/dinov3-requirements.txt | cut -c1-12)"

r() { ssh -T -o BatchMode=yes "$GPU_HOST" "$@"; }
say() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
r true 2>/dev/null || { echo "no open login to $GPU_HOST: run 'ssh $GPU_HOST' in a terminal and leave it open" >&2; exit 1; }
BASE="${BASE:-/raid/scratch/$(r id -un)/wave-analysis}"
DINO=/data/ext/dinov3-$DINOV3_COMMIT
EXTRA=/data/envs/dinov3-extra-$EXTRA_ID
INIT=/data/pretrain/init/dinov3_vitb16_lvd1689m_teacher.pth
run_name() { [[ "${1:-}" =~ ^pretrain-[a-z0-9-]+$ ]] || { echo "which run? (see: pretrain.sh status)" >&2; exit 1; }; }
need_code() {
    r test -f "$BASE/repo/$COMMIT/GIT_COMMIT" || { echo "commit $COMMIT is not on the server: run.sh push" >&2; exit 1; }
    r test -f "$BASE/envs/$LOCK/pyvenv.cfg" || { echo "no environment for this uv.lock: run.sh setup" >&2; exit 1; }
}
need_gpu() {
    [[ "${GPU:-}" =~ ^[0-9]+$ ]] || { echo "set GPU=<index> (see: run.sh check)" >&2; exit 1; }
}
# docker run options shared by every container here
common() {
    echo "--user \$(id -u):\$(id -g) -e HOME=/scratch -e USER=wave -v $BASE:/data" \
        "-e COMMIT=$COMMIT -e ENV_DIR=/data/envs/$LOCK -e DINOV3_DIR=$DINO -e EXTRA_DIR=$EXTRA" \
        "-e HF_HOME=/data/cache/hf -e TORCH_HOME=/data/cache/torch -e PYTHONUNBUFFERED=1"
}

cmd="${1:-}"
shift || true
case "$cmd" in
setup)
    need_code
    r bash -s <<EOF
set -e
docker image inspect $IMAGE >/dev/null 2>&1 || docker pull -q $IMAGE
mkdir -p $BASE/ext $BASE/pretrain/init $BASE/logs
if [ ! -f $BASE/ext/dinov3-$DINOV3_COMMIT/.complete ]; then
    rm -rf $BASE/ext/dinov3-$DINOV3_COMMIT
    docker run --rm $(common) $IMAGE bash -c "git clone -q $DINOV3_REPO $DINO && cd $DINO && git checkout -q $DINOV3_COMMIT && touch .complete"
fi
docker run --rm $(common) -e UV_CACHE_DIR=/data/.uv-cache $IMAGE \
    /data/bin/uv pip install -q --python /data/envs/$LOCK/bin/python --target $EXTRA --no-deps \
    -r /data/repo/$COMMIT/deploy/dgx/dinov3-requirements.txt
docker run --rm $(common) $IMAGE bash -c 'cd /data/repo/\$COMMIT && PYTHONPATH=\$PWD/src:\$DINOV3_DIR:\$EXTRA_DIR \
    \$ENV_DIR/bin/python -c "import dinov3.train.train, wave_analysis.pretraining.buoy_views; print(\"dinov3 trainer imports\")"; gcc --version | head -1'
EOF
    say "DINOv3 $DINOV3_COMMIT and its libraries ready"
    ;;
weights)
    need_code
    r bash -s <<EOF
set -e
docker run --rm $(common) $IMAGE bash -c 'cd /data/repo/\$COMMIT && export PYTHONPATH=\$PWD/src:\$DINOV3_DIR:\$EXTRA_DIR &&
    \$ENV_DIR/bin/python scripts/dinov3_weights.py to-dinov3 --out $INIT &&
    \$ENV_DIR/bin/python scripts/dinov3_weights.py check'
EOF
    ;;
start)
    mode="${1:?smoke or full}"
    shift
    extra="$*"
    need_code
    need_gpu
    [[ "$extra" =~ ^[A-Za-z0-9\ ._=/:-]*$ ]] || { echo "overrides may hold only letters, digits, spaces and ._=/:-" >&2; exit 1; }
    [[ "$MEM" =~ ^[0-9]+g$ ]] || { echo "MEM is a size in GB, e.g. 160g" >&2; exit 1; }
    case "$mode" in
    smoke)
        # Both stages, the unfreezing and two teacher checkpoints in about 700 steps.
        extra="optim.epochs=1 train.OFFICIAL_EPOCH_LENGTH=700 schedules.lr.warmup_iterations=100 \
schedules.lr.freeze_last_layer_iterations=50 schedules.lr.freeze_student_iterations=200 \
schedules.lr.backbone_warmup_after_freeze=50 evaluation.eval_period_iterations=350 checkpointing.period=350 $extra"
        run="${RUN:-pretrain-vitb16-fino-$(date -u +%Y%m%dt%H%M)-smoke}" ;;
    full) run="${RUN:-pretrain-vitb16-fino-$(date -u +%Y%m%dt%H%M)}" ;;
    *) echo "start smoke or start full" >&2; exit 1 ;;
    esac
    run_name "$run"
    r test -f "$BASE/ext/dinov3-$DINOV3_COMMIT/.complete" || { echo "run pretrain.sh setup first" >&2; exit 1; }
    r test -f "$BASE${INIT#/data}" || { echo "run pretrain.sh weights first" >&2; exit 1; }
    r "docker ps -a --format '{{.Names}}' | grep -qx wave-$run" && { echo "run $run is running" >&2; exit 1; }
    r bash -s <<EOF
set -e
docker run -d --rm --name wave-$run --label app=wave-analysis --label run=$run \
    --gpus device=$GPU --cpus $CPUS --memory $MEM --shm-size 64g \
    --mount type=tmpfs,destination=/scratch,tmpfs-size=12g \
    $(common) -e MALLOC_ARENA_MAX=2 -e RUN=$run -e MODE=pretrain -e PKG=$PKG -e CONFIG=$CONFIG -e INIT=$INIT \
    -e NUM_WORKERS=$((CPUS - 4)) -e "EXTRA_ARGS=$extra" \
    $IMAGE bash -c 'bash /data/repo/$COMMIT/deploy/container/pretrain.sh >>/data/logs/$run.log 2>&1; s=\$?;
        echo "finished with exit code \$s at \$(date -u +%FT%TZ)" >>/data/logs/$run.log; exit \$s' >/dev/null
EOF
    say "started $run on GPU $GPU; follow it with: deploy/dgx/pretrain.sh logs $run -f"
    ;;
probe)
    run_name "${1:-}"
    src="$1"
    shift
    extra="$*"
    need_code
    need_gpu
    [[ "$extra" =~ ^[A-Za-z0-9\ ._=/:,-]*$ ]] || { echo "bad probe options" >&2; exit 1; }
    name="$src-probe-$(date -u +%Y%m%dt%H%M)"
    r bash -s <<EOF
set -e
docker run -d --rm --name wave-$name --label app=wave-analysis --label run=$name \
    --gpus device=$GPU --cpus $CPUS --memory 96g --shm-size 32g \
    --mount type=tmpfs,destination=/scratch,tmpfs-size=8g \
    $(common) -e RUN=$src -e MODE=probe -e PKG=$PROBE_PKG -e CONFIG=$PROBE_CONFIG \
    -e NUM_WORKERS=$((CPUS - 4)) -e "EXTRA_ARGS=$extra" \
    $IMAGE bash -c 'bash /data/repo/$COMMIT/deploy/container/pretrain.sh >>/data/logs/$name.log 2>&1; s=\$?;
        echo "finished with exit code \$s at \$(date -u +%FT%TZ)" >>/data/logs/$name.log; exit \$s' >/dev/null
EOF
    say "probing $src on GPU $GPU; follow it with: deploy/dgx/pretrain.sh logs $name -f"
    ;;
status)
    r bash -s <<EOF
docker ps --filter label=app=wave-analysis --format '{{.Names}} ({{.Status}})' | grep '^wave-pretrain' | sed 's/^/running: /'
cd $BASE/logs 2>/dev/null || exit 0
for f in \$(ls -t pretrain-*.log 2>/dev/null | head -6); do echo "\${f%.log}: \$(tail -1 \$f | cut -c1-200)"; done
EOF
    ;;
logs)
    run_name "${1:-}"
    r tail -n 40 ${2:+-f} "$BASE/logs/$1.log"
    ;;
stop)
    run_name "${1:-}"
    r docker stop "wave-$1"
    ;;
fetch)
    # Logs, config, metrics and probe results; not the checkpoints (they stay on the server).
    run_name "${1:-}"
    mkdir -p "runs/$1"
    r "cd $BASE/pretrain/$1 && tar -cf - --exclude='ckpt' --exclude='*.pth' --exclude='*.npy' ." | tar -x -C "runs/$1"
    r "cat $BASE/logs/$1.log" >"runs/$1/train.log"
    r "cat $BASE/logs/$1-probe-*.log 2>/dev/null" >"runs/$1/probe.log" || true
    say "results in runs/$1"
    ;;
*)
    sed -n '2,23p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
