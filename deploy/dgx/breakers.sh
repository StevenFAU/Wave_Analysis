#!/usr/bin/env bash
# Breaker types on Yin et al.'s VWBT_9000 (docs/experiments/breaker_types.md) on the shared
# GPU server, the way deploy/dgx/run.sh trains: the server runs a pushed commit and each
# call starts one detached container labelled app=wave-analysis on the GPU given.
#
#   export GPU_HOST=<ssh-alias>                    # the server name in ~/.ssh/config
#   deploy/dgx/breakers.sh fetch                   # figshare v5 files into BASE/ext, MD5-checked
#   deploy/dgx/breakers.sh push                    # the code of this commit
#   deploy/dgx/breakers.sh decoder                 # OpenCV (its own FFmpeg) beside the environment
#   GPU=<index> deploy/dgx/breakers.sh features <tag> [--limit 24]   # decode once, frozen features
#   GPU=<index> deploy/dgx/breakers.sh heads <features run> <tag> [fit options]  # heads + score
#   deploy/dgx/breakers.sh results <run>           # the run's small files into runs/<run>
#   deploy/dgx/run.sh status | logs <run> [-f] | stop <run>
#
# GPU must be given and have MIN_FREE_GB (default 30) free (see run.sh check). Runs are
# named video-breakers-<utc>-<kind>-<tag>; the feature cache (about 6 GB) stays on the
# server under BASE/runs. Only this project's containers and BASE are touched.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GPU_HOST="${GPU_HOST:?set GPU_HOST to the GPU server name in ~/.ssh/config}"
IMAGE=python:3.12.16-slim-trixie
COMMIT="$(git rev-parse HEAD)"
LOCK="$(sha256sum uv.lock | cut -c1-12)"
CV2=opencv-python-headless==5.0.0.93 # uv.lock's version (imaging extra)
CV2_DIR="cv2-5.0.0.93"
SRC=ext/vwbt9000-v5
MIN_FREE_GB="${MIN_FREE_GB:-30}"

r() { ssh -T -o BatchMode=yes "$GPU_HOST" "$@"; }
say() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
r true 2>/dev/null || { echo "no open login to $GPU_HOST: run 'ssh $GPU_HOST' in a terminal and leave it open" >&2; exit 1; }
BASE="${BASE:-/raid/scratch/$(r id -un)/wave-analysis}"

need_code() {
    r test -f "$BASE/repo/$COMMIT/GIT_COMMIT" || { echo "commit $COMMIT is not on the server: breakers.sh push" >&2; exit 1; }
    r test -f "$BASE/envs/$LOCK/pyvenv.cfg" || { echo "no environment for this uv.lock: run.sh setup" >&2; exit 1; }
}
pick_gpu() {  # sets gpu
    local free
    gpu="${GPU:?set GPU to a GPU with enough free memory (run.sh check)}"
    [[ "$gpu" =~ ^[0-7]$ ]] || { echo "bad GPU '$gpu'" >&2; exit 1; }
    free="$(r "nvidia-smi -i $gpu --query-gpu=memory.used,memory.total --format=csv,noheader,nounits" |
        awk -F', *' '{printf "%d", ($2 - $1) / 1024}')"
    ((free >= MIN_FREE_GB)) || { echo "GPU $gpu has $free GB free, less than $MIN_FREE_GB" >&2; exit 1; }
    say "GPU $gpu: $free GB free"
}
start() {  # start <run> <gpu> <mode> <extra args>
    local run="$1" gpu="$2" mode="$3" extra="$4"
    r "docker ps -a --format '{{.Names}}' | grep -qx wave-$run" && { echo "run $run exists; wait a minute" >&2; exit 1; }
    r bash -s <<EOF
set -e
docker image inspect $IMAGE >/dev/null 2>&1 || docker pull -q $IMAGE >/dev/null
docker run -d --rm --name wave-$run --label app=wave-analysis --label run=$run \
    --gpus device=$gpu --cpus 16 --memory 64g --shm-size 16g \
    --mount type=tmpfs,destination=/scratch,tmpfs-size=8g \
    --user \$(id -u):\$(id -g) -e HOME=/scratch -v $BASE:/data \
    -e RUN=$run -e MODE=$mode -e COMMIT=$COMMIT -e ENV_DIR=/data/envs/$LOCK -e CV2_DIR=/data/envs/$CV2_DIR \
    -e SRC=/data/$SRC -e "EXTRA_ARGS=$extra" -e NUM_WORKERS=12 -e TORCH_HOME=/data/cache/torch \
    -e HF_HOME=/data/cache/hf -e HF_HUB_OFFLINE=1 -e PYTHONUNBUFFERED=1 -e TORCH_DISABLE_NATIVE_JIT=1 \
    $IMAGE bash -c 'bash /data/repo/$COMMIT/deploy/container/breakers.sh >/data/logs/$run.log 2>&1; s=\$?;
        echo "finished with exit code \$s at \$(date -u +%FT%TZ)" >>/data/logs/$run.log; exit \$s' >/dev/null
EOF
    say "started $run on GPU $gpu; follow it with: deploy/dgx/run.sh logs $run -f"
}

cmd="${1:-}"
shift || true
case "$cmd" in
fetch)
    # figshare v5 (doi:10.6084/m9.figshare.28814993.v5): file id and the record's MD5
    r bash -s <<EOF
set -euo pipefail
mkdir -p $BASE/$SRC && cd $BASE/$SRC
get() {
    if [ -f "\$1" ] && [ "\$(md5sum "\$1" | cut -d' ' -f1)" = "\$3" ]; then echo "\$1 there"; return; fi
    curl -sS -L --retry 5 --retry-delay 10 -C - -o "\$1.part" "https://ndownloader.figshare.com/files/\$2"
    have="\$(md5sum "\$1.part" | cut -d' ' -f1)"
    [ "\$have" = "\$3" ] || { echo "\$1: MD5 \$have, the record says \$3" >&2; exit 1; }
    mv "\$1.part" "\$1" && echo "\$1 downloaded, MD5 \$3"
}
get loso_split_by_database_filename.csv 68107222 e85ec743df64c5abfc674bfbe550b538
get ERA5_offshore_wave_wind_by_clip.csv 56176808 a15273a3b886fc053a2b93c5fddb91b3
get Set_A.zip 56199590 64d21dd37d5d9c88b079e97ba60fb82c
get Set_B.zip 56199587 fe0090db9f5b482e9867ed3f7de9c99f
EOF
    ;;
push)
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "commit first: the server runs a commit, not a working tree" >&2
        exit 1
    fi
    if r test -f "$BASE/repo/$COMMIT/GIT_COMMIT"; then
        say "code $COMMIT already there"
    else
        r mkdir -p "$BASE/repo" "$BASE/logs" "$BASE/runs"
        r rm -rf "$BASE/repo/$COMMIT.part"
        git archive --format=tar --prefix="$COMMIT.part/" HEAD | r tar -x -C "$BASE/repo"
        r "echo $COMMIT > $BASE/repo/$COMMIT.part/GIT_COMMIT && mv $BASE/repo/$COMMIT.part $BASE/repo/$COMMIT"
        say "code $COMMIT copied"
    fi
    ;;
decoder)
    # The environment (run.sh setup) has no video decoder; OpenCV's wheel brings FFmpeg.
    # Installed without dependencies into its own folder, put on PYTHONPATH by the container.
    need_code
    r bash -s <<EOF
set -e
test -d $BASE/envs/$CV2_DIR/cv2 && { echo "decoder there"; exit 0; }
docker run --rm --user \$(id -u):\$(id -g) -e HOME=/tmp -v $BASE:/data $IMAGE bash -c \
    "/data/bin/uv pip install --quiet --no-deps --target /data/envs/$CV2_DIR.part $CV2 \
     --python /data/envs/$LOCK/bin/python && mv /data/envs/$CV2_DIR.part /data/envs/$CV2_DIR"
echo "decoder installed: $CV2"
EOF
    ;;
features)
    tag="${1:?a tag, e.g. v1}"
    shift
    extra="$*"
    [[ "$tag" =~ ^[a-z0-9-]+$ ]] || { echo "tag: a-z, 0-9 and - only" >&2; exit 1; }
    [[ "$extra" =~ ^[A-Za-z0-9\ ._=/:-]*$ ]] || { echo "options may hold only letters, digits, spaces and ._=/:-" >&2; exit 1; }
    need_code
    r test -d "$BASE/envs/$CV2_DIR/cv2" || { echo "no decoder: breakers.sh decoder" >&2; exit 1; }
    r test -f "$BASE/$SRC/Set_B.zip" || { echo "no data: breakers.sh fetch" >&2; exit 1; }
    pick_gpu
    start "video-breakers-$(date -u +%Y%m%dt%H%M)-features-$tag" "$gpu" features "$extra"
    ;;
heads)
    features="${1:?the features run, e.g. video-breakers-...-features-v1}"
    tag="${2:?a tag, e.g. v1}"
    shift 2
    extra="$*"
    [[ "$features" =~ ^video-breakers-[a-z0-9-]+$ && "$tag" =~ ^[a-z0-9-]+$ ]] || { echo "bad run or tag" >&2; exit 1; }
    [[ "$extra" =~ ^[A-Za-z0-9\ ._=/:-]*$ ]] || { echo "options may hold only letters, digits, spaces and ._=/:-" >&2; exit 1; }
    need_code
    r test -f "$BASE/runs/$features/features/meta.json" || { echo "no finished feature cache $features" >&2; exit 1; }
    pick_gpu
    start "video-breakers-$(date -u +%Y%m%dt%H%M)-heads-$tag" "$gpu" heads "--cache /data/runs/$features/features $extra"
    ;;
results)
    run="${1:?which run}"
    [[ "$run" =~ ^video-breakers-[a-z0-9-]+$ ]] || { echo "bad run" >&2; exit 1; }
    mkdir -p "runs/$run"
    r "cd $BASE/runs/$run && find . -type f -size -50M ! -name '*.npy' -print0 | tar --null -T - -cf -" | tar -x -C "runs/$run"
    r cat "$BASE/logs/$run.log" >"runs/$run/run.log"
    say "small files of $run in runs/$run"
    ;;
*)
    sed -n '2,18p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
