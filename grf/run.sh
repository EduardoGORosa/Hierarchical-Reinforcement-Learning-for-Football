#!/usr/bin/env bash
# Convenience wrapper around the grf-ppo container.
#
#   ./run.sh build            rebuild the image
#   ./run.sh smoke            run environment sanity checks
#   ./run.sh train [args...]  run PPO (args forwarded to src/train_ppo.py)
#   ./run.sh watch [args...]  play episodes with a trained model, render to video
#   ./run.sh hrl <mod> [args] run hrl/<mod>.py as a module (check, intents, wrappers, train_hrl)
#   ./run.sh shell            interactive shell inside the container
#   ./run.sh tb               tensorboard on ./runs at http://localhost:6006
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=grf-ppo:latest
mkdir -p "$HERE/runs"

# --gpus is skipped when no NVIDIA runtime is present so the image still works
# on a CPU-only machine.
GPU_ARGS=()
if docker info 2>/dev/null | grep -qi nvidia; then
  GPU_ARGS=(--gpus all)
fi

# -it only when attached to a real terminal, so the same script works from
# scripts, CI and non-interactive shells.
TTY_ARGS=()
[ -t 0 ] && TTY_ARGS=(-it)

# Custom scenarios (grf/scenarios/*.py) are mounted into gfootball's own scenarios package, so
# --level <file name> finds them. DOCKER_EXTRA adds flags, e.g. DOCKER_EXTRA="--cpu-shares=256"
# runs a job at lower CPU priority than the default 1024.
SCENARIO_ARGS=()
for f in "$HERE"/scenarios/*.py; do
  [ -e "$f" ] && SCENARIO_ARGS+=(-v "$f:/usr/local/lib/python3.10/dist-packages/gfootball/scenarios/$(basename "$f"):ro")
done
read -r -a EXTRA_ARGS <<< "${DOCKER_EXTRA:-}"

docker_run() {
  docker run --rm "${TTY_ARGS[@]}" \
    "${GPU_ARGS[@]}" "${SCENARIO_ARGS[@]}" "${EXTRA_ARGS[@]}" \
    --shm-size=2g \
    -u "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e USER=grf \
    -v "$HERE/src:/work/src:ro" \
    -v "$HERE/runs:/work/runs" \
    -w /work \
    "$@"
}

cmd="${1:-smoke}"; shift || true

case "$cmd" in
  build) docker build -t "$IMAGE" "$HERE" ;;
  smoke) docker_run "$IMAGE" python3 /work/src/smoke_test.py ;;
  train) docker_run "$IMAGE" python3 /work/src/train_ppo.py "$@" ;;
  # The headless views keep SDL_VIDEODRIVER=dummy from the image: pointing SDL
  # at a display makes the engine initialise its graphics pipeline even when
  # render=False, which turns a 3-second job into a 10-minute one on software GL.
  #
  # --view 3d does need a GL display, so it borrows the host's X server (XWayland
  # counts) and the real GPU. Software GL "works" but takes minutes per episode;
  # with the NVIDIA driver passed through it is faster than real time.
  watch)
    WATCH_ARGS=()
    if [[ " $* " == *" 3d "* ]]; then
      if [ ! -S "/tmp/.X11-unix/X${DISPLAY##*:}" ] && [ ! -d /tmp/.X11-unix ]; then
        echo "error: --view 3d needs a running X server; none found at /tmp/.X11-unix" >&2
        exit 1
      fi
      WATCH_ARGS=(
        -e "DISPLAY=${DISPLAY:-:0}"
        -e SDL_VIDEODRIVER=x11
        -e NVIDIA_DRIVER_CAPABILITIES=all
        -v /tmp/.X11-unix:/tmp/.X11-unix
      )
    fi
    docker_run "${WATCH_ARGS[@]}" "$IMAGE" python3 /work/src/watch.py "$@" ;;
  hrl)   mod="${1:-check}"; shift || true
         docker_run -w /work/src "$IMAGE" python3 -m "hrl.$mod" "$@" ;;
  shell) docker_run "$IMAGE" bash ;;
  tb)    docker_run -p 6006:6006 "$IMAGE" \
           tensorboard --logdir /work/runs --host 0.0.0.0 --port 6006 ;;
  *)     echo "unknown command: $cmd"; sed -n '2,10p' "$0"; exit 1 ;;
esac
