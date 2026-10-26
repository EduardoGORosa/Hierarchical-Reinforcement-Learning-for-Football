# Google Research Football + PPO baseline.
#
# Built as a container because gfootball (2019-era C++/SDL engine, gym<=0.21)
# does not compile or install against a current Arch userland (GCC 16, Python 3.14).
# Ubuntu 22.04 gives GCC 11 and Python 3.10, which the engine builds cleanly on,
# while still being new enough for a CUDA 12.8 PyTorch (needed for Blackwell GPUs).
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# --- system deps for the gfootball engine -----------------------------------
RUN apt-get update && apt-get install -yq --no-install-recommends \
      git cmake build-essential ca-certificates \
      libgl1-mesa-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev \
      libsdl2-gfx-dev libboost-all-dev libdirectfb-dev \
      mesa-utils xvfb \
      python3 python3-dev python3-pip \
 && rm -rf /var/lib/apt/lists/*

COPY constraints.txt /tmp/constraints.txt
ENV PIP_CONSTRAINT=/tmp/constraints.txt

# --- python deps ------------------------------------------------------------
# gym 0.21 ships a malformed requirement ("opencv-python>=3.") in its metadata.
# pip >= 24.1 enforces PEP 440 strictly and refuses the package outright, and
# setuptools >= 66 rejects it at build time -- so both are pinned back for this
# one install. gfootball has required gym<=0.21 since 2019 and never moved on.
RUN python3 -m pip install --upgrade "pip==24.0" \
 && python3 -m pip install "setuptools==65.5.0" "wheel==0.38.4" \
 && python3 -m pip install --no-build-isolation "gym==0.21.0"

RUN python3 -m pip install \
      "numpy==1.26.4" \
      "opencv-python==4.10.0.84" \
      "pygame==2.5.2" \
      psutil absl-py six

# --- gfootball engine (the slow part: compiles the C++ game) ----------------
ENV CMAKE_BUILD_PARALLEL_LEVEL=16
COPY vendor/football /gfootball-src
RUN cd /gfootball-src && python3 -m pip install --no-deps --no-build-isolation .
# Verified from / so it exercises the installed package rather than the source
# tree, and instantiates the engine so a missing .so fails the build, not a run.
RUN cd / && python3 -c "\
import gfootball, gfootball.env as e; \
env = e.create_environment(env_name='academy_empty_goal_close', representation='simple115v2'); \
o = env.reset(); \
print('gfootball ok:', gfootball.__file__); \
print('obs', o.shape, 'actions', env.action_space.n); \
env.close()"

# --- learning stack ---------------------------------------------------------
RUN python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu128
RUN python3 -m pip install "stable-baselines3[extra]==2.4.1" "gymnasium==0.29.1" tensorboard rich

# gfootball writes traces/videos here; keep it off the read-only source tree.
ENV SDL_VIDEODRIVER=dummy
WORKDIR /work
