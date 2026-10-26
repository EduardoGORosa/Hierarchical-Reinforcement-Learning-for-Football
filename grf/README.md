# GRF environment + PPO baseline

Working environment for the thesis experiments: Google Research Football running
under a current toolchain, with a PPO baseline that trains end to end.

The point of this directory is to make the environment a solved problem, so the
research contribution can be about hierarchy rather than about build systems.

## Why a container

gfootball 2.10.3 is a 2019 codebase: a C++/SDL game engine plus a Python layer
pinned to `gym<=0.21`. It does not build or install against this machine's
userland (Arch, GCC 16, Python 3.14), and installing the system libraries it
needs would require root. Ubuntu 22.04 in a container gives GCC 11 and Python
3.10, which the engine compiles on cleanly, while still being new enough to run
a CUDA 12.8 PyTorch — required for the RTX 5070 (Blackwell, `sm_120`).

Three specific breakages are pinned around in the `Dockerfile`:

| Problem | Fix |
| --- | --- |
| gym 0.21's metadata has a malformed requirement (`opencv-python>=3.`); pip >= 24.1 rejects the package and setuptools >= 66 rejects the build | pin `pip==24.0`, `setuptools==65.5.0`, install with `--no-build-isolation` |
| `six` is imported by `gfootball.env` but missing from gfootball's own `requirements.txt` and `setup.py` | install it explicitly |
| numpy 2 breaks gym-0.21-era code paths | pin `numpy==1.26.4` image-wide via `constraints.txt` |

The build ends by instantiating a real scenario, so a broken engine fails the
image build rather than a training run three days later.

## Layout

```
Dockerfile          the environment
constraints.txt     numpy pin applied to every pip install
run.sh              build / smoke / train / tb / shell
vendor/football     gfootball 2.10.3 @ 3d9e754 (vendored, unpatched)
src/gfootball_env.py  legacy-gym -> Gymnasium adapter
src/smoke_test.py     environment sanity checks
src/train_ppo.py      SB3 PPO baseline
src/watch.py          play episodes with a trained model, render to video
src/render_pitch.py   top-down tactical renderer for episode dumps
src/plot_curve.py     learning curves from monitor.csv
src/hrl/             two-layer agent: TODO boilerplate + checks (./run.sh hrl check)
runs/               training output (gitignored)
```

## Use

```bash
./run.sh build           # ~10 min, mostly the CUDA torch download
./run.sh smoke           # verify engine, spaces, rollout, SB3 compatibility
./run.sh train           # PPO on academy_empty_goal_close
./run.sh tb              # tensorboard at localhost:6006
```

Training options:

```bash
./run.sh train --level 11_vs_11_easy_stochastic \
               --representation extracted --stacked \
               --timesteps 5000000 --n-envs 16 --seed 1
```

`--representation simple115v2` uses an MLP on CPU; `extracted` (the 72x96
minimap, SMM) and `pixels` use a CNN on the GPU. The device is chosen
automatically and can be forced with `--device`.

## Watching a trained agent

```bash
./run.sh watch --run /work/runs/empty_goal_close_seed0 --episodes 3
```

Plays episodes with the saved model, writes one `.mp4` per episode into
`runs/<run>/videos/`, and prints the return of each. Paths passed to `--run`
are container paths, so they start with `/work/runs/...`; on the host the same
files are under `grf/runs/...` and open in any player (mpv, vlc).

Three views, via `--view`:

| view | what it is |
| --- | --- |
| `pitch` (default) | a drawn top-down tactical view: full pitch markings, every player, ball trail, possession, controlled agent ringed in yellow. Pure cv2 from the trace — no GPU, no GL, no display. |
| `2d` | gfootball's own dump renderer. A bare rectangle and a few letters; kept for cross-checking against upstream. |
| `3d` | the game's real broadcast camera — stadium, kits, minimap. Opens a live window on the desktop *and* writes an `.avi`. `run.sh` passes the host X socket and the GPU through automatically when it sees `--view 3d`. |

For `pitch` and `2d`, rendering happens after play, from the episode dump, so
recording costs nothing at rollout time. `src/render_pitch.py` also runs
standalone on any `.dump`. `3d` is different: it must render during play,
because the frames come from the game engine itself.

### How 3d works, and why it is off by default

The engine needs a real GL context. Inside the container only software GL
(llvmpipe) is available by default, and that takes **over ten minutes for a
single 30-step episode** — which is why the other views deliberately keep
`SDL_VIDEODRIVER=dummy`: merely pointing SDL at a display makes the engine spin
up its graphics pipeline even when `render=False`, and a 3-second job becomes a
10-minute one.

With the host's X server and the NVIDIA driver passed through, it is faster
than real time: a full 3001-step 11v11 match renders in about 72 seconds.
`run.sh` adds this only for `--view 3d`:

```
-e DISPLAY=$DISPLAY -e SDL_VIDEODRIVER=x11 \
-e NVIDIA_DRIVER_CAPABILITIES=all -v /tmp/.X11-unix:/tmp/.X11-unix
```

`NVIDIA_DRIVER_CAPABILITIES=all` is the part that matters — the default
container setup exposes `compute` (so CUDA and PyTorch work) but not
`graphics`, so libGLX_nvidia is simply absent until you ask for it. XWayland
satisfies the X requirement, so this works on a Wayland desktop; on a headless
machine there is no display to borrow and `pitch` is the view to use.

A coordinate note that cost some time to confirm, and is worth knowing before
reading any GRF visualisation: raw observations use **one absolute frame** for
both teams (verified on 11v11 — left team spans x in [-1.01, 0.00], right team
[0.01, 1.01]), with no mirroring of the right team. Scenario *definitions* are
different: `builder.AddPlayer` coordinates for the right team are mirrored by
the engine, which is why `academy_empty_goal_close` declares its opponent
keeper at `x=+1.0` and the observation reports it at `x=-1.0` — parked at the
far goal, which is precisely what leaves the near goal empty.

## The gym/Gymnasium boundary

gfootball speaks the legacy gym API (`reset()` returns an observation,
`step()` returns a 4-tuple); SB3 2.x speaks Gymnasium (`reset(seed=...)`
returning `(obs, info)`, `step()` returning a 5-tuple). Both libraries are
installed, and `src/gfootball_env.py` converts at the boundary — spaces
included — instead of patching the vendored engine.

One judgement call is recorded there and should be restated in the methodology
chapter: GRF emits a single `done`, which the adapter maps to `terminated`,
never `truncated`. A scenario that runs out of time is a failed attempt, not an
arbitrary cutoff, so the critic should not bootstrap past it.

## Baseline hyperparameters

Defaults in `train_ppo.py` are the tuned PPO configuration published with the
GRF paper (`gfootball/examples/repro_checkpoint_easy.sh`), translated from
OpenAI Baselines' PPO2 to SB3: `nsteps -> n_steps`, `nminibatches -> batch_size`,
`noptepochs -> n_epochs`, `cliprange -> clip_range`, plus `lam=0.95`,
`vf_coef=0.5`.

Two things worth keeping straight in the writing:

- GRF's core action set is **19 discrete actions** (confirmed by the smoke test).
  Counts of 21 come from tallying sticky/held actions separately.
- PPO is **one of three** baselines benchmarked in Kurach et al. (2019),
  alongside IMPALA and Ape-X DQN — not "the" standard GRF baseline.

## Notes for the experiments

- `plot_curve.py` plots return against **environment steps**, which is the axis
  the sample-efficiency comparison has to be made on.
- gfootball keeps process-global state, so one environment per process:
  training uses `SubprocVecEnv` with `start_method="spawn"` (forking a process
  that has already initialised the SDL engine is not safe).
- Seeds matter here. Anything reported should be several seeds with spread
  shown, not a single run.
