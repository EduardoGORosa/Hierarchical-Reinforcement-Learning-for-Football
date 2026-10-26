"""Render an episode in GRF's broadcast (3D) camera, headless.

The engine draws the real 3D view even with SDL_VIDEODRIVER=dummy: frames come
back in obs['frame'], so no GPU, no X server and no --gpus are needed (~16
steps/s). Rendering does not affect the simulation, and a fixed
game_engine_random_seed replays an episode exactly, so this searches for an
episode with the wanted outcome without rendering, then replays just that one
with the camera on.

Usage:
    python3 render3d.py --level academy_3_vs_1_with_keeper --out /work/runs/3d
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

BUILTIN_AI = 19
FONT = cv2.FONT_HERSHEY_DUPLEX
DIRS = {0: 5, -1: 4, -2: 3, -3: 2, 4: 1, -4: 1, 3: 8, 2: 7, 1: 6}


def scripted(o):
    me = o["left_team"][o["active"]]
    if o["ball_owned_team"] == 0 and o["ball_owned_player"] == o["active"]:
        if me[0] > 0.7 and abs(me[1]) < 0.25:
            return 12
        t = (1.0, 0.0)
    else:
        t = o["ball"][:2]
    return DIRS[round(math.atan2(t[1] - me[1], t[0] - me[0]) / (math.pi / 4))]


def end_reason(o):
    """Why an academy episode stopped, read from its final observation."""
    if o["score"][0] > 0:
        return "GOAL"
    if o["score"][1] > 0:
        return "conceded"
    if o["steps_left"] <= 0:
        return "time limit"
    if o["ball_owned_team"] == 1:
        return "lost possession"
    if o["game_mode"] != 0:
        return "ball out of play"
    return "other"


def label(frame, top, bottom=None):
    """Burn a caption into a copy of the frame."""
    f = frame.copy()
    cv2.rectangle(f, (0, f.shape[0] - 64), (f.shape[1], f.shape[0]), (18, 18, 18), -1)
    cv2.putText(f, top, (24, f.shape[0] - 38), FONT, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
    if bottom:
        cv2.putText(f, bottom, (24, f.shape[0] - 12), FONT, 0.9, (90, 90, 245), 2, cv2.LINE_AA)
    return f


def make(level, controller, seed, render):
    import gfootball.env as football_env
    return football_env.create_environment(
        env_name=level, representation="raw", rewards="scoring",
        number_of_left_players_agent_controls=1, render=render,
        other_config_options={"action_set": "v2" if controller == "builtin" else "default",
                              "game_engine_random_seed": seed})


def act(controller, o, rng):
    if controller == "builtin":
        return BUILTIN_AI
    if controller == "scripted":
        return scripted(o)
    return int(rng.integers(19))


def run(level, controller, seed, episodes, render, capture_ep=None, out=None, fps=10,
        capture_all=False, caption=""):
    env = make(level, controller, seed, render)
    rng = np.random.default_rng(0)
    writer, result = None, []
    for ep in range(episodes):
        o = env.reset(); done, steps, R = False, 0, 0.0
        shots = []
        capturing = render and (capture_all or ep == capture_ep)
        while not done:
            o, r, done, _ = env.step(act(controller, o[0], rng))
            steps += 1; R += r
            if capturing:
                f = np.asarray(o[0]["frame"])[:, :, ::-1]      # RGB -> BGR
                if writer is None:
                    h, w = f.shape[:2]
                    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                shots.append(f)
        why = end_reason(o[0])
        if capturing:
            top = f"{caption}  |  step {{}}/{steps}"
            for i, f in enumerate(shots):
                writer.write(label(f, top.format(i + 1)))
            for _ in range(fps):                      # hold the last frame with the outcome
                writer.write(label(shots[-1], top.format(steps), f"episode ended: {why}"))
        result.append({"episode": ep, "goal": R > 0, "steps": steps, "why": why})
        print(f"   episode {ep + 1}: {why} after {steps} steps", flush=True)
        if capture_ep is not None and ep == capture_ep:
            break
    if writer is not None:
        writer.release()
    env.close()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", required=True)
    ap.add_argument("--controller", default="builtin", choices=["builtin", "scripted", "random"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--search", type=int, default=12, help="episodes to search for a goal")
    ap.add_argument("--out", type=Path, default=Path("/work/runs/3d"))
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--all", action="store_true",
                    help="render every searched episode into one clip instead of the first goal")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.level.replace('academy_', '')}_{args.controller}"
    target = args.out / (f"{stem}_seed{args.seed}_all_3d.mp4" if args.all else f"{stem}_3d.mp4")

    if args.all:
        cap = f"{args.level.replace('academy_', '')}  |  {args.controller}  |  seed {args.seed}"
        res = run(args.level, args.controller, args.seed, args.search, render=True,
                  out=target, fps=args.fps, capture_all=True, caption=cap)
        goals = sum(r["goal"] for r in res)
        print(f"{goals} goals in {len(res)} episodes")
        print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")
        return 0

    # Pass 1: find the first scoring episode, without paying for rendering.
    res = run(args.level, args.controller, args.seed, args.search, render=False)
    goals = [r["episode"] for r in res if r["goal"]]
    if not goals:
        print(f"{args.level}: no goal in {args.search} episodes with seed {args.seed}")
        return 1
    ep = goals[0]
    print(f"{args.level}: goal on episode {ep} ({res[ep]['steps']} steps) -> rendering it")

    # Pass 2: same seed, same controller, render only that episode.
    run(args.level, args.controller, args.seed, ep + 1, render=True, capture_ep=ep,
        out=target, fps=args.fps)
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
