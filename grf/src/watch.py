"""Watch a trained agent play: run episodes and render them to video.

GRF records an episode as a `.dump` trace, which is then rendered separately.
Two views are available:

  pitch  (default) a drawn top-down tactical view: full pitch markings, every
         player, possession, ball trail, and the controlled agent highlighted.
         Pure numpy/cv2 from the trace, so it needs no GPU, no GL and no X.
  2d     gfootball's own built-in dump renderer. Minimal (a rectangle and a few
         letters) but it is the upstream tool, kept for cross-checking.
  3d     the game's broadcast camera. Needs a GPU-backed GL context, which a
         headless container does not have -- software GL takes minutes per
         episode. Left in deliberately, but expect it to be impractical here.

Usage:
    python3 watch.py --run runs/empty_goal_close_seed0 --episodes 3
    python3 watch.py --run runs/empty_goal_close_seed0 --view 3d
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True,
                    help="training run directory (containing final_model.zip)")
    ap.add_argument("--model", type=Path, default=None,
                    help="checkpoint to load (default: <run>/final_model.zip)")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--view", choices=["pitch", "2d", "3d"], default="pitch")
    ap.add_argument("--fps", type=int, default=10, help="output video frame rate")
    ap.add_argument("--scale", type=float, default=10.0, help="pitch view: pixels per metre")
    ap.add_argument("--level", default=None, help="override the scenario from config.json")
    ap.add_argument("--representation", default=None, help="override the representation")
    ap.add_argument("--stacked", dest="stacked", action="store_true", default=None,
                    help="override: model was trained on 4-frame-stacked observations")
    ap.add_argument("--no-stacked", dest="stacked", action="store_false",
                    help="override: model was trained without frame stacking")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of taking the argmax")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from gfootball_env import GFootballEnv

    run_dir = args.run.resolve()
    model_path = args.model or (run_dir / "final_model.zip")
    if not model_path.exists():
        print(f"error: no model at {model_path}", file=sys.stderr)
        return 1

    # Recover how the agent was trained, so the observation it sees now matches
    # what its network expects. A mismatch here fails silently as bad play.
    cfg_path = run_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    level = args.level or cfg.get("level", "academy_empty_goal_close")
    representation = args.representation or cfg.get("representation", "simple115v2")
    stacked = cfg.get("stacked", False) if args.stacked is None else args.stacked
    stacked = bool(stacked)

    video_dir = run_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"[watch] model  {model_path}")
    print(f"[watch] level  {level} ({representation}{', stacked' if stacked else ''})")
    print(f"[watch] view   {args.view}")
    if args.view == "3d" and not os.environ.get("DISPLAY"):
        print("[watch] error: --view 3d needs DISPLAY set and an X socket mounted. "
              "Use ./run.sh watch, which wires that up.", file=sys.stderr)
        return 1

    env = GFootballEnv(
        env_name=level,
        representation=representation,
        stacked=stacked,
        rewards=cfg.get("rewards", "scoring,checkpoints"),
        logdir=str(video_dir),
        write_full_episode_dumps=True,
        write_video=(args.view == "3d"),
        render=(args.view == "3d"),
    )
    model = PPO.load(str(model_path), device="cpu")

    # SB3 wraps image observations in VecTransposeImage during training, so an
    # image-trained model expects channel-first (C, H, W) while the raw GRF env
    # emits channel-last (H, W, C). Reapply that transpose here, or the CNN
    # silently receives garbage rather than raising.
    want = tuple(model.observation_space.shape)
    have = tuple(env.observation_space.shape)
    needs_transpose = len(want) == 3 and want != have and want == (have[2], have[0], have[1])
    if needs_transpose:
        print(f"[watch] transposing observations {have} -> {want} (SB3 image convention)")
    elif want != have:
        print(f"[watch] error: model expects {want} but env gives {have}. "
              f"Check --level / --representation / --stacked.", file=sys.stderr)
        return 1

    def to_model(o):
        return np.transpose(o, (2, 0, 1)) if needs_transpose else o

    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        total, steps, done = 0.0, 0, False
        while not done:
            action, _ = model.predict(to_model(obs), deterministic=not args.stochastic)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
            steps += 1
        returns.append(total)
        print(f"[watch] episode {ep + 1}/{args.episodes}: return {total:+.2f} in {steps} steps")
    env.close()

    print(f"[watch] mean return {sum(returns) / len(returns):+.3f} over {len(returns)} episodes")

    dumps = sorted(video_dir.glob("*.dump"))
    if not dumps:
        print("[watch] no episode dumps were written", file=sys.stderr)
        return 1

    # Rendering happens after play rather than during it, so recording costs
    # nothing at rollout time.
    if args.view == "pitch":
        import render_pitch
        for dump in dumps:
            print(f"[watch] rendering {dump.name} ...")
            render_pitch.render(dump, video_dir / f"{dump.stem}.mp4",
                                fps=args.fps, scale=args.scale)
    elif args.view == "2d":
        from gfootball.env import script_helpers
        helper = script_helpers.ScriptHelpers()
        for dump in dumps:
            print(f"[watch] rendering {dump.name} ...")
            helper.dump_to_video(str(dump))

    videos = sorted(video_dir.glob("*.mp4")) + sorted(video_dir.glob("*.avi")) \
        + sorted(video_dir.glob("*.webm"))
    if videos:
        print(f"\n[watch] {len(videos)} video(s) in {video_dir}:")
        for v in videos:
            print(f"  {v.name}  ({v.stat().st_size / 1e6:.1f} MB)")
    else:
        print("\n[watch] no video produced", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
