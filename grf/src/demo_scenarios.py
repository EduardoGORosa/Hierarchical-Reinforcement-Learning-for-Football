"""Record demo episodes of academy scenarios for three reference controllers.

Before any agent is trained there is nothing to `watch.py`, but it is still
worth seeing the task: where everyone starts, what "solved" looks like and why
the obvious plan fails. Each scenario is played by

  builtin   GRF's own AI steering our controlled player (action_set v2, action
            19 every step) -- a stand-in for a competent policy.
  scripted  the naive heuristic from probe_scenarios.py: go to the ball, run at
            goal, shoot inside the box.
  random    uniform over the 19 default actions -- where RL training starts.

For each controller it plays --episodes episodes and keeps up to two traces,
<controller>_typical (first episode with the most common outcome) and
<controller>_goal (first goal, if any), renders them with render_pitch, and
writes <controller>.json with the outcome of every episode played.

Usage:
    python3 demo_scenarios.py --out /work/runs/demo
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCENARIOS = [
    "academy_pass_and_shoot_with_keeper",
    "academy_3_vs_1_with_keeper",
    "academy_run_to_score_with_keeper",
]
BUILTIN_AI = 19
# Sector of the angle to the target (multiples of 45 degrees, y grows downward)
# -> GRF movement action.
DIRS = {0: 5, -1: 4, -2: 3, -3: 2, 4: 1, -4: 1, 3: 8, 2: 7, 1: 6}


def move_to(p, t):
    return DIRS[round(math.atan2(t[1] - p[1], t[0] - p[0]) / (math.pi / 4))]


def scripted(o):
    me = o["left_team"][o["active"]]
    if o["ball_owned_team"] == 0 and o["ball_owned_player"] == o["active"]:
        return 12 if me[0] > 0.7 and abs(me[1]) < 0.25 else move_to(me, (1.0, 0.0))
    return move_to(me, o["ball"][:2])


def end_reason(o, score_before):
    """Why an academy episode stopped, read from its final observation."""
    if o["score"][0] > score_before[0]:
        return "goal"
    if o["score"][1] > score_before[1]:
        return "conceded"
    if o["steps_left"] <= 0:
        return "time limit"
    if o["ball_owned_team"] == 1:
        return "lost possession"
    if o["game_mode"] != 0:
        return "ball out of play"
    return "other"


def play(level, controller, episodes, workdir, seed):
    import gfootball.env as football_env

    env = football_env.create_environment(
        env_name=level, representation="raw", rewards="scoring",
        number_of_left_players_agent_controls=1,
        write_full_episode_dumps=True, logdir=str(workdir),
        other_config_options={"action_set": "v2" if controller == "builtin" else "default"},
    )
    rng = np.random.default_rng(seed)
    results, dumps = [], []
    for ep in range(episodes):
        o = env.reset()[0]
        done, steps = False, 0
        while not done:
            if controller == "builtin":
                a = BUILTIN_AI
            elif controller == "scripted":
                a = scripted(o)
            else:
                a = int(rng.integers(19))
            obs, _, done, _ = env.step(a)
            o = obs[0]
            steps += 1
        # The episode_done trace is finalised on the terminal step, so the
        # newest file in the log directory belongs to this episode.
        dumps.append(max(workdir.glob("*.dump"), key=lambda p: p.stat().st_mtime))
        results.append({"episode": ep, "outcome": end_reason(o, (0, 0)), "steps": steps})
    env.close()

    # typical: the first episode with the most common outcome.
    # goal: the first goal, if any -- what solving the task looks like.
    outcomes = [r["outcome"] for r in results]
    typical = outcomes.index(max(set(outcomes), key=outcomes.count))
    picks = {"typical": typical}
    if "goal" in outcomes:
        picks["goal"] = outcomes.index("goal")
    return results, {k: (ep, dumps[ep]) for k, ep in picks.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/work/runs/demo"))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--scenario", choices=SCENARIOS, default=None)
    ap.add_argument("--controller", choices=["builtin", "scripted", "random"], default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import render_pitch

    for level in [args.scenario] if args.scenario else SCENARIOS:
        for controller in [args.controller] if args.controller else ["builtin", "scripted", "random"]:
            out = args.out / level
            work = out / f"_dumps_{controller}"
            work.mkdir(parents=True, exist_ok=True)
            results, picks = play(level, controller, args.episodes, work, args.seed)
            shown = {}
            for kind, (ep, dump) in picks.items():
                name = f"{controller}_{kind}"
                shutil.copy(dump, out / f"{name}.dump")
                render_pitch.render(out / f"{name}.dump", out / f"{name}.mp4")
                shown[kind] = results[ep]
            shutil.rmtree(work)
            goals = sum(r["outcome"] == "goal" for r in results)
            summary = {"level": level, "controller": controller, "shown": shown,
                       "episodes": results, "goal_rate": goals / len(results)}
            (out / f"{controller}.json").write_text(json.dumps(summary, indent=1))
            print(f"{level:40} {controller:8} goals {goals}/{len(results)} | "
                  + ", ".join(f"{k}: ep {v['episode']} {v['outcome']} {v['steps']} steps"
                              for k, v in shown.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
