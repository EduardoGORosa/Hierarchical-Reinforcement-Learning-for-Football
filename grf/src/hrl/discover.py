"""Light option discovery (variation 3): which behaviours does the trained FLAT agent actually use?

    ./run.sh hrl discover --run ov10_flat_s0 --episodes 400 --k 5

Plays the frozen flat model, cuts every episode into WINDOW-step windows, describes each window with
football features, and clusters them (k-means, numpy, standardised features, best of 10 starts).
Each cluster is then named by its most distinctive features, the "discover first, name later"
route (thesis_ideas 3.4, route 2), done as analysis rather than as a training method: if the
clusters line up with pass / dribble / shoot, the designed options are anchored in the agent's own
behaviour; if a large cluster has no designed option, that is a missing option.

Features per window (actions are GRF's default set):
  share of steps pressing: forward (4,5,6), back (1,2,8), sideways (3,7), idle (0), pass (9,10,11),
  shot (12), sprint (13), dribble (17)
  ball forward and sideways movement (m), owner changed (0/1), we still own the ball at the end,
  mean distance from the ball to the nearest opponent (m), distance to goal at the start (m)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # grf/src

WINDOW = 10
GROUPS = {"forward": (4, 5, 6), "back": (1, 2, 8), "sideways": (3, 7), "idle": (0,),
          "pass": (9, 10, 11), "shot": (12,), "sprint": (13,), "dribble": (17,)}
NAMES = list(GROUPS) + ["ball_dx_m", "ball_dy_m", "owner_changed", "kept_ball", "pressure_m", "goal_dist_m"]


def window_features(raws: list, actions: list[int]) -> np.ndarray:
    from hrl.options import X_METRES, Y_METRES
    a = np.asarray(actions)
    f = [float(np.isin(a, g).mean()) for g in GROUPS.values()]
    b0, b1 = raws[0]["ball"], raws[-1]["ball"]
    f += [(b1[0] - b0[0]) * X_METRES, abs(b1[1] - b0[1]) * Y_METRES]
    owners = [r["ball_owned_player"] for r in raws if r["ball_owned_team"] == 0]
    f += [float(len(set(owners)) > 1), float(raws[-1]["ball_owned_team"] == 0)]
    press = []
    for r in raws:
        d = np.asarray(r["right_team"])[1:] - np.asarray(r["ball"][:2])  # field players, not the keeper
        press.append(float(np.min(np.hypot(d[:, 0] * X_METRES, d[:, 1] * Y_METRES))))
    f += [float(np.mean(press)), float(np.hypot((1 - b0[0]) * X_METRES, b0[1] * Y_METRES))]
    return np.asarray(f, dtype=float)


def collect(run: str, episodes: int, seed: int) -> np.ndarray:
    from stable_baselines3 import PPO
    from gfootball_env import GFootballEnv
    model = PPO.load(f"/work/runs/{run}/final_model.zip", device="cpu")
    env = GFootballEnv(env_name="academy_3_vs_1_with_keeper", representation="simple115v2", rewards="scoring")
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        raws, acts, done = [env.raw_observation()], [], False
        while not done:
            a = int(model.predict(obs, deterministic=False)[0])
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
            raws.append(env.raw_observation())
            acts.append(a)
        for s in range(0, len(acts) - WINDOW + 1, WINDOW):
            rows.append(window_features(raws[s:s + WINDOW + 1], acts[s:s + WINDOW]))
    env.close()
    return np.asarray(rows)


def kmeans(x: np.ndarray, k: int, starts: int = 10, iters: int = 100, seed: int = 0):
    rng, best = np.random.default_rng(seed), None
    for _ in range(starts):
        c = x[rng.choice(len(x), k, replace=False)]
        for _ in range(iters):
            lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(2), 1)
            new = np.asarray([x[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = float(((x - c[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, lab, c)
    return best[1], best[2]


def name(z_centre: np.ndarray) -> str:
    """The cluster's two most distinctive features (largest standardised deviation from the mean)."""
    top = np.argsort(-np.abs(z_centre))[:2]
    return " / ".join(("more " if z_centre[i] > 0 else "less ") + NAMES[i] for i in top)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="ov10_flat_s0")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=50_000)
    p.add_argument("--out", default="/work/runs/variations/discover.json")
    a = p.parse_args()
    x = collect(a.run, a.episodes, a.seed)
    mu, sd = x.mean(0), x.std(0) + 1e-9
    lab, c = kmeans((x - mu) / sd, a.k)
    clusters = []
    for j in np.argsort(-np.bincount(lab, minlength=a.k)):
        raw_mean = x[lab == j].mean(0)
        clusters.append(dict(size=float(np.mean(lab == j)), name=name(c[j]),
                             mean={n: float(v) for n, v in zip(NAMES, raw_mean)}))
        print(f"{clusters[-1]['size']:5.0%}  {clusters[-1]['name']}")
        print("       " + "  ".join(f"{n}={v:.2f}" for n, v in clusters[-1]["mean"].items()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(run=a.run, episodes=a.episodes, window=WINDOW, windows=len(x),
                                           overall={n: float(v) for n, v in zip(NAMES, mu)}, clusters=clusters),
                                      indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
