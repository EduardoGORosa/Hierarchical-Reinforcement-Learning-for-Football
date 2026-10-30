"""Evaluate any frozen agent on any scenario, with match statistics. Shared by the 2026-10-29 quick tests.

    ./run.sh hrl evaluate_runs --agent flat:ov10_flat_s0 --level cf_3_vs_1_with_keeper --episodes 600
    ./run.sh hrl evaluate_runs --agent hrl:m10_mu_s0:p3_opt_s0 --tilt 2:0.8,0.1,0.1
    ./run.sh hrl evaluate_runs --agent const:p3_opt_s0:1          # "always dribble": z = 1 at every decision
    ./run.sh hrl evaluate_runs --agent ma_flat:ma_flat_s0 --level ma_3_vs_1_with_keeper

Agents (run names under /work/runs):
  flat:<run>                 a train_ppo model on the single-agent env
  hrl:<mu run>:<options run> mu over frozen options, rebuilt from the mu run's config.json
  const:<options run>:<z>    the same frozen options with mu replaced by "always z": the known-answer
                             baseline any trained mu has to beat
  ma_flat:<run>              a multi-agent (shared-weights) flat model, multiagent.py
  ma_hrl:<mu run>:<options run>  team mu over multi-agent options

Every worker has its own engine seed (seed * 100 + rank), so the 8 workers are independent samples
(one game process alone is close to one sample). Match statistics are counted per env step by
MatchStats, under every wrapper, so flat and hierarchical agents are measured the same way.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # grf/src

from hrl.match_stats import MatchStats  # noqa: E402

RUNS = Path("/work/runs")


def _base(level: str):
    from gfootball_env import GFootballEnv
    return MatchStats(GFootballEnv(env_name=level, representation="simple115v2", rewards="scoring"))


def make_env(spec: list[str], level: str, seed: int):
    def _init():
        sys.path.insert(0, "/work/src")
        kind = spec[0]
        if kind == "flat":
            env = _base(level)
        elif kind in ("hrl", "const"):
            from stable_baselines3 import PPO
            from hrl.execution import SMDPEnv, StyleConditionedEnv
            from hrl.options import use_initiation, use_option_set
            opt_run = spec[2] if kind == "hrl" else spec[1]
            ocfg = json.loads((RUNS / opt_run / "config.json").read_text())
            mcfg = json.loads((RUNS / spec[1] / "config.json").read_text()) if kind == "hrl" else {}
            use_option_set(ocfg.get("option_set", "actions"))
            use_initiation(bool(mcfg.get("initiation", False)))
            pi = PPO.load(str(RUNS / opt_run / "final_model.zip"), device="cpu")
            env = SMDPEnv(_base(level), lambda o: int(pi.predict(o, deterministic=False)[0]), 0.993,
                          bool(mcfg.get("with_primitives", False)), mcfg.get("termination", "clock"),
                          rng=np.random.default_rng(seed))
            if mcfg.get("style_input"):
                env = StyleConditionedEnv(env, mcfg.get("style_bonus", 0.3), mcfg.get("style_alpha", 1.0))
                env.set_style(np.ones(3))
        elif kind == "ma_flat":
            from hrl.multiagent import MultiAgentGame
            cfg = json.loads((RUNS / spec[1] / "config.json").read_text())
            env = MatchStats(MultiAgentGame(level, cfg.get("agents", 3)))
        elif kind == "ma_hrl":
            from stable_baselines3 import PPO
            from hrl.options import use_option_set
            ocfg = json.loads((RUNS / spec[2] / "config.json").read_text())
            use_option_set(ocfg.get("option_set", "actions"))
            from hrl.multiagent import MultiAgentGame, TeamSMDPEnv
            pi = PPO.load(str(RUNS / spec[2] / "final_model.zip"), device="cpu")
            env = TeamSMDPEnv(MatchStats(MultiAgentGame(level, ocfg.get("agents", 3))),
                              lambda o: pi.predict(o, deterministic=False)[0], 0.993)
        else:
            raise ValueError(kind)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(spec: list[str], level: str, episodes: int, seed: int, tilt=None, n_envs: int = 8) -> dict:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from hrl.options import OPTIONS, use_option_set

    kind = spec[0]
    if kind in ("hrl", "const", "ma_hrl"):
        opt_run = spec[1] if kind == "const" else spec[2]
        use_option_set(json.loads((RUNS / opt_run / "config.json").read_text()).get("option_set", "actions"))
        from hrl.options import OPTIONS
    model = None if kind == "const" else PPO.load(str(RUNS / spec[1] / "final_model.zip"), device="cpu")
    torch.manual_seed(seed)  # after load, which reseeds torch with the training seed
    venv = SubprocVecEnv([make_env(spec, level, seed * 100 + i) for i in range(n_envs)], start_method="spawn")
    obs = venv.reset()
    rows, picks, steps = [], Counter(), Counter()
    ma = kind == "ma_flat"
    while len(rows) < episodes:
        if kind == "const":
            act = np.full(n_envs, int(spec[2]))
        elif ma:  # every agent of every game, one shared policy
            from hrl.multiagent import with_ids
            flat_obs = np.concatenate([with_ids(o) for o in obs], axis=0)
            act = model.predict(flat_obs, deterministic=False)[0].reshape(obs.shape[0], -1)
        elif tilt is None:
            act = model.predict(obs, deterministic=False)[0]
        else:  # test-time tilt (node 39): + lam * ln(w) on the option logits
            with torch.no_grad():
                logits = model.policy.get_distribution(torch.as_tensor(obs, dtype=torch.float32)).distribution.logits
                add = torch.zeros(logits.shape[1])
                add[:len(tilt[1])] = tilt[0] * torch.log(torch.tensor(tilt[1]))
                act = torch.distributions.Categorical(logits=logits + add).sample().numpy()
        obs, r, done, infos = venv.step(act)
        for i, info in enumerate(infos):
            if kind in ("hrl", "const", "ma_hrl"):
                a = int(np.asarray(act[i]).ravel()[0])
                name = OPTIONS[a].name if a < len(OPTIONS) else "primitive"
                picks[name] += 1
                steps[name] += info.get("tau", 1)
            if done[i] and "match" in info and len(rows) < episodes:
                rows.append(info["match"])
    venv.close()
    n_dec, n_st = sum(picks.values()), sum(steps.values())
    mean = lambda k: float(np.mean([m[k] for m in rows]))  # noqa: E731
    return dict(
        agent=":".join(spec), level=level, seed=seed, episodes=len(rows), tilt=list(tilt) if tilt else None,
        goal_rate=mean("goal"), steps=mean("steps"), passes=mean("passes"), shots=mean("shots"),
        carries=mean("carries"), mate_early_vx=mean("mate_early_vx"), mate_ahead=mean("mate_ahead"),
        lanes={k: float(np.mean([m["lanes"][k] for m in rows])) for k in ("left", "centre", "right")},
        decision_share={k: v / n_dec for k, v in picks.items()} if n_dec else {},
        env_step_share={k: v / n_st for k, v in steps.items()} if n_st else {})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--level", default="academy_3_vs_1_with_keeper")
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tilt", default=None, help="lam:w1,w2,... (hrl only)")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--out", default="/work/runs/variations/results.jsonl")
    p.add_argument("--label", default="")
    a = p.parse_args()
    tilt = None
    if a.tilt:
        lam, w = a.tilt.split(":")
        tilt = (float(lam), [float(v) for v in w.split(",")])
    row = evaluate(a.agent.split(":"), a.level, a.episodes, a.seed, tilt, a.n_envs)
    row["label"] = a.label
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
