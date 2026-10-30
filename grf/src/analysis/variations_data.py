"""Collect every number of the 2026-10-29 quick tests into one JSON for the report page.

    ./run.sh hrl ../analysis/variations_data      (or: python3 /work/src/analysis/variations_data.py in the container)

Reads runs/variations/{results.jsonl, plateau.json, discover.json, style_profiles.json}, the runs'
monitor.csv files and, for mu runs, the hrl/env_steps scalar (mu's own counter counts decisions).
Writes runs/variations/report_data.json. Curves are goal rate per 100k env (game) steps, binned.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

RUNS = Path("/work/runs")
V = RUNS / "variations"
BIN = 100_000


def monitor(run: str) -> dict[str, np.ndarray]:
    rows = list(csv.reader(open(RUNS / run / "monitor.csv")))
    h = next(i for i, r in enumerate(rows) if r and r[0] == "r")
    cols = rows[h]
    data = np.asarray([[float(v) for v in r] for r in rows[h + 1:] if r and len(r) == len(cols)])
    return {c: data[:, i] for i, c in enumerate(cols)}


def binned(steps: np.ndarray, goal: np.ndarray, min_n: int = 30) -> list[list[float]]:
    idx = (steps // BIN).astype(int)
    return [[float((b + 0.5) * BIN), float(goal[idx == b].mean())] for b in np.unique(idx) if (idx == b).sum() >= min_n]


def flat_curve(run: str, agents: int = 1, offset: float = 0.0):
    if not (RUNS / run / "monitor.csv").exists():
        return None
    m = monitor(run)
    goal = (m["episode_goal"] if "episode_goal" in m else m["r"]) > 0
    # multi-agent: each game's episode appears once per agent, so game steps = cumsum(l) / agents
    return binned(offset + np.cumsum(m["l"]) / agents, goal)


def mu_curve(run: str, offset: float = 0.0):
    if not (RUNS / run / "monitor.csv").exists():
        return None
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    pairs = []
    for f in sorted((RUNS / run).glob("tb/*/events.out.tfevents.*")):
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
        ea.Reload()
        if "hrl/env_steps" in ea.Tags()["scalars"]:
            pairs += [(e.step, e.value) for e in ea.Scalars("hrl/env_steps")]
    if not pairs:
        return None
    dec, env = np.asarray(sorted(pairs)).T
    m = monitor(run)
    goal = m["episode_env_reward"] if "episode_env_reward" in m else m["r"]  # older runs: no cost, reward = goals
    return binned(offset + np.interp(np.cumsum(m["l"]), dec, env), goal > 0)


def load(name: str):
    p = V / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    results = {}
    if (V / "results.jsonl").exists():
        for line in (V / "results.jsonl").read_text().splitlines():
            r = json.loads(line)
            results[r.get("label") or r["agent"]] = r
    diag = {r: load(f"../{r}/diagnose.json") for r in ("p3_opt_s0", "v2_targ_opt_s0", "v4_ma_opt_s0", "v6_joint_opt_s0")}
    curves = {
        "flat_3v1": flat_curve("ov_flat_s0"),
        "flat_cf": flat_curve("v1_cf_flat_train_s0"),
        "flat_bonus01": flat_curve("v5_bonus01_s0"),
        "flat_bonus10": flat_curve("v5_bonus10_s0"),
        "ma_flat": flat_curve("v4_ma_flat_s0", agents=3),
        "ma_mu": mu_curve("v4_ma_mu_s0", offset=2_000_000),
        "hrl_3v1": mu_curve("ov_mu_s0", offset=2_000_000),
        "targ_mu": mu_curve("v2_targ_mu_s0", offset=2_000_000),
        "tr_mu": mu_curve("v4_tr_mu_s0"),
        "tr_flat": flat_curve("v4_tr_flat_s0"),
        "v6_ctrl": mu_curve("v6_ctrl_s0"), "v6_hient": mu_curve("v6_hient_s0"), "v6_joint": mu_curve("v6_joint_mu_s0"),
    }
    out = dict(results=results, curves={k: v for k, v in curves.items() if v}, diagnose={k: v for k, v in diag.items() if v},
               plateau=load("plateau.json"), discover=load("discover.json"), styles=load("style_profiles.json"))
    (V / "report_data.json").write_text(json.dumps(out))
    print(f"results {len(results)}  curves {sorted(out['curves'])}  diagnose {sorted(out['diagnose'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
