"""Does each option do its own job, and does pi act differently per z? The acceptance test of
an option set (node 37b; the evidence in thesis_ideas 3.4c came from an earlier version).

    ./run.sh hrl diagnose /work/runs/<phase-options run> [--episodes 200]

It reads the run's config.json (level, option set, termination) and final_model.zip, then:
  1. per option z: plays whole episodes with z fixed, restarting the same z after every ending
     (like options.replay_option). How its runs end, its subgoals reached per episode, goals.
  2. z-sensitivity: on states pi visited, pi(. | s, z) for every z. The mean over states of the
     largest total-variation distance (TV) between two z's (0 = identical, 1 = no action in
     common), and, for reference, how often the most likely action changes with z. The TV is the
     one accepted on: a near-uniform pi flips its argmax on tiny differences (node 37b).
Same episode seeds for every z, so the options face the same games.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # grf/src

from hrl.hints import DiagnosticReport, OptionRow  # noqa: E402

MIN_REACHED = 0.05  # each option's own subgoal, as a fraction of its runs. Random play: <= 0.04 (node 37a)
MIN_MAX_TV = (
    0.25  # mean max TV between two z's. The collapsed zone run: 0.05 (node 37b)
)
KEEP_EVERY = 5  # keep every 5th visited state for part 2
MAX_STATES = 2000


def _config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def _env(cfg: dict):
    if cfg["level"] == "toy":
        from hrl.toy import KnownAnswerPitch

        return KnownAnswerPitch(x0=0.0)
    from gfootball_env import GFootballEnv

    return GFootballEnv(
        env_name=cfg["level"],
        representation=cfg["representation"],
        rewards=cfg["rewards"],
    )


def _load(run_dir: Path):
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    return PPO.load(str(run_dir / "final_model.zip"), device="cpu")


def play_option(run_dir: Path, z: int, episodes: int, seed: int) -> dict:
    """Part 1 for one z. Returns the row's counts and a sample of the states visited."""
    from hrl.execution import one_hot, raw_frame
    from hrl.options import OptionStart, termination, use_option_set

    cfg = _config(run_dir)
    use_option_set(cfg.get("option_set", "zones"))
    from hrl.options import OPTIONS

    model, env, rng = _load(run_dir), _env(cfg), np.random.default_rng(seed + z)
    ends, reached, goals, states = Counter(), [], 0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        raw, t, done, n_reached, ret = raw_frame(env), 0, False, 0, 0.0
        start = OptionStart(z, 0, float(raw["ball"][0]), raw)
        while not done:
            if t % KEEP_EVERY == 0:
                states.append(obs)
            action = int(
                model.predict(np.concatenate([obs, one_hot(z)]), deterministic=False)[0]
            )
            prev = raw
            obs, r, term, trunc, _ = env.step(action)
            raw, t, done, ret = raw_frame(env), t + 1, term or trunc, ret + r
            end = termination(
                start, prev, raw, t, done, action, cfg.get("termination", "clock"), rng
            )
            if end:
                ends[end] += 1
                n_reached += end == "reached"
                start = OptionStart(z, t, float(raw["ball"][0]), raw)
        reached.append(n_reached)
        goals += ret > 0
    env.close()
    n = max(1, sum(ends.values()))
    row = OptionRow(
        name=OPTIONS[z].name,
        reached=ends["reached"] / n,
        failed=ends["failed"] / n,
        timeout=ends["timeout"] / n,
        reached_per_episode=float(np.mean(reached)),
        goal_rate=goals / episodes,
    )
    return dict(row=row, states=states)


def _play(args: tuple) -> dict:
    return play_option(*args)


def action_probs(model, obs: np.ndarray) -> np.ndarray:
    """pi's action probabilities for a batch of observations (one_hot(z) already appended)."""
    import torch

    with torch.no_grad():
        dist = model.policy.get_distribution(torch.as_tensor(obs, dtype=torch.float32))
        return dist.distribution.probs.numpy()


def z_sensitivity(model, states: np.ndarray, n_options: int) -> tuple[float, float]:
    """(fraction of states where the argmax action changes with z, mean max TV across z)."""
    from hrl.execution import one_hot

    probs = np.stack(
        [
            action_probs(
                model,
                np.concatenate(
                    [states, np.tile(one_hot(z, n_options), (len(states), 1))], axis=1
                ),
            )
            for z in range(n_options)
        ],
        1,
    )
    argmax = probs.argmax(2)  # (states, z)
    changes = float(np.mean((argmax != argmax[:, :1]).any(1)))
    tv = 0.5 * np.abs(probs[:, :, None, :] - probs[:, None, :, :]).sum(
        3
    )  # (states, z, z)
    return changes, float(tv.max((1, 2)).mean())


def accepted(report: DiagnosticReport) -> bool:
    """Is this option set good enough to train mu on?"""
    every_option_reaches = all(
        row["reached"] >= MIN_REACHED for row in report["options"]
    )
    pi_reads_z = report["max_tv"] >= MIN_MAX_TV
    return every_option_reaches and pi_reads_z


def diagnose(
    run_dir: Path, episodes: int = 200, seed: int = 10_000
) -> DiagnosticReport:
    from hrl.options import use_option_set

    cfg = _config(run_dir)
    use_option_set(cfg.get("option_set", "zones"))
    from hrl.options import OPTIONS

    jobs = [(run_dir, z, episodes, seed) for z in range(len(OPTIONS))]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        played = pool.map(_play, jobs)
    states = np.asarray([s for p in played for s in p["states"]], dtype=np.float32)
    keep = np.random.default_rng(0).permutation(len(states))[:MAX_STATES]
    change, tv = z_sensitivity(_load(run_dir), states[keep], len(OPTIONS))
    return DiagnosticReport(
        option_set=cfg.get("option_set", "zones"),
        episodes=episodes,
        options=[p["row"] for p in played],
        argmax_change=change,
        max_tv=tv,
        n_states=len(keep),
    )


def show(report: DiagnosticReport) -> None:
    print(
        f"\noption set: {report['option_set']}   {report['episodes']} episodes per option\n"
    )
    print(
        f"  {'option':10s} {'reached':>8s} {'failed':>7s} {'timeout':>8s} {'reached/ep':>11s} {'goals':>6s}"
    )
    for r in report["options"]:
        print(
            f"  {r['name']:10s} {r['reached']:8.2f} {r['failed']:7.2f} {r['timeout']:8.2f}"
            f" {r['reached_per_episode']:11.2f} {r['goal_rate']:6.2f}"
        )
    print(
        f"\n  z-sensitivity over {report['n_states']} states: mean max TV {report['max_tv']:.2f}"
        f"   (argmax changes with z on {report['argmax_change']:.0%} of states)"
    )
    try:
        verdict = "yes" if accepted(report) else "no"
    except NotImplementedError as e:
        verdict = f"(TODO: {e})"
    print(
        f"  accepted: {verdict}   (needs reached >= {MIN_REACHED} for every option, "
        f"mean max TV >= {MIN_MAX_TV})"
    )


def main(argv: list[str]) -> int:
    run_dir = Path(argv[0])
    episodes = int(argv[argv.index("--episodes") + 1]) if "--episodes" in argv else 200
    report = diagnose(run_dir, episodes)
    show(report)
    (run_dir / "diagnose.json").write_text(json.dumps(report, indent=1))
    print(f"\n  wrote {run_dir / 'diagnose.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
