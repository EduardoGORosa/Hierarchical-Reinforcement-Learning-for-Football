"""Evaluation that can support a claim: distinct game seeds, then honest intervals.

Session `hierarchical-rl-football`, nodes 45-47 (layer 7 of the build map). Sources: tcc_brief.md section 4
(trap 1: a fixed game_engine_random_seed replays ONE episode forever); Agarwal
et al. 2021, "Deep RL at the Edge of the Statistical Precipice" (interquartile
mean and bootstrap intervals for few-run results).

Two levels of seed, never to be confused:
  - TRAINING seed: which of the 5 runs (network init, sampling). The thing whose
    spread an interval describes.
  - GAME seed: which episode the engine plays. Evaluation sweeps a fixed list
    (EVAL_SEEDS), identical for every run and every arm.

Checks:  python3 -m hrl.check 45   (... 47). Node 45's tests use the toy pitch, so
they need gymnasium: run them in the container.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from hrl.hints import Policy

if TYPE_CHECKING:          # gymnasium is container-only; annotations don't need it at run time
    import gymnasium

EVAL_SEEDS = tuple(range(1000, 1100))


# --- provided: an env whose episodes differ by seed and nothing else ---------

def make_eval_env(level: str, game_seed: int, representation: str = "simple115v2") -> gymnasium.Env:
    """A GFootballEnv that plays the episode `game_seed` picks (container only).

    reverse_team_processing is pinned so the seed varies the noise only
    (brief, trap 3).
    """
    from gfootball_env import GFootballEnv
    return GFootballEnv(
        env_name=level, representation=representation, rewards="scoring",
        other_config_options={"game_engine_random_seed": game_seed,
                              "reverse_team_processing": False},
    )


# --- node 45 ------------------------------------------------------------------

def evaluate_policy(policy: Policy, make_env: Callable[[int], gymnasium.Env],
                    seeds: Sequence[int] = EVAL_SEEDS) -> float:
    """Win rate of `policy` (obs -> action) over one episode per game seed.

    make_env(seed) builds the env for one seed. For GRF, bind the level first, e.g.
    functools.partial(make_eval_env, "academy_3_vs_1_with_keeper"). The tests pass the toy
    pitch instead: taking the builder as an argument is what makes this testable without the
    engine. A win is an episode whose total reward is > 0. Close each env.
    """
    # TODO(node 45): for each seed: env = make_env(seed), reset, step until terminated or
    # truncated, record win or not, close. Return the fraction won.
    # A policy trained with SB3 is `lambda obs: int(model.predict(obs, deterministic=True)[0])`.
    raise NotImplementedError("TODO(node 45)")


# --- node 46: from 5 numbers to a claim --------------------------------------

def iqm(scores: Sequence[float] | np.ndarray) -> float:
    """Interquartile mean: drop the lowest and highest quarter, average the rest.

    For n values, drop int(0.25 * n) from each end (what scipy.stats.trim_mean
    does, which rliable uses). With 5 runs that is one from each end.
    """
    # TODO(node 46)
    raise NotImplementedError("TODO(node 46)")


def bootstrap_ci(scores: Sequence[float] | np.ndarray, stat: Callable[[np.ndarray], float] = np.mean, n_boot: int = 10_000,
                 level: float = 0.95, rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Percentile bootstrap interval for stat(scores): resample the runs WITH
    replacement n_boot times, compute stat on each resample, and return the
    (lower, upper) percentiles that hold `level` of them.
    """
    # TODO(node 46): rng.choice(scores, size=(n_boot, n)), apply stat along
    # axis 1 (np.apply_along_axis, or a loop), then np.percentile.
    raise NotImplementedError("TODO(node 46)")


# --- node 47: a possession value as a potential --------------------------------

def shaped_reward(r: float, phi_s: float, phi_next: float, gamma: float, done: bool) -> float:
    """Reward plus potential-based shaping, F = gamma * Phi(s') - Phi(s) (Ng, Harada & Russell
    1999). Phi is a possession value, e.g. an EPV model fitted inside GRF (brief, section 5).
    At the end of an episode there is no next state, so Phi(s') counts as 0.
    """
    # TODO(node 47): one line, with one conditional.
    raise NotImplementedError("TODO(node 47)")
