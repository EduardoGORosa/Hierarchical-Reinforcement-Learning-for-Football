"""Policy-gradient basics, on the exact example the teaching nodes use.

Session `hierarchical-rl-football`, nodes 23, 24, 22, 25, 26 (layer 1 of the
build map). Every number a test expects is one you can reproduce by hand from these facts:

    two actions, a1 (index 0) and a2 (index 1)
    p = pi_theta(a1) = 1 / (1 + e^(-theta))      pi_theta(a2) = 1 - p
    R(a1) = 10, R(a2) = 0                         (supplied, not derived)
    ONE action per episode: an episode is one draw and one reward
    J(theta) = 10 p                               (the objective: average reward)

The point is not the example. It is that every function here reappears, under
the same name, when the agent is a football team: `episode_writes` becomes one
row of the PPO loss, `reweighted_gradient` becomes the ratio inside it, and
`ppo_clip` is the loss itself.

Tests live in hrl/tests/test_n23_*.py ... test_n26_*.py. From grf/src (host is
fine, numpy only):

    python3 -m hrl.check 24          # one node: its unit tests, then its stack test
    python3 -m hrl.check --upto 26   # the whole layer, in teaching order
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


REWARD = np.array([10.0, 0.0])


# --- provided ---------------------------------------------------------------


def probs(theta: float) -> np.ndarray:
    """(pi_theta(a1), pi_theta(a2)) for the logistic two-action policy."""
    p = 1.0 / (1.0 + np.exp(-theta))
    return np.array([p, 1.0 - p])


def objective(theta: float) -> float:
    """J(theta) = sum over actions of probability x reward = 10 p."""
    return float(probs(theta) @ REWARD)


def sample_batch(theta: float, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Play n one-action episodes at theta. Returns (actions, rewards)."""
    actions = rng.choice(2, size=n, p=probs(theta))
    return actions, REWARD[actions]


# --- node 23: what the log-gradient IS --------------------------------------


def grad_log_prob_by_nudge(theta: float, action: int, h: float = 1e-6) -> float:
    """Slope of ln pi_theta(action) in theta, measured by a nudge of size h.

    This is Check 64 as code: nudge theta, see how far ln pi moved, divide by h.
    """
    return (np.log(probs(theta + h)[action]) - np.log(probs(theta)[action])) / h


# --- node 24: what ONE episode writes down -----------------------------------


def episode_writes(theta: float, action: int, reward: float) -> float:
    """The number one episode contributes to the gradient estimate at theta.

    Node 24 compares three candidates: R, pi*R and R * grad ln pi. Only one of
    them averages to dJ/dtheta. Put that one here.
    """
    return reward * grad_log_prob_by_nudge(theta, action)


def batch_gradient(theta: float, actions: Sequence[int] | np.ndarray, rewards: Sequence[float] | np.ndarray) -> float:
    """Estimate dJ/dtheta from a batch that was played AT theta (on-policy)."""
    batch_grad = 0
    c = 0
    for action, reward in zip(actions, rewards):
        batch_grad += episode_writes(theta, action, reward)
        c += 1

    return batch_grad / c


# --- node 22: data from an older policy --------------------------------------


def reweighted_gradient(
    theta: float, theta_old: float, actions: Sequence[int] | np.ndarray, rewards: Sequence[float] | np.ndarray
) -> float:
    """Estimate dJ/dtheta at theta from a batch that was played at theta_old.

    Averaging episode_writes over this batch gives the wrong quantity (the
    check shows it: 1.345 instead of 1.966). Each episode's contribution needs
    one extra factor. Which one, and why, is node 22.
    """
    batch_grad = 0
    c = 0
    for action, reward in zip(actions, rewards):
        batch_grad += (
            probs(theta)[action] / probs(theta_old)[action]
        ) * episode_writes(theta, action, reward)
        c += 1

    return batch_grad / c


# --- node 25: clipping, which is PPO -----------------------------------------


def ppo_clip(ratio: float | np.ndarray, advantage: float | np.ndarray, eps: float = 0.2) -> np.ndarray | float:
    """PPO's clipped surrogate for one sample (Schulman et al. 2017, eq. 7):

        min( ratio * A,  clip(ratio, 1 - eps, 1 + eps) * A )

    Works element-wise on arrays. eps = 0.2 is the paper's continuous-control
    setting. The GRF baseline in train_ppo.py uses 0.08.
    """
    return np.minimum(ratio * advantage, np.clip(ratio, 1 - eps, 1 + eps) * advantage)


def surrogate(
    theta: float, theta_old: float, actions: Sequence[int] | np.ndarray, rewards: Sequence[float] | np.ndarray, eps: float = 0.2
) -> float:
    """PPO's objective for one batch: the average of ppo_clip(ratio, R) over its episodes.
    Provided. PPO never computes a gradient formula by hand: it builds this ONE number and lets
    the slope of it (backprop in a network, a nudge here) do the rest. Here the reward R plays
    the advantage's role; node 26 replaces it with R minus a baseline.
    """
    ratio = probs(theta)[actions] / probs(theta_old)[actions]
    return float(np.mean(ppo_clip(ratio, np.asarray(rewards, dtype=float), eps)))


# --- node 26: a baseline -----------------------------------------------------


def episode_writes_with_baseline(
    theta: float, action: int, reward: float, baseline: float
) -> float:
    """episode_writes, with a fixed number `baseline` subtracted from the reward."""

    advantage = reward - baseline
    return advantage * grad_log_prob_by_nudge(theta, action)


def exact_mean_and_variance(theta: float, baseline: float) -> tuple[float, float]:
    """Mean and variance of episode_writes_with_baseline over one draw, EXACTLY.

    No sampling: there are only two possible episodes, so enumerate them and
    weight each by its probability.
    """
    p = probs(theta)
    x = np.array(
        [episode_writes_with_baseline(theta, a, REWARD[a], baseline) for a in (0, 1)]
    )
    mean = float(p @ x)
    return mean, float(p @ (x - mean) ** 2)
