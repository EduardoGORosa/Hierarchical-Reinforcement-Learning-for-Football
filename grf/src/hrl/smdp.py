"""The arithmetic of a high-level policy whose decisions last a variable number of steps.

Session `hierarchical-rl-football`, nodes 27-29. Sources: Sutton, Precup & Singh
1999 ("MDP + options = SMDP"); Klissarov et al. 2025, sections 3.2.1 (the SMDP,
eq. 13-14), 7.1.1 (call-and-return) and 7.3.1 (model-free learning of mu: the
experience tuple below); Bradtke & Duff 1994 (SMDP learning).

One decision of mu = one option, run by its policy until beta ends it.
Suppose it lasted tau steps and the environment paid r_0, ..., r_{tau-1}. The
high-level policy mu then sees ONE transition (survey 7.1.1):

    (state where the option began, option, discounted sum of the tau rewards,
     state where it ended)

and bootstraps from the end state with gamma^tau, not gamma. Nothing crashes if
you use gamma. The agent just trains towards the wrong goal.

Checks:  python3 -m hrl.check 27   (and 28, 29)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# --- node 27: finite geometric sum (bedrock) ---------------------------------


def geometric_sum(gamma: float, tau: int) -> float:
    """1 + gamma + gamma^2 + ... + gamma^(tau-1), without a loop.

    Watch the gamma == 1 case: the closed form divides by zero there.
    """

    if gamma == 1:
        return tau

    return (1 - gamma**tau) / (1 - gamma)


# --- node 28: one transition of mu -------------------------------------------


def option_return(rewards: Sequence[float] | np.ndarray, gamma: float) -> float:
    """Discounted sum of the rewards collected while ONE option ran: r(s, o) of the
    survey's eq. 14, for one sample.

    rewards[0] arrives on the option's first step and is not discounted.
    """
    op_return = sum([reward * (gamma**i) for i, reward in enumerate(rewards)])
    return op_return


def smdp_target(
    option_reward: float, tau: int, next_value: float, gamma: float, done: bool
) -> float:
    """Target for V_mu(state where the option began).

    The option ran for tau steps. `next_value` is V_mu at the state where it
    ended. If the attack ended (done), there is nothing to bootstrap from.
    """
    return option_reward if done else option_reward + (gamma**tau) * next_value


def flat_gae(
    rewards: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    last_value: float,
    dones: Sequence[float] | np.ndarray,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Ordinary GAE (Schulman et al. 2016), as SB3's RolloutBuffer computes it.

    Provided, as the reference to generalise: every step discounts by gamma.
    dones[t] = 1 means the episode ended at step t. Returns advantages.
    """
    T = len(rewards)
    adv = np.zeros(T)
    next_adv, next_value = 0.0, last_value
    for t in reversed(range(T)):
        live = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * live - values[t]
        next_adv = delta + gamma * lam * live * next_adv
        adv[t] = next_adv
        next_value = values[t]
    return adv


# --- node 29: GAE for mu ----------------------------------------------------


def smdp_gae(
    option_rewards: Sequence[float] | np.ndarray,
    taus: Sequence[int] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    last_value: float,
    dones: Sequence[float] | np.ndarray,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """GAE over decisions of mu, where decision t lasted taus[t] steps.

    Same loop as flat_gae. Only the discount changes: decision t's step to the
    next decision is gamma^taus[t]. Keep lambda per DECISION (not per step), so
    with every tau = 1 this must return exactly what flat_gae returns.
    """
    T = len(option_rewards)
    adv = np.zeros(T)
    next_adv, next_value = 0.0, last_value
    for t in reversed(range(T)):
        live = 1.0 - dones[t]
        delta = option_rewards[t] + (gamma ** taus[t]) * next_value * live - values[t]
        next_adv = delta + (gamma ** taus[t]) * lam * live * next_adv
        adv[t] = next_adv
        next_value = values[t]
    return adv
