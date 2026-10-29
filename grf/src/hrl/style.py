"""Steering team style through the top layer's intent probabilities.

Session `hierarchical-rl-football`, nodes 38-44 (layer 6 of the build map). Design: thesis_ideas.md section
3.2 and docs/steering_team_style.html.

A style is a fixed preference over the six intents. At each top-layer decision:

    new p(intent)  proportional to  trained p(intent) x preference(intent)^lambda

which in code is one line before the softmax: logits + lambda * log(preference).
lambda = 0 changes nothing, and a preference that is equal for every intent
changes nothing at any lambda.

Option order everywhere: hrl.options.SUBGOALS. The style tilts mu, the high-level
policy; the option policies are not touched.

Checks:  python3 -m hrl.check 38   (... 44)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# Worked example from thesis_ideas.md 3.2 (invented numbers, supplied):
# left wing, centre, right wing, enter box, shoot, keep possession.
EXAMPLE_TRAINED = np.array([0.15, 0.10, 0.20, 0.25, 0.20, 0.10])
WING_PLAY = np.array([0.30, 0.10, 0.30, 0.10, 0.10, 0.10])


# --- node 38: softmax ---------------------------------------------------------

def softmax(logits: Sequence[float] | np.ndarray) -> np.ndarray:
    """Turn a vector of scores into probabilities that sum to 1.

    Subtract the largest score first. Node 33 shows why that changes nothing
    mathematically, and why it stops exp() overflowing for large scores.
    """
    z = np.asarray(logits, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


# --- node 39: the style rule --------------------------------------------------

def styled_logits(logits: Sequence[float] | np.ndarray, preference: Sequence[float] | np.ndarray, lam: float) -> np.ndarray:
    """The top layer's scores after applying a style of strength lam."""
    return np.asarray(logits, dtype=float) + lam * np.log(np.asarray(preference, dtype=float))


def styled_probs(trained_probs: Sequence[float] | np.ndarray, preference: Sequence[float] | np.ndarray, lam: float) -> np.ndarray:
    """Same rule, starting from probabilities instead of scores. Provided."""
    return softmax(styled_logits(np.log(trained_probs), preference, lam))


# --- node 40: KL divergence ---------------------------------------------------

def kl(q: Sequence[float] | np.ndarray, p: Sequence[float] | np.ndarray) -> float:
    """KL(q || p) = sum over intents of q * ln(q / p). Zero exactly when q == p.

    Read it as: how surprised, on average, someone who expected p is by play
    that actually follows q.
    """
    q, p = np.asarray(q, dtype=float), np.asarray(p, dtype=float)
    return float(np.sum(q * np.log(q / p)))


def style_score(q: Sequence[float] | np.ndarray, preference: Sequence[float] | np.ndarray) -> float:
    """How much of the style q shows: the average of ln(preference) under q. Provided."""
    return float(np.asarray(q) @ np.log(preference))


def share_of_choices_changed(q: Sequence[float] | np.ndarray, p: Sequence[float] | np.ndarray) -> float:
    """Total variation: of every 100 intent choices, how many the style moves. Provided."""
    return 0.5 * float(np.abs(np.asarray(q) - np.asarray(p)).sum())


# --- node 44: what PPO must store --------------------------------------------

def styled_log_prob(
    logits: Sequence[float] | np.ndarray, preference: Sequence[float] | np.ndarray, lam: float, intent: int
) -> float:
    """ln of the probability the STYLED top layer gave to the intent it chose.

    If training runs with a style switched on (Option B), this, not the
    unstyled log-prob, is what the rollout buffer must hold, and what the ratio
    in ppo_clip must be computed from. Node 42 is why.
    """
    # TODO(node 44): one line, from styled_logits and softmax.
    raise NotImplementedError("TODO(node 44)")
