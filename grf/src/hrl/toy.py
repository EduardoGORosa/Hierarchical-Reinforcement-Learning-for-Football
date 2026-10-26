"""KnownAnswerPitch: a stand-in for GRF, small enough that every right answer is exact.

Used by the tests of layers 4-7, and by `train_hrl.py --level toy`, so the whole pipeline can
be run and debugged in seconds without the engine. That's the brief's "tier-0 known-answer
control": if something fails here, it isn't GRF's fault.

The rules, all of them:
  - one player, our team, owns the ball from the start at (x0, 0)
  - action 5 ("right") carries the ball +0.02 in x, capped at 1.0
  - action 12 (shot) ends the episode: reward 1 if x > GOAL_X (0.69), else 0. The toy's own
    rule, kept apart from options.BOX_X (GRF's real box) so the known answers never move
  - every other action does nothing
  - the episode also ends after 100 steps, with reward 0
  - observation: (x, y) as float32
  - raw_observation(): the same keys GRF's raw observation uses for the ball, so options.py
    can't tell the difference

Seeds: reset(seed=s) with x0=None starts the ball at x0 = (s % 100) / 100, so a list of seeds
is a list of known starting positions. That's what node 45's test uses.

Needs gymnasium (container). ASK questions about it the same way as about your own code.
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from hrl import options as O
from hrl.hints import RawObs, StepResult

RIGHT = 5
N_ACTIONS = 19
GOAL_X = 0.69          # a shot from x > GOAL_X scores (the toy's rule, not GRF's box)
MAX_STEPS = 100
STEP = 0.02


class KnownAnswerPitch(gymnasium.Env[np.ndarray, int]):

    observation_space: gymnasium.spaces.Box
    """(x, y) of the ball."""
    action_space: gymnasium.spaces.Discrete
    """GRF's 19 actions. Only RIGHT and O.SHOT_ACTION do anything."""
    x0: float | None
    """Where the ball starts. None: from the seed, x0 = (seed % 100) / 100."""
    x: float
    """The ball's x now."""
    t: int
    """Steps since reset. The episode ends at MAX_STEPS."""
    owner: int
    """Who owns the ball: O.OURS until a shot, then O.NOBODY."""
    dir: tuple[float, float]
    """The ball's last movement (dx, dy): (0.05, 0) after a shot, so is_shot can see it."""

    def __init__(self, x0: float | None = 0.0) -> None:
        self.x0 = x0
        self.observation_space = gymnasium.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.action_space = gymnasium.spaces.Discrete(N_ACTIONS)

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        start = self.x0 if self.x0 is not None else ((seed or 0) % 100) / 100
        self.x, self.t, self.owner, self.dir = float(start), 0, O.OURS, (0.0, 0.0)
        return self._obs(), {}

    def raw_observation(self) -> RawObs:
        return {"ball": np.array([self.x, 0.0, 0.1]), "ball_owned_team": self.owner,
                "ball_direction": np.array([*self.dir, 0.0])}

    def step(self, action: int) -> StepResult:
        self.t += 1
        if action == RIGHT:
            self.x = min(self.x + STEP, 1.0)
        if action == O.SHOT_ACTION:
            self.owner, self.dir = O.NOBODY, (0.05, 0.0)
            return self._obs(), float(self.x > GOAL_X), True, False, {}
        return self._obs(), 0.0, self.t >= MAX_STEPS, False, {}

    def _obs(self) -> np.ndarray:
        return np.array([self.x, 0.0], dtype=np.float32)
