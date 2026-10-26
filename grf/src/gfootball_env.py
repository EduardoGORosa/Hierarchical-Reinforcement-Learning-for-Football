"""Gymnasium-compatible wrapper around Google Research Football.

gfootball 2.10.3 targets the legacy `gym` API (<=0.21): `reset()` returns a bare
observation and `step()` returns a 4-tuple. Stable-Baselines3 2.x speaks
Gymnasium (5-tuple `step`, `reset(seed=..., options=...)`).

Rather than patching gfootball, this module keeps both libraries installed side
by side and converts at the boundary. That keeps the vendored engine pristine
and makes the conversion explicit and inspectable.
"""

from __future__ import annotations

import gym as legacy_gym
import gymnasium
import numpy as np
from gymnasium import spaces as gymnasium_spaces

# GRF's core action set. Counted as 19 discrete actions; tallies of 21 come from
# counting the sticky/held direction actions separately.
NUM_ACTIONS = 19


def _to_gymnasium_space(space):
    """Translate a legacy gym space into its Gymnasium equivalent."""
    if isinstance(space, legacy_gym.spaces.Box):
        return gymnasium_spaces.Box(
            low=space.low, high=space.high, shape=space.shape, dtype=space.dtype
        )
    if isinstance(space, legacy_gym.spaces.Discrete):
        return gymnasium_spaces.Discrete(int(space.n))
    if isinstance(space, legacy_gym.spaces.MultiDiscrete):
        return gymnasium_spaces.MultiDiscrete(np.asarray(space.nvec))
    raise TypeError(f"unsupported legacy gym space: {type(space).__name__}")


class GFootballEnv(gymnasium.Env):
    """A single-agent GRF scenario exposed through the Gymnasium API."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_name: str = "academy_empty_goal_close",
        representation: str = "simple115v2",
        rewards: str = "scoring",
        stacked: bool = False,
        logdir: str = "",
        write_goal_dumps: bool = False,
        write_full_episode_dumps: bool = False,
        render: bool = False,
        **kwargs,
    ):
        # Imported lazily so the engine is only loaded inside worker processes.
        import gfootball.env as football_env

        self._env = football_env.create_environment(
            env_name=env_name,
            representation=representation,
            rewards=rewards,
            stacked=stacked,
            logdir=logdir,
            write_goal_dumps=write_goal_dumps,
            write_full_episode_dumps=write_full_episode_dumps,
            render=render,
            number_of_left_players_agent_controls=1,
            **kwargs,
        )
        self.observation_space = _to_gymnasium_space(self._env.observation_space)
        self.action_space = _to_gymnasium_space(self._env.action_space)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None and hasattr(self._env, "seed"):
            self._env.seed(seed)
        obs = self._env.reset()
        return self._cast(obs), {}

    def step(self, action):
        obs, reward, done, info = self._env.step(int(action))
        # GRF signals a single `done`. Scenario time limits are part of the task
        # definition (an academy scenario that runs out of time is a failure, not
        # an arbitrary cutoff), so it maps to `terminated` rather than
        # `truncated`. Worth restating in the methodology chapter: it decides
        # whether the critic bootstraps past the final step.
        return self._cast(obs), float(reward), bool(done), False, info

    def raw_observation(self) -> dict:
        """The engine's raw state for the controlled player (ball, teams, owner,
        score...), whatever `representation` the policy sees. Used by the
        intent tests in hrl/, which need positions, not a feature vector."""
        return self._env.unwrapped.observation()[0]

    def _cast(self, obs):
        return np.asarray(obs, dtype=self.observation_space.dtype)

    def close(self):
        self._env.close()
