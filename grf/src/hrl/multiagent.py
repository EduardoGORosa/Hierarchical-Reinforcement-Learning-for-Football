"""Three learning agents instead of one controlled player plus two AI teammates (variation 4).

GRF can bind N agents to N players (number_of_left_players_agent_controls=N). The scenario
ma_3_vs_1_with_keeper (grf/scenarios/) makes our keeper uncontrollable, so the three agents are
the three attackers for the whole episode: no AI teammates left, only the AI keeper.

Pieces, each usable alone:
  MultiAgentGame      one game, observation (N, 115) simple115v2 per agent, action MultiDiscrete([19]*N),
                      one team reward (scoring is the same number for every agent)
  AgentsAsSlots       a VecEnv over G games that shows SB3 G*N single-agent slots. Every slot is one
                      agent, with one_hot(agent id) appended, so ONE PPO network is shared by all
                      agents: IPPO with parameter sharing, the strongest family on 3v1 in Song et
                      al. 2023. SB3's num_timesteps then counts agent-steps: divide by N for game steps.
  TeamOptionLearningEnv  phase options for the team: one z for the whole team (thesis_ideas 3.4, a
                      captain's call), appended to every agent's row. The subgoals are already ball
                      events (pass, carry past the defender, shot), so options.termination works on
                      the team's raw frame unchanged
  TeamSMDPEnv         phase mu: one decision z for the team; all N agents then act through the
                      frozen multi-agent option policy until beta. Single-agent interface, so
                      train_hrl's SMDP buffer is reused as it is

Slot observation layout, the same everywhere: [simple115v2 of agent i, one_hot(z) if options, one_hot(i)].
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from hrl.execution import combine_rewards, one_hot
from hrl.options import OptionStart, option_reward, termination


def with_ids(obs: np.ndarray) -> np.ndarray:
    """(N, d) -> (N, d + N): agent i's row gets one_hot(i)."""
    n = obs.shape[0]
    return np.concatenate([obs, np.eye(n, dtype=np.float32)], axis=1)


class MultiAgentGame(gymnasium.Env):
    """One GRF game with `agents` learning agents. Same seeding rule as GFootballEnv."""

    metadata = {"render_modes": []}

    def __init__(self, level: str = "ma_3_vs_1_with_keeper", agents: int = 3, rewards: str = "scoring"):
        import gfootball.env as football_env
        self.n = agents
        self._env = football_env.create_environment(
            env_name=level, representation="simple115v2", rewards=rewards,
            number_of_left_players_agent_controls=agents)
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (agents, 115), np.float32)
        self.action_space = gymnasium.spaces.MultiDiscrete([19] * agents)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._env.unwrapped._config["game_engine_random_seed"] = int(self.np_random.integers(0, 2_000_000_000))
        return np.asarray(self._env.reset(), dtype=np.float32).reshape(self.n, 115), {}

    def step(self, action):
        obs, reward, done, info = self._env.step([int(a) for a in np.asarray(action).ravel()])
        r = float(np.asarray(reward).ravel()[0])
        return np.asarray(obs, dtype=np.float32).reshape(self.n, 115), r, bool(done), False, info

    def raw_observation(self) -> dict:
        """The raw frame from agent 0's side. Ball, teams and owner are the same for every agent."""
        return self._env.unwrapped.observation()[0]

    def close(self):
        self._env.close()


class TeamOptionLearningEnv(gymnasium.Wrapper):
    """OptionLearningEnv for a team: z uniform at reset and after every beta, shared by all agents.
    Observation (N, 115 + K); reward combine_rewards(team reward, option reward); info option_end."""

    def __init__(self, env: MultiAgentGame, mix="both", weight=1.0, rng=None):
        super().__init__(env)
        from hrl.execution import N_OPTIONS
        self.k = N_OPTIONS
        self.mix, self.weight = mix, weight
        self.rng = rng or np.random.default_rng()
        n = env.n
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (n, 115 + self.k), np.float32)
        self.log: list[tuple[str, str, int]] = []

    def _raw(self):
        return self.env.unwrapped.raw_observation()

    def _start(self, raw):
        self.start = OptionStart(int(self.rng.integers(self.k)), self.t, float(raw["ball"][0]), raw)

    def _obs(self, obs):
        return np.concatenate([obs, np.tile(one_hot(self.start.z, self.k), (obs.shape[0], 1))], axis=1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.t = 0
        self._start(self._raw())
        return self._obs(obs), info

    def step(self, action):
        prev = self._raw()
        obs, r_env, term, trunc, info = self.env.step(action)
        self.t += 1
        raw = self._raw()
        end = termination(self.start, prev, raw, self.t, term or trunc)
        r_o = option_reward(self.start, prev, raw, self.t)
        z = self.start.z
        if end:
            from hrl.execution import OPTIONS
            self.log.append((OPTIONS[z].name, end, self.t - self.start.step))
            if not (term or trunc):
                self._start(raw)
        info = dict(info, option_end=end)
        return self._obs(obs), combine_rewards(r_env, r_o, self.mix, self.weight), term, trunc, info


class TeamSMDPEnv(gymnasium.Env):
    """Phase mu for the team. Observation: agent 0's simple115v2 (it holds every player's position).
    Action z; then every agent acts as pi([obs_i, one_hot(z), one_hot(i)]) until beta.
    Reward: the discounted task reward over the option; info tau, option_end, episode_env_reward."""

    def __init__(self, env, option_policy: Callable[[np.ndarray], np.ndarray], gamma: float, rng=None):
        from hrl.execution import N_OPTIONS
        self.env, self.pi, self.gamma = env, option_policy, gamma
        self.k = N_OPTIONS
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (115,), np.float32)
        self.action_space = gymnasium.spaces.Discrete(N_OPTIONS)

    def _raw(self):
        return self.env.unwrapped.raw_observation()

    def reset(self, *, seed=None, options=None):
        self._obs, info = self.env.reset(seed=seed, options=options)
        self._t, self._ep = 0, 0.0
        return self._obs[0], info

    def step(self, z):
        from hrl.smdp import option_return
        z = int(z)
        start = OptionStart(z, self._t, float(self._raw()["ball"][0]), self._raw())
        rewards, info = [], {}
        while True:
            rows = np.concatenate([self._obs, np.tile(one_hot(z, self.k), (self._obs.shape[0], 1))], axis=1)
            actions = np.asarray(self.pi(with_ids(rows))).ravel()
            prev = self._raw()
            self._obs, r, term, trunc, info = self.env.step(actions)
            self._t += 1
            rewards.append(r)
            reason = termination(start, prev, self._raw(), self._t, term or trunc)
            if reason is not None:
                break
        self._ep += float(sum(rewards))
        info = dict(info, tau=len(rewards), option_end=reason, env_reward=float(sum(rewards)),
                    episode_env_reward=self._ep)
        return self._obs[0], option_return(rewards, self.gamma), term, trunc, info

    def close(self):
        self.env.close()


class AgentsAsSlots(VecEnv):
    """A VecEnv of G multi-agent games (any VecEnv whose observations are (N, d)), shown to SB3 as
    G*N single-agent slots with one_hot(agent id) appended. Rewards and dones are the game's, copied
    to each of its agents; infos are copied too (option_end, episode stats)."""

    def __init__(self, games: VecEnv, agents: int):
        self.games, self.n = games, agents
        d = games.observation_space.shape[-1]
        space = gymnasium.spaces.Box(-np.inf, np.inf, (d + agents,), np.float32)
        super().__init__(games.num_envs * agents, space, gymnasium.spaces.Discrete(19))

    def _flat(self, obs):
        return np.concatenate([with_ids(o) for o in obs], axis=0)

    def reset(self):
        return self._flat(self.games.reset())

    def step_async(self, actions):
        self.games.step_async(np.asarray(actions).reshape(self.games.num_envs, self.n))

    def step_wait(self):
        obs, r, done, infos = self.games.step_wait()
        out_infos = []
        for g, info in enumerate(infos):
            for i in range(self.n):
                inf = dict(info)
                if "terminal_observation" in info:
                    inf["terminal_observation"] = with_ids(info["terminal_observation"])[i]
                out_infos.append(inf)
        return self._flat(obs), np.repeat(r, self.n), np.repeat(done, self.n), out_infos

    def close(self):
        self.games.close()

    def get_attr(self, attr_name, indices=None):
        return [v for v in self.games.get_attr(attr_name) for _ in range(self.n)]

    def set_attr(self, attr_name, value, indices=None):
        self.games.set_attr(attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [v for v in self.games.env_method(method_name, *args, **kwargs) for _ in range(self.n)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [v for v in self.games.env_is_wrapped(wrapper_class) for _ in range(self.n)]

    def seed(self, seed=None):
        return [None] * self.num_envs


def make_games(level: str, agents: int, n_games: int, seed: int, wrap: Callable[[MultiAgentGame, int], Any] | None = None):
    """SubprocVecEnv of n_games MultiAgentGames (optionally wrapped), then AgentsAsSlots."""
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def make(rank):
        def _init():
            import sys
            sys.path.insert(0, "/work/src")
            env = MultiAgentGame(level, agents)
            if wrap is not None:
                env = wrap(env, rank)
            env.reset(seed=seed + rank)
            return env
        return _init
    return AgentsAsSlots(SubprocVecEnv([make(i) for i in range(n_games)], start_method="spawn"), agents)
