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

Slot observation layout: [simple115v2 of agent i, one_hot(z) if options, one_hot(i)], except design A
(--ma-design roles), whose last part is the agent's role instead: one_hot(carrier / near / far).

Two hierarchical designs (2026-10-30), chosen with train_hrl --ma-design:
  team    one team z, agents told apart by id (the first version)
  roles   design A: one team PLAY (option set "plays"), agents told apart by role
  agents  design B: each agent's own option and own mu (option set "agent"); AgentOptionLearningEnv,
          AgentSMDPEnv
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

    def players(self) -> list[int]:
        """left_team index of the player each agent controls (fixed in the ma_ scenarios)."""
        return [int(o["active"]) for o in self._env.unwrapped.observation()]


def roles(raw: dict, players: list[int]) -> np.ndarray:
    """(N, 3) one-hot role per agent, recomputed on every frame: carrier (has the ball, or is
    nearest to it), near supporter (the other agent nearest the carrier), far supporter."""
    from hrl.options import carrier, closeness_to_ball
    c = carrier(raw)
    out = np.zeros((len(players), 3), dtype=np.float32)
    others = [i for i, p in enumerate(players) if p != c]
    if len(others) == len(players):  # the carrier is not one of the agents (keeper): nearest agent carries
        d = [closeness_to_ball(raw["left_team"][p], raw["ball"][:2]) for p in players]
        others.remove(int(np.argmin(d)))
    for i, p in enumerate(players):
        if i not in others:
            out[i, 0] = 1
    others.sort(key=lambda i: closeness_to_ball(raw["left_team"][players[i]], raw["left_team"][c]))
    for rank, i in enumerate(others):
        out[i, 1 if rank == 0 else 2] = 1
    return out

    def close(self):
        self._env.close()


def option_keywords() -> tuple[str, ...]:
    """Monitor columns for the options phase: per episode, options started (n_<name>) and reached (r_<name>)."""
    from hrl.execution import OPTIONS
    return tuple(f"{k}_{o.name}" for o in OPTIONS for k in ("n", "r"))


class _OptionCounts:
    """Counts option starts and subgoals reached in the current episode; written into the terminal info."""

    def _count_reset(self):
        from hrl.execution import OPTIONS
        self.counts = {k: 0 for k in option_keywords()}
        self._names = [o.name for o in OPTIONS]

    def _count(self, z: int, end) -> None:
        if end is not None:
            self.counts[f"n_{self._names[z]}"] += 1
            self.counts[f"r_{self._names[z]}"] += end == "reached"


class TeamOptionLearningEnv(_OptionCounts, gymnasium.Wrapper):
    """OptionLearningEnv for a team: z uniform at reset and after every beta, shared by all agents.
    Observation (N, 115 + K); reward combine_rewards(team reward, option reward); info option_end."""

    def __init__(self, env: MultiAgentGame, mix="both", weight=1.0, rng=None, with_roles: bool = False):
        super().__init__(env)
        from hrl.execution import N_OPTIONS
        self.k = N_OPTIONS
        self.mix, self.weight = mix, weight
        self.rng = rng or np.random.default_rng()
        self.with_roles = with_roles  # design A: append each agent's role (carrier / near / far)
        n = env.n
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (n, 115 + self.k + 3 * with_roles), np.float32)
        self.log: list[tuple[str, str, int]] = []

    def _raw(self):
        return self.env.unwrapped.raw_observation()

    def _start(self, raw):
        self.start = OptionStart(int(self.rng.integers(self.k)), self.t, float(raw["ball"][0]), raw)

    def _obs(self, obs):
        rows = [obs, np.tile(one_hot(self.start.z, self.k), (obs.shape[0], 1))]
        if self.with_roles:
            rows.append(roles(self._raw(), self.env.unwrapped.players()))
        return np.concatenate(rows, axis=1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.t = 0
        self._count_reset()
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
        self._count(z, end)
        if end:
            from hrl.execution import OPTIONS
            self.log.append((OPTIONS[z].name, end, self.t - self.start.step))
            if not (term or trunc):
                self._start(raw)
        info = dict(info, option_end=end)
        if term or trunc:
            info.update(self.counts)
        return self._obs(obs), combine_rewards(r_env, r_o, self.mix, self.weight), term, trunc, info


class TeamSMDPEnv(gymnasium.Env):
    """Phase mu for the team. Observation: agent 0's simple115v2 (it holds every player's position).
    Action z; then every agent acts as pi([obs_i, one_hot(z), one_hot(i)]) until beta.
    Reward: the discounted task reward over the option; info tau, option_end, episode_env_reward."""

    def __init__(self, env, option_policy: Callable[[np.ndarray], np.ndarray], gamma: float, rng=None,
                 with_roles: bool = False):
        from hrl.execution import N_OPTIONS
        self.env, self.pi, self.gamma = env, option_policy, gamma
        self.with_roles = with_roles
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
            tag = roles(self._raw(), self.env.unwrapped.players()) if self.with_roles else np.eye(rows.shape[0], dtype=np.float32)
            actions = np.asarray(self.pi(np.concatenate([rows, tag], axis=1))).ravel()
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


class AgentOptionLearningEnv(_OptionCounts, gymnasium.Wrapper):
    """Design B, phase options: every agent runs its OWN option z_i (uniform at reset and after its own
    beta), sees one_hot(z_i), and is paid combine_rewards(team reward, ITS subgoal). Reward and
    option_end are per agent: reward (N,), info["option_end_agents"]."""

    def __init__(self, env: MultiAgentGame, mix="both", weight=1.0, rng=None):
        super().__init__(env)
        from hrl.execution import N_OPTIONS
        self.k, self.mix, self.weight = N_OPTIONS, mix, weight
        self.rng = rng or np.random.default_rng()
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (env.n, 115 + self.k), np.float32)

    def _raw(self):
        return self.env.unwrapped.raw_observation()

    def _start(self, i, raw):
        self.starts[i] = OptionStart(int(self.rng.integers(self.k)), self.t, float(raw["ball"][0]), raw,
                                     player=self.players[i])

    def _obs(self, obs):
        return np.concatenate([obs, np.stack([one_hot(s.z, self.k) for s in self.starts])], axis=1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.t, self.players = 0, self.env.unwrapped.players()
        self.starts = [None] * len(self.players)
        self._count_reset()
        raw = self._raw()
        for i in range(len(self.players)):
            self._start(i, raw)
        return self._obs(obs), info

    def step(self, action):
        prev = self._raw()
        obs, r_env, term, trunc, info = self.env.step(action)
        self.t += 1
        raw = self._raw()
        rewards, ends = np.zeros(len(self.players), dtype=np.float32), []
        for i, st in enumerate(self.starts):
            end = termination(st, prev, raw, self.t, term or trunc)
            rewards[i] = combine_rewards(r_env, option_reward(st, prev, raw, self.t), self.mix, self.weight)
            ends.append(end)
            self._count(st.z, end)
            if end and not (term or trunc):
                self._start(i, raw)
        info = dict(info, option_end_agents=ends)
        if term or trunc:
            info.update(self.counts)
        return self._obs(obs), rewards, term, trunc, info


class AgentSMDPEnv(gymnasium.Env):
    """Design B, phase mu: each agent's mu picks its own option; observation (N, 115), action (N,).
    All agents act through the frozen option policy until ANY agent's option ends; then every agent
    decides again (synchronous decisions, so one tau for all and the SMDP buffer is unchanged).
    Reward: the discounted team reward over the window, the same for every agent."""

    def __init__(self, env, option_policy: Callable[[np.ndarray], np.ndarray], gamma: float):
        from hrl.execution import N_OPTIONS
        self.env, self.pi, self.gamma, self.k = env, option_policy, gamma, N_OPTIONS
        n = env.unwrapped.n
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (n, 115), np.float32)
        self.action_space = gymnasium.spaces.MultiDiscrete([N_OPTIONS] * n)

    def _raw(self):
        return self.env.unwrapped.raw_observation()

    def reset(self, *, seed=None, options=None):
        self._obs, info = self.env.reset(seed=seed, options=options)
        self._t, self._ep = 0, 0.0
        self.players = self.env.unwrapped.players()
        return self._obs, info

    def step(self, zs):
        from hrl.smdp import option_return
        zs = [int(z) for z in np.asarray(zs).ravel()]
        raw0 = self._raw()
        starts = [OptionStart(z, self._t, float(raw0["ball"][0]), raw0, player=p) for z, p in zip(zs, self.players)]
        rewards, info, reason = [], {}, None
        while True:
            rows = np.concatenate([self._obs, np.stack([one_hot(z, self.k) for z in zs])], axis=1)
            actions = np.asarray(self.pi(with_ids(rows))).ravel()
            prev = self._raw()
            self._obs, r, term, trunc, info = self.env.step(actions)
            self._t += 1
            rewards.append(r)
            ends = [termination(st, prev, self._raw(), self._t, term or trunc) for st in starts]
            if any(e is not None for e in ends):
                reason = next(e for e in ends if e is not None)
                break
        self._ep += float(sum(rewards))
        info = dict(info, tau=len(rewards), option_end=reason, env_reward=float(sum(rewards)),
                    episode_env_reward=self._ep, option_end_agents=ends)
        return self._obs, option_return(rewards, self.gamma), term, trunc, info

    def close(self):
        self.env.close()


class AgentsAsSlots(VecEnv):
    """A VecEnv of G multi-agent games (any VecEnv whose observations are (N, d)), shown to SB3 as
    G*N single-agent slots with one_hot(agent id) appended. Rewards and dones are the game's, copied
    to each of its agents; infos are copied too (option_end, episode stats)."""

    def __init__(self, games: VecEnv, agents: int, ids: bool = True):
        self.games, self.n, self.ids = games, agents, ids
        d = games.observation_space.shape[-1]
        space = gymnasium.spaces.Box(-np.inf, np.inf, (d + agents * ids,), np.float32)
        a = games.action_space
        n_actions = int(a.nvec[0]) if isinstance(a, gymnasium.spaces.MultiDiscrete) else 19
        super().__init__(games.num_envs * agents, space, gymnasium.spaces.Discrete(n_actions))

    def _flat(self, obs):
        return np.concatenate([with_ids(o) if self.ids else o for o in obs], axis=0)

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
                    t = info["terminal_observation"]
                    inf["terminal_observation"] = (with_ids(t) if self.ids else t)[i]
                if "option_end_agents" in info:  # design B: each agent's own option end
                    inf["option_end"] = info["option_end_agents"][i]
                out_infos.append(inf)
        r = np.asarray(r, dtype=np.float32)
        rew = r.reshape(-1) if r.ndim == 2 else np.repeat(r, self.n)  # per-agent rewards, or the team's
        return self._flat(obs), rew, np.repeat(done, self.n), out_infos

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


def make_games(level: str, agents: int, n_games: int, seed: int, wrap: Callable[[MultiAgentGame, int], Any] | None = None,
               ids: bool = True):
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
    return AgentsAsSlots(SubprocVecEnv([make(i) for i in range(n_games)], start_method="spawn"), agents, ids)
