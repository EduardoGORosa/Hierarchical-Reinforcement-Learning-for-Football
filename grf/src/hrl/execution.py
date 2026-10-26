"""Interact, in the survey's Algorithm 1: how options turn into primitive actions.

Framework: Klissarov et al. 2025 (arXiv:2506.14045). The survey's generic loop is

    Omega <- OptionSpecification      options.py (fixed, by hand)
    O     <- OptionLearning           phase 1 below: learn pi(a | s, z) for every option
    D     <- Interact(Env, O, mu)     these wrappers: the EXECUTION MODEL (section 7)
    mu    <- PolicyImprovement        phase 2 below: learn the high-level policy mu(o | s)

and the survey leaves every one of these as a documented choice. The choices made here, each
with its section, so each can be cited and varied:

  Separate phases (8.1 "Non-Stationarity"). Learning options and mu at once is non-stationary.
      The survey names two remedies, an option-learning phase in which mu picks options uniformly
      at random, and learning the options before fixing them; using both is our choice.
  Combining rewards (8.3). How option reward and task reward mix during option learning is
      open ("a thorough examination ... has yet to be made"). combine_rewards offers the three
      positions the survey lists, so the choice is an experimental variable, not a hidden
      default.
  Execution model (7.1). Call-and-return: mu picks an option and waits for beta. The survey's
      alternative is interruption: every k steps, stop the option if continuing it is worth less
      than choosing afresh, Q(s, o) < V(s). The interruption theorem (Sutton, Precup & Singh
      1999) says this never hurts WHEN THE OPTIONS ARE FIXED, which they are in phase 2.
  Options and primitives (survey 3.2.1: mu maps S to a distribution over O and A). Letting mu
      also pick a primitive action keeps the best flat policy representable, since the
      survey's 2.2 notes that restricting to options can exclude it. with_primitives=True.

Single-agent: one controlled player. The three-agent version (every agent gets the same z)
is still an open question for the advisor.

Nodes 32-36 (layer 4 of the build map). Tests run in the container (gymnasium needed):
    ./run.sh hrl check 32          (... 36)
They use hrl.toy.KnownAnswerPitch, a stand-in for GRF whose right answers are exact.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import gymnasium
import numpy as np

from hrl.options import OPTIONS, OptionStart, option_reward, termination
from hrl.smdp import option_return
from hrl.hints import (
    ActionKind,
    Info,
    Policy,
    RawObs,
    RawSource,
    RewardMix,
    StepResult,
    TerminationMode,
)

N_OPTIONS = len(OPTIONS)
N_PRIMITIVES = 19
REWARD_MIXES = ("option", "both", "env")


def raw_frame(env: gymnasium.Env[np.ndarray, int]) -> RawObs:
    """The game's raw frame (ball, owner, ...), from the env under any wrappers."""
    return cast(RawSource, env.unwrapped).raw_observation()


def one_hot(z: int, n: int = N_OPTIONS) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[z] = 1.0
    return v


# --- node 33: combining rewards (survey 8.3) ----------------------------------


def combine_rewards(
    r_env: float, r_option: float, mix: RewardMix, weight: float = 0.1
) -> float:
    """The reward an option policy is trained on, for the survey's three positions:

    "option"  option reward only: the lower level ignores the task (Dayan & Hinton 1993)
    "both"    r_env + weight * r_option (the softer choice, as in FeUdal Networks,
              Vezhnevets et al. 2017; thesis_ideas' +0.1 is this with weight 0.1)
    "env"     task reward only, no option reward at all (as in option-critic, Bacon et al.
              2017). The options then differ only through z and their terminations.
    """
    if mix == "option":
        return r_option
    if mix == "env":
        return r_env

    return r_env + r_option * weight


# --- node 35: interruption (survey 7.1.2) -------------------------------------


def interrupt_by_advantage(q_so: float, v_s: float) -> bool:
    """Sutton, Precup & Singh's criterion: stop option o in state s when continuing it is
    worth less than choosing afresh."""
    # TODO(node 35): True means "stop now". q_so is the value of carrying on with the running
    #   option from s, v_s the value of letting mu choose again from s. On a tie, carrying on
    #   costs nothing, so the option keeps running. SMDPEnv's should_interrupt calls this.
    return q_so < v_s


# --- node 36: mu over options AND primitives (survey 3.2.1) -------------------


def action_meaning(a: int, with_primitives: bool) -> tuple[ActionKind, int]:
    """What mu's action number a means.

    0 .. N_OPTIONS-1                        -> ("option", z)        run option z until beta
    N_OPTIONS .. N_OPTIONS+N_PRIMITIVES-1   -> ("primitive", k)     GRF action k, one step
    Anything else, or a primitive when with_primitives is False, is a bug: raise ValueError.
    """
    # TODO(node 36): SMDPEnv.step reads the pair: ("option", z) runs option z through the loop,
    #   ("primitive", k) sends GRF action k to the game for one step (tau = 1). A number outside
    #   mu's menu means the action space and this table disagree: raise ValueError, or mu
    #   silently plays something it never chose.
    print(a, N_OPTIONS - 1, N_OPTIONS + N_PRIMITIVES - 1)
    if a >= 0 and a <= N_OPTIONS - 1:
        print("option", a)
        return ("option", a)
    elif a >= N_OPTIONS and a <= N_OPTIONS + N_PRIMITIVES - 1 and with_primitives:
        print("primitive", a - N_OPTIONS)
        return ("primitive", a - N_OPTIONS)

    raise ValueError


# --- phase 1: OptionLearning ---------------------------------------------------


class OptionLearningEnv(gymnasium.Wrapper[np.ndarray, int, np.ndarray, int]):
    """The data option policies learn from, with mu uniform over options (survey 8.1).

    observation = [base observation, one_hot(z)]   one network for every option (Principle 3)
    reward      = combine_rewards(env reward, option reward, mix, weight)
    A new z is drawn uniformly at reset and whenever beta ends the current option.
    """

    observation_space: gymnasium.spaces.Box
    """The base observation with N_OPTIONS more numbers on the end: one_hot(z)."""
    mix: RewardMix
    """Which reward the option policies learn from (node 33)."""
    weight: float
    """r_o's weight when mix is "both"."""
    termination_mode: TerminationMode
    """"clock" or "geometric": how beta times an option out (node 31)."""
    rng: np.random.Generator
    """Draws each new z, and flips the coin in geometric mode."""
    log: list[tuple[str, str, int]]
    """(option, reason, duration) for every option that ended: the key debug table."""
    start: OptionStart
    """The running option: its z, and the clock and ball x when it began."""
    t: int
    """The environment clock: steps since reset, across all options."""
    prev_raw: RawObs
    """The raw frame before the current step. beta and r_o compare it with the new one."""

    def __init__(
        self,
        env: gymnasium.Env[np.ndarray, int],
        mix: RewardMix = "both",
        weight: float = 0.1,
        termination_mode: TerminationMode = "clock",
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(env)
        assert mix in REWARD_MIXES
        base = env.observation_space
        assert isinstance(base, gymnasium.spaces.Box), (
            "one_hot(z) is appended to a Box observation"
        )
        self.observation_space = gymnasium.spaces.Box(
            low=np.concatenate([base.low, np.zeros(N_OPTIONS)]).astype(np.float32),
            high=np.concatenate([base.high, np.ones(N_OPTIONS)]).astype(np.float32),
            dtype=np.float32,
        )
        self.mix, self.weight, self.termination_mode = mix, weight, termination_mode
        self.rng = rng or np.random.default_rng()
        self.log = []

    def _start(self, raw: RawObs) -> None:
        self.start = OptionStart(
            int(self.rng.integers(N_OPTIONS)), self.t, float(raw["ball"][0])
        )

    def _obs(self, base_obs: np.ndarray) -> np.ndarray:
        return np.concatenate([base_obs, one_hot(self.start.z)]).astype(np.float32)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        base_obs, info = self.env.reset(**kwargs)
        self.t = 0
        self.prev_raw = raw_frame(self.env)
        self._start(self.prev_raw)
        return self._obs(base_obs), info

    def step(self, action: int) -> StepResult:
        self.prev_raw = raw_frame(self.env)
        base_obs, r_env, terminated, truncated, info = self.env.step(action)
        self.t += 1
        raw = raw_frame(self.env)
        end = termination(
            self.start,
            self.prev_raw,
            raw,
            self.t,
            truncated or terminated,
            action,
            self.termination_mode,
            self.rng,
        )

        r_o = option_reward(self.start, self.prev_raw, raw, self.t, action)

        if end:
            self.log.append((OPTIONS[self.start.z].name, end, self.t - self.start.step))
            if not (terminated or truncated):
                self._start(raw)

        r_combined = combine_rewards(r_env, r_o, self.mix, self.weight)

        return (
            self._obs(base_obs),
            r_combined,
            terminated,
            truncated,
            Info(**info, option_end=end),
        )


# --- phase 2: PolicyImprovement of mu ------------------------------------------


class SMDPEnv(gymnasium.Env[np.ndarray, int]):
    """The data mu learns from: one step() here is one whole option (call-and-return, 7.1.1).

    action      z in 0..5 runs option z through the frozen option policy until beta.
                With with_primitives, 6..24 runs primitive action (a - 6) for one step.
    reward      smdp.option_return of the ENVIRONMENT rewards during the option, with no
                option reward: mu is paid by the task only
    info["tau"] how many environment steps it took; the trainer discounts by gamma^tau
    should_interrupt(base_obs, z) -> bool, checked every `interrupt_every` steps if given.
                Leave it None for call-and-return. The theorem covers fixed options and mu,
                so use it for evaluation, not for training.
    """

    observation_space: gymnasium.spaces.Space[np.ndarray]
    """The game's own: mu sees the same observation the game gives."""
    action_space: gymnasium.spaces.Discrete
    """One number per option (0..5), then one per primitive if with_primitives."""
    env: gymnasium.Env[np.ndarray, int]
    """The game underneath: the toy pitch or GRF. Its step() is ONE environment step."""
    pi: Policy
    """The frozen option policy: observation + one_hot(z) -> a GRF action."""
    gamma: float
    """Discount inside an option, for option_return."""
    with_primitives: bool
    """mu may also pick a single GRF action, run for one step (node 36)."""
    termination_mode: TerminationMode
    """"clock" or "geometric": how beta times an option out (node 31)."""
    should_interrupt: Callable[[np.ndarray, int], bool] | None
    """(observation, z) -> stop the option now? None is plain call-and-return (node 35)."""
    interrupt_every: int
    """Ask should_interrupt every this many steps."""
    rng: np.random.Generator
    """Flips the coin in geometric mode."""
    _obs: np.ndarray
    """The latest observation: what pi sees next, and what mu gets back."""
    _t: int
    """The environment clock: steps since reset, across all options. OptionStart.step reads it."""

    def __init__(
        self,
        env: gymnasium.Env[np.ndarray, int],
        option_policy: Policy,
        gamma: float,
        with_primitives: bool = False,
        termination_mode: TerminationMode = "clock",
        should_interrupt: Callable[[np.ndarray, int], bool] | None = None,
        interrupt_every: int = 1,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.env, self.pi, self.gamma = env, option_policy, gamma
        self.with_primitives, self.termination_mode = with_primitives, termination_mode
        self.should_interrupt, self.interrupt_every = should_interrupt, interrupt_every
        self.rng = rng or np.random.default_rng()
        self.observation_space = env.observation_space
        self.action_space = gymnasium.spaces.Discrete(
            N_OPTIONS + (N_PRIMITIVES if with_primitives else 0)
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._obs, info = self.env.reset(seed=seed, options=options)
        self._t = 0
        return self._obs, info

    def step(self, a: int) -> StepResult:
        # Without primitives every action is an option, so node 32 doesn't need node 36.
        kind, a = (
            action_meaning(int(a), True) if self.with_primitives else ("option", int(a))
        )
        if kind == "primitive":  # one step, tau = 1
            self._obs, r, terminated, truncated, info = self.env.step(a)
            self._t += 1
            return (
                self._obs,
                float(r),
                terminated,
                truncated,
                Info(**info, tau=1, option_end="primitive"),
            )
        prev = raw_frame(self.env)
        start = OptionStart(a, self._t, float(prev["ball"][0]))
        rewards, reason, terminated, truncated, info = [], None, False, False, {}
        while True:
            action = self.pi(np.concatenate([self._obs, one_hot(a)]))
            prev = raw_frame(self.env)
            self._obs, r, terminated, truncated, info = self.env.step(action)
            self._t += 1
            rewards.append(r)
            tau = len(rewards)
            reason = termination(
                start,
                prev,
                raw_frame(self.env),
                self._t,
                terminated or truncated,
                action,
                mode=self.termination_mode,
                rng=self.rng,
            )

            if reason is not None:
                break

            if (
                self.should_interrupt
                and self.should_interrupt(self._obs, a)
                and tau % self.interrupt_every == 0
            ):
                reason = "interrupted"
                break

        r_o = option_return(rewards, self.gamma)

        return (
            self._obs,
            r_o,
            terminated,
            truncated,
            Info(**info, tau=tau, option_end=reason),
        )
