"""The structured types of the hrl package: the dicts the game hands over (their keys) and the
strings that can only take a few values. Hover a key or a name to see the line under it."""

from __future__ import annotations

from typing import Any, Callable, Literal, Protocol, TypedDict

import numpy as np

# --- strings from a fixed set ----------------------------------------------------------------
Lane = Literal["left", "centre", "right"]
Termination = Literal["reached", "failed", "timeout"]
"""What beta returns (options.termination, node 31)."""
OptionEnd = Literal["reached", "failed", "timeout", "interrupted", "primitive", "unavailable"]
ActionKind = Literal["option", "primitive"]
SubgoalKind = Literal[
    "forward_in_lane", "ball_in_box", "is_shot", "keep_ball", "completed_pass", "beat_defender",
    "shot_released", "carry_to_lane", "carry_to_box", "give_and_go", "through_ball", "switch_play",
    "agent_carry", "agent_pass", "agent_shoot", "get_open", "receive",
]
"""Which test ends an option as reached: the options.py function of the same name."""
OptionSetName = Literal["zones", "actions", "targeted", "plays", "agent"]
"""Which options exist (options.OPTION_SETS): "zones" the six pitch-zone options (nodes 30-37),
"actions" pass / dribble / shoot (node 37a)."""

# --- the dicts ----------------------------------------------------------------------------


class _BallKeys(TypedDict):
    """The ball keys: always present (also in the tests and the toy pitch)."""

    ball: np.ndarray
    """[x, y, z]. x: -1 our goal .. +1 theirs. y: -0.42 .. +0.42, negative = our left.
    Lane: options.lane(y). Box: x > options.BOX_X and |y| < options.BOX_HALF_WIDTH."""
    ball_direction: np.ndarray
    """[dx, dy, dz]: how far the ball moved on the last step."""
    ball_owned_team: int
    """-1 nobody, 0 us, 1 them (options.NOBODY / OURS / THEIRS)."""


class RawObs(_BallKeys, total=False):
    """GRF's raw observation (a dict): obs["ball"], obs["ball_owned_team"], ...
    Positions are pitch units, not metres: options.to_metres converts."""

    ball_rotation: np.ndarray
    ball_owned_player: int
    left_team: np.ndarray
    """(N, 2): [x, y] of each of our players (we are the left team)."""
    left_team_direction: np.ndarray
    left_team_tired_factor: np.ndarray
    left_team_yellow_card: np.ndarray
    left_team_active: np.ndarray
    left_team_roles: np.ndarray
    """0 GK, 1 CB, 2 LB, 3 RB, 4 DM, 5 CM, 6 LM, 7 RM, 8 AM, 9 CF."""
    right_team: np.ndarray
    """(N, 2): [x, y] of each opponent."""
    right_team_direction: np.ndarray
    right_team_tired_factor: np.ndarray
    right_team_yellow_card: np.ndarray
    right_team_active: np.ndarray
    right_team_roles: np.ndarray
    active: int
    """Index in left_team of the player we control."""
    designated: int
    sticky_actions: np.ndarray
    score: list[int]
    """[our goals, their goals]."""
    steps_left: int
    game_mode: int
    """0 normal, 1 kick-off, 2 goal kick, 3 free kick, 4 corner, 5 throw-in, 6 penalty."""


class Info(TypedDict, total=False):
    """The dict env.step() returns last: info["tau"], info["option_end"], ..."""

    tau: int
    """Environment steps the option took (1 for a primitive)."""
    option_end: OptionEnd
    score_reward: float
    """GRF: the goals-only part of the reward."""
    style_bonus: float
    """StyleConditionedEnv: the style reward added to this decision (style_bonus x w[z] when z reached its subgoal)."""
    env_reward: float
    """SMDPEnv: the game's rewards during this decision, summed undiscounted, before the deliberation cost."""
    episode_env_reward: float
    """SMDPEnv: env_reward summed since reset. VecMonitor logs it, so goals stay countable under a cost."""
    dumps: list[Any]


class OptionRow(TypedDict):
    """One option's line in the diagnostic (diagnose.py, node 37b). Fractions are of its runs."""

    name: str
    reached: float
    """Runs that ended with its own subgoal reached."""
    failed: float
    """Runs that ended because the attack ended or they won the ball."""
    timeout: float
    """Runs that hit the clock (HORIZON steps)."""
    reached_per_episode: float
    goal_rate: float
    """Episodes with a goal, with this z forced for the whole episode."""


class DiagnosticReport(TypedDict):
    """What diagnose.py measures for one trained option policy (node 37b)."""

    option_set: str
    episodes: int
    """Episodes played per option."""
    options: list[OptionRow]
    argmax_change: float
    """Fraction of visited states where pi's most likely action changes with z."""
    max_tv: float
    """Mean over states of the largest total-variation distance between pi(. | s, z) for two z."""
    n_states: int


# --- environment step and policy --------------------------------------------------------------
StepResult = tuple[np.ndarray, float, bool, bool, Info]
"""obs, reward, terminated, truncated, info."""
Policy = Callable[[np.ndarray], int]
"""An option policy: observation + one_hot(z) -> a GRF action."""


class RawSource(Protocol):
    """An env that also hands over the game's raw frame: the toy pitch and GFootballEnv."""

    def raw_observation(self) -> RawObs: ...

# --- the survey's design choices, as flags (train_hrl.py) ------------------------------------
RewardMix = Literal["option", "both", "env"]
TerminationMode = Literal["clock", "geometric"]
"""How beta times an option out: "clock" after HORIZON steps, "geometric" a 1/HORIZON coin each step (node 31)."""
