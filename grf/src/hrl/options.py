"""The option set: six designed subgoal options, specified the way the survey specifies options.

Framework: Klissarov, Bagaria, Luo, Konidaris, Precup & Machado, "Discovering Temporal
Structure: An Overview of Hierarchical Reinforcement Learning" (arXiv:2506.14045),
section 3.2. Every choice below is one of the survey's named design principles, so each
can be defended by citation and varied as an ablation.

  Definition 1   an option is a triple <I, pi, beta>
  Definition 3   an OPTION REWARD r_o makes the option purposeful: a "subgoal option"
                 (Design Principle 1: one objective per option)
  Definition 4   an OPTION IDENTIFIER z names the option. Here z is an index 0..5, fed as a
                 one-hot input to ONE shared network pi(a | s, z) (Design Principle 3)
  Principle 2    constrained initiation and termination:
                   I(s) = 1 everywhere (as in option-critic);
                   beta = subgoal reached, OR the attack is over, OR a clock of H_o steps.

The clock matters. A step limit makes beta depend on elapsed time, so the option is NOT Markov
in the state alone: Sutton, Precup & Singh (1999, sec. 2) call these semi-Markov options. Two
consequences (same paper, sec. 6):
  - learning mu over whole options (SMDP learning) stays valid;
  - intra-option learning, which needs Markov options, does not apply.
A memoryless alternative with the same expected length: terminate with constant probability
1/H_o on every step (a standard geometric-length construction, not from the survey). It is
termination mode "geometric", an ablation you can report. It removes only the clock's memory:
the lane and keep subgoals still read OptionStart, so only enter_box and shoot become Markov.

In Algorithm 1 of the survey this file is OptionSpecification, done once and by hand. The
objectives are fixed, and only the option policies are learned (survey: "some specify options
with fixed objectives and only learn the option components").

Coordinates are GRF's raw observation: x from -1 (our goal) to +1 (theirs), y from -0.42 to
+0.42. Convert to metres before computing any distance. Open item (thesis_ideas 6.1): confirm on
a rendered clip that NEGATIVE y is the attacking team's LEFT (GRF's action "top" moves towards
negative y). If it isn't, swap the two wing lanes and leave the tests as they are.

Two option sets (OPTION_SETS), chosen once per process with use_option_set:
  "zones"    the six above (nodes 30-37). Parked: on 3v1 they collapsed (thesis_ideas 3.4c).
  "actions"  pass / dribble / shoot, subgoals that fit a ~33-step 3v1 attack (node 37a).
The default is "zones", so every test of nodes 30-37 still sees the six.

Checks:        python3 -m hrl.check 30      (and 31, 37a)
Demo dumps:    ./run.sh hrl options         (runs every subgoal test over grf/runs/demo)
"""

from __future__ import annotations
from math import sqrt

from typing import Sequence, cast

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from hrl.hints import (
    Lane,
    OptionSetName,
    RawObs,
    SubgoalKind,
    Termination,
    TerminationMode,
)

# Scales from the engine, not from a 105 x 68 m pitch: observations are engine metres divided by
# X_FIELD_SCALE = 54.4 and Y_FIELD_SCALE = -83.6 (gfootball_engine/src/defines.hpp:55-56).
X_METRES = 54.4  # one x unit, in metres
Y_METRES = 83.6  # one y unit, in metres
LANE_EDGE = 0.14  # |y| <= 0.14 is the centre lane
# Penalty area as the engine's referee defines it (onthepitch/referee.cpp:466-469):
# |y| < 20.15 m and x beyond pitchHalfW - 16.5 = 38.5 m.
BOX_X = 0.708  # 38.5 / 54.4
BOX_HALF_WIDTH = 0.241  # 20.15 / 83.6
GOAL_HALF_WIDTH = 0.044  # goal mouth, gfootball/doc/observation.md
FORWARD_METRES = 5.0  # a lane subgoal needs this much forward progress
HORIZON = (
    30  # H_o: 3 s of game time. use_option_set changes it to the chosen set's horizon
)
ACTION_HORIZON = 20  # the "actions" set's H_o, measured (node 37a): a pass takes ~14 steps, a dribble ~19
SHOT_ACTION = 12  # "shot" in GRF's default (v1) action set
SHOT_MIN_SPEED = 0.03  # x units per step towards their goal. Demo dumps: shots 0.037-0.068, other releases <= 0.025
SHOT_AIM_MARGIN = 0.005  # y units past each post still counted as aimed. Demo dumps: shots aimed to |y| <= 0.0445, a firm pass to 0.052
OPTION_REWARD = 1.0  # r_o when the subgoal is reached (Definition 3)
SHOOT_RANGE_M = 25.0  # --initiation: shoot only from within this distance of the goal centre
INITIATION = False  # --initiation: options.use_initiation(True) switches the initiation sets on

OURS, THEIRS, NOBODY = 0, 1, -1  # values of obs["ball_owned_team"]


@dataclass(frozen=True)
class Option:
    """One option (Definition 1, 3, 4). It ends when its subgoal is reached, when the attack ends
    or they win the ball, or after `horizon` steps: termination(), node 31. pi is NOT stored
    here: all options share one network, which takes z as an input (Design Principle 3)."""

    z: int
    name: str
    subgoal: (
        SubgoalKind  # which test says "reached": the function of the same name below
    )
    lane: Lane | None = None  # for forward_in_lane only
    horizon: int = HORIZON

    def initiation(self, obs: RawObs) -> bool:
        """I(s). Off (the default): available everywhere, Design Principle 2's simplest constraint.
        With use_initiation(True), an action option starts only where it can work:
        pass with the ball, dribble with the ball and a defender to beat, shoot within SHOOT_RANGE_M."""
        if not INITIATION or self.subgoal not in ("completed_pass", "beat_defender", "shot_released"):  # targeted set: no I(s)
            return True
        if not we_own_ball(obs):
            return False
        if self.subgoal == "beat_defender":
            return defender_to_beat(obs) != -1
        if self.subgoal == "shot_released":
            dx, dy = to_metres((1.0 - float(obs["ball"][0]), float(obs["ball"][1])))
            return sqrt(dx * dx + dy * dy) <= SHOOT_RANGE_M
        return True


ZONE_OPTIONS = (
    Option(0, "left_wing", subgoal="forward_in_lane", lane="left"),
    Option(1, "centre", subgoal="forward_in_lane", lane="centre"),
    Option(2, "right_wing", subgoal="forward_in_lane", lane="right"),
    Option(3, "enter_box", subgoal="ball_in_box"),
    Option(4, "shoot", subgoal="is_shot"),
    Option(5, "keep", subgoal="keep_ball"),
)
ACTION_OPTIONS = (  # node 37a: the styles of thesis_ideas 3.4b, one option each
    Option(
        0, "pass", subgoal="completed_pass", horizon=ACTION_HORIZON
    ),  # combination play
    Option(
        1, "dribble", subgoal="beat_defender", horizon=ACTION_HORIZON
    ),  # individual play
    Option(2, "shoot", subgoal="shot_released", horizon=ACTION_HORIZON),  # direct play
)
TARGETED_OPTIONS = (  # variation 2 (2026-10-29): dribble split by where it carries the ball
    Option(0, "pass", subgoal="completed_pass", horizon=ACTION_HORIZON),
    Option(1, "carry_left", subgoal="carry_to_lane", lane="left", horizon=ACTION_HORIZON),
    Option(2, "carry_centre", subgoal="carry_to_lane", lane="centre", horizon=ACTION_HORIZON),
    Option(3, "carry_right", subgoal="carry_to_lane", lane="right", horizon=ACTION_HORIZON),
    Option(4, "carry_box", subgoal="carry_to_box", horizon=ACTION_HORIZON),
    Option(5, "shoot", subgoal="shot_released", horizon=ACTION_HORIZON),
)
OPTION_SETS: dict[
    OptionSetName, tuple[tuple[Option, ...], int]
] = {  # name -> (options, H_o)
    "zones": (ZONE_OPTIONS, HORIZON),
    "actions": (ACTION_OPTIONS, ACTION_HORIZON),
    "targeted": (TARGETED_OPTIONS, ACTION_HORIZON),
}
OPTIONS = ZONE_OPTIONS
SUBGOALS = tuple(o.name for o in OPTIONS)


def use_initiation(on: bool) -> None:
    """Switch the action options' initiation sets on or off for this process (--initiation).
    Like use_option_set, call it inside every SubprocVecEnv worker."""
    global INITIATION
    INITIATION = on


def use_option_set(name: OptionSetName) -> None:
    """Make `name` the option set of this process: OPTIONS, SUBGOALS and HORIZON here, and
    OPTIONS and N_OPTIONS in hrl.execution. Call it once, before building any env (each
    SubprocVecEnv worker is its own process, so train_hrl calls it inside every worker)."""
    global OPTIONS, SUBGOALS, HORIZON
    OPTIONS, HORIZON = OPTION_SETS[name]
    SUBGOALS = tuple(o.name for o in OPTIONS)
    execution = sys.modules.get("hrl.execution")
    if execution is not None:
        setattr(execution, "OPTIONS", OPTIONS)
        setattr(execution, "N_OPTIONS", len(OPTIONS))


@dataclass
class OptionStart:
    """What an option has to remember about the moment it started. That it needs any memory
    at all is exactly why it isn't Markov."""

    z: int  # which option is running: SUBGOALS[z] is its name
    step: int  # the environment step it started on
    ball_x: (
        float  # the ball's x when it started; lane subgoals measure progress from here
    )
    raw: RawObs | None = (
        None  # the whole raw frame when it started: pass and dribble read it
    )


# --- node 30: the subgoals, i.e. what each option reward pays for ------------


def to_metres(xy: Sequence[float] | np.ndarray) -> np.ndarray:
    """(x, y) in pitch units -> (x, y) in metres."""
    return (xy[0] * X_METRES, xy[1] * Y_METRES)


def lane(y: float) -> Lane:
    """'left', 'centre' or 'right'. The centre lane includes its edges."""

    if y > LANE_EDGE:
        return "right"
    elif y < -LANE_EDGE:
        return "left"

    return "centre"


def we_own_ball(obs: RawObs) -> bool:
    return obs["ball_owned_team"] == OURS


def is_shot(prev_obs: RawObs, obs: RawObs, action: int | None = None) -> bool:
    """Did one of our players shoot on this step?

    If `action` is given (a learning agent): it pressed SHOT_ACTION while we had the ball.
    If `action` is None (the built-in AI, whose dumps only record "builtin_ai"), infer it from
    the ball: we owned it on prev_obs, nobody owns it now, it moves towards their goal (+x)
    faster than SHOT_MIN_SPEED, and its straight line carried on to x = 1 lands within
    GOAL_HALF_WIDTH + SHOT_AIM_MARGIN of the centre.
    """
    ball_x_speed, ball_y_speed, ball_x, ball_y = (
        obs["ball_direction"][0],
        obs["ball_direction"][1],
        obs["ball"][0],
        obs["ball"][1],
    )

    if prev_obs["ball_owned_team"] == OURS and action == SHOT_ACTION:
        return True
    elif (
        prev_obs["ball_owned_team"] == OURS
        and action == None
        and obs["ball_owned_team"] == NOBODY
        and ball_x_speed > SHOT_MIN_SPEED
    ):
        return (
            abs(ball_y + ball_y_speed * (1 - ball_x) / ball_x_speed)
            <= GOAL_HALF_WIDTH + SHOT_AIM_MARGIN
        )

    return False


def forward_in_lane(obs: RawObs, start_x: float, lane_name: Lane) -> bool:
    """We own the ball, it's in `lane_name`, and it's at least FORWARD_METRES further forward
    (x, in metres) than start_x, where it was when the option started."""
    if we_own_ball(obs) and lane(obs["ball"][1]) == lane_name:
        return (
            to_metres(obs["ball"][:2])[0]
            >= to_metres([start_x, obs["ball"][1]])[0] + FORWARD_METRES
        )

    return False


def ball_in_box(obs: RawObs) -> bool:
    """We own the ball inside the penalty area."""
    return (
        we_own_ball(obs)
        and obs["ball"][0] > BOX_X
        and abs(obs["ball"][1]) < BOX_HALF_WIDTH
    )


def keep_ball(obs: RawObs, steps_run: int, steps_needed: int) -> bool:
    """We still own the ball, and steps_run >= steps_needed."""
    return we_own_ball(obs) and steps_needed <= steps_run


# --- node 37a: subgoals that fit a 3v1 attack -----------------------------------


def closeness_to_ball(player, ball):
    return np.sqrt(
        np.square(abs(player[0] - ball[0]) * X_METRES)
        + np.square(abs(player[1] - ball[1]) * Y_METRES)
    )


def carrier(obs: RawObs) -> int:
    """Index in obs["left_team"] of the player whose ball it is: ball_owned_player when we own
    it, otherwise our player nearest the ball, in metres."""

    if we_own_ball(obs):
        return obs["ball_owned_player"]

    best_distance = float("inf")
    carrier_index = 0

    ball = obs["ball"][:-1]
    for i, player in enumerate(obs["left_team"]):
        distance = closeness_to_ball(player, ball)
        if distance < best_distance:
            best_distance = distance
            carrier_index = i

    return carrier_index


def completed_pass(start_obs: RawObs, obs: RawObs) -> bool:
    """We own the ball now, and the player who has it is not carrier(start_obs)."""
    return we_own_ball(obs) and carrier(start_obs) != carrier(obs)


def defender_to_beat(start_obs: RawObs) -> int:
    """Index in obs["right_team"] of the defender this option has to beat: of the opponents
    in front of the ball (x greater than the ball's x) who are not the goalkeeper
    (right_team_roles 0), the nearest to the ball in metres. -1 if there is none."""

    opponent_index = -1  # no defender yet
    best_distance = float("inf")

    ball = start_obs["ball"][:-1]
    for i, opponent in enumerate(start_obs["right_team"]):
        if start_obs["right_team_roles"][i] == 0:  # the keeper, at any index
            continue
        if opponent[0] <= ball[0]:  # behind the ball
            continue
        distance = closeness_to_ball(opponent, ball)
        if distance < best_distance:
            best_distance = distance
            opponent_index = i

    return opponent_index


def beat_defender(start_obs: RawObs, obs: RawObs) -> bool:
    """carrier(start_obs) still has the ball (we own it and it's his), and the ball is past
    the defender: ball x greater than defender_to_beat(start_obs)'s x NOW. False if there
    was no defender to beat."""
    # TODO(node 37a): subgoal_reached calls this for option "dribble". Use the defender's
    #   position on THIS frame: he runs back, and the ball has to get past where he is, not
    #   where he was. If a pass counts here (another player has the ball), dribble and pass
    #   pay for the same thing.
    if defender_to_beat(start_obs) == -1:
        return False

    return (
        we_own_ball(obs)
        and carrier(start_obs) == carrier(obs)
        and obs["right_team"][defender_to_beat(start_obs)][0] < obs["ball"][0]
    )


def carry_to_lane(start_obs: RawObs, obs: RawObs, lane_name: Lane) -> bool:
    """Variation 2: carrier(start_obs) still has the ball, it is in `lane_name`, and it is at least
    FORWARD_METRES further forward than at the start. The same player, so a pass never counts."""
    return (
        we_own_ball(obs)
        and carrier(start_obs) == carrier(obs)
        and lane(float(obs["ball"][1])) == lane_name
        and (float(obs["ball"][0]) - float(start_obs["ball"][0])) * X_METRES >= FORWARD_METRES
    )


def carry_to_box(start_obs: RawObs, obs: RawObs) -> bool:
    """Variation 2: carrier(start_obs) still has the ball and it is inside the penalty area."""
    return carrier(start_obs) == carrier(obs) and ball_in_box(obs)


def shot_released(prev_obs: RawObs, obs: RawObs) -> bool:
    """is_shot read from the ball's flight only (no action): the ball has to leave the foot.
    Provided. With the button counted, holding `shot` while still on the ball paid every step:
    17-20 "shots" per episode and no goals in a 500k-step run (node 37b)."""
    return is_shot(prev_obs, obs)


def subgoal_reached(
    start: OptionStart,
    prev_obs: RawObs,
    obs: RawObs,
    step: int,
    action: int | None = None,
) -> bool:
    """Has the running option's subgoal been reached on this step? Provided: it looks up the
    option's row in OPTIONS and runs that row's test."""
    option = OPTIONS[start.z]
    match option.subgoal:
        case "forward_in_lane":
            assert option.lane is not None
            return forward_in_lane(obs, start.ball_x, option.lane)
        case "ball_in_box":
            return ball_in_box(obs)
        case "is_shot":
            return is_shot(prev_obs, obs, action)
        case "keep_ball":
            return keep_ball(obs, step - start.step, option.horizon)
        case "completed_pass":
            assert start.raw is not None, "OptionStart needs the start frame (raw=...)"
            return completed_pass(start.raw, obs)
        case "beat_defender":
            assert start.raw is not None, "OptionStart needs the start frame (raw=...)"
            return beat_defender(start.raw, obs)
        case "shot_released":
            return shot_released(prev_obs, obs)
        case "carry_to_lane":
            assert start.raw is not None and option.lane is not None
            return carry_to_lane(start.raw, obs, option.lane)
        case "carry_to_box":
            assert start.raw is not None
            return carry_to_box(start.raw, obs)


def option_reward(
    start: OptionStart,
    prev_obs: RawObs,
    obs: RawObs,
    step: int,
    action: int | None = None,
) -> float:
    """r_o (Definition 3): OPTION_REWARD on the step the subgoal is reached, else 0.

    This is the option's OWN reward, separate from the environment's. How the two are mixed
    is a separate decision (execution.combine_rewards, survey 8.3).
    """
    return OPTION_REWARD if subgoal_reached(start, prev_obs, obs, step, action) else 0.0


# --- node 31: beta, with the survey's two admissible forms --------------------


def termination(
    start: OptionStart,
    prev_obs: RawObs,
    obs: RawObs,
    step: int,
    done: bool,
    action: int | None = None,
    mode: TerminationMode = "clock",
    rng: np.random.Generator | None = None,
) -> Termination | None:
    """beta. Returns 'reached', 'failed', 'timeout', or None to keep going.

    'failed' means the attack is over (done) or the other team owns the ball.
    mode="clock":      'timeout' once step - start.step >= HORIZON   (non-Markov)
    mode="geometric":  'timeout' with probability 1 / HORIZON on each step (memoryless; same
                       expected length)
    Order matters: a goal on the last step of `shoot` counts as reached, and `keep`'s subgoal
    lands exactly on the clock's step.
    """
    if subgoal_reached(start, prev_obs, obs, step, action):
        return "reached"
    if done or obs["ball_owned_team"] == THEIRS:
        return "failed"
    if mode == "clock" and step - start.step >= HORIZON:
        return "timeout"
    elif mode == "geometric" and rng.random() < 1 / HORIZON:
        return "timeout"

    return None


# --- provided: run the subgoal tests over recorded episodes (no training) ----


def load_dump(path: Path) -> list[RawObs]:
    """The raw observations of one GRF episode dump (one pickle per step). Container only:
    unpickling needs gfootball installed."""
    frames = []
    with open(path, "rb") as f:
        while True:
            try:
                frames.append(pickle.load(f)["observation"])
            except EOFError:
                return frames


def replay_option(frames: list[RawObs], z: int) -> list[tuple[int, int, Termination]]:
    """Run option z over a recorded episode, restarting it each time it ends. Returns
    (start_step, end_step, reason) for each run. This is thesis_ideas 6.2: do the tests fire
    when your eye says they should?"""
    runs, start = [], OptionStart(z, 0, float(frames[0]["ball"][0]), frames[0])
    for t in range(1, len(frames)):
        reason = termination(
            start, frames[t - 1], frames[t], t, done=t == len(frames) - 1
        )
        if reason:
            runs.append((start.step, t, reason))
            start = OptionStart(z, t, float(frames[t]["ball"][0]), frames[t])
    return runs


def main(argv: list[str]) -> int:
    if "--option-set" in argv:
        i = argv.index("--option-set")
        use_option_set(cast(OptionSetName, argv[i + 1]))
        argv = argv[:i] + argv[i + 2 :]
    root = Path(argv[0] if argv else "/work/runs/demo")
    for dump in sorted(root.glob("*/*.dump")):
        frames = load_dump(dump)
        print(f"\n{dump.parent.name}/{dump.name}  ({len(frames)} steps)")
        for o in OPTIONS:
            runs = replay_option(frames, o.z)
            counts = {
                k: sum(r[2] == k for r in runs)
                for k in ("reached", "failed", "timeout")
            }
            ones = sum(1 for r in runs if r[2] == "reached" and r[1] - r[0] == 1)
            print(f"  {o.name:10s} {counts}  reached after 1 step: {ones}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
