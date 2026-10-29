"""Node 37a · subgoals that fit a 3v1 attack: pass, dribble, shoot. Your code: options.carrier,
completed_pass, defender_to_beat, beat_defender."""

import numpy as np

from hrl import options as O
from hrl.hints import RawObs
from hrl.tests._support import swap

NODE, TITLE, LAYER, NEEDS = "37a", "action subgoals (pass, dribble, shoot)", 3, (30, 31)

OUR_START = ((-1.0, 0.0), (0.6, 0.0), (0.7, 0.2), (0.7, -0.2))  # academy_3_vs_1_with_keeper


def frame(ball_xy, team=O.OURS, player=1, ours=OUR_START, theirs=((1.0, 0.0), (0.75, 0.0)),
          roles=(0, 1)) -> RawObs:
    """A raw frame: their keeper first (role 0), then a centre back (role 1), as in 3v1."""
    return {"ball": np.array([*ball_xy, 0.1]), "ball_direction": np.zeros(3),
            "ball_owned_team": team, "ball_owned_player": player if team != O.NOBODY else -1,
            "left_team": np.array(ours, float), "right_team": np.array(theirs, float),
            "right_team_roles": np.array(roles)}


LOOSE_IS_1 = lambda o: int(o["ball_owned_player"]) if o["ball_owned_team"] == O.OURS else 1


def test_unit_carrier():
    assert O.carrier(frame((0.62, 0.0), O.OURS, 2)) == 2, "we own it: the owner"
    # Loose ball at (0, 0). Player 1 is 0.10 x-units away (5.4 m), player 2 0.07 y-units (5.9 m):
    # nearer in pitch units, further in metres.
    ours = ((-1.0, 0.0), (0.10, 0.0), (0.0, 0.07))
    assert O.carrier(frame((0.0, 0.0), O.NOBODY, ours=ours)) == 1, "loose: nearest in METRES"


def test_unit_completed_pass():
    with swap(O, carrier=LOOSE_IS_1):
        start = frame((0.62, 0.0), O.OURS, 1)
        assert O.completed_pass(start, frame((0.70, 0.2), O.OURS, 2)), "1 -> 2, received"
        assert not O.completed_pass(start, frame((0.66, 0.0), O.OURS, 1)), "1 still has it"
        assert not O.completed_pass(start, frame((0.66, 0.1), O.NOBODY)), "in flight"
        assert not O.completed_pass(start, frame((0.70, 0.2), O.THEIRS, 2)), "intercepted by their 2"
        kickoff = frame((0.62, 0.0), O.NOBODY)  # loose at the start: the fake says player 1's
        assert O.completed_pass(kickoff, frame((0.70, 0.2), O.OURS, 3)), "from a loose start"


def test_unit_defender_to_beat():
    assert O.defender_to_beat(frame((0.62, 0.0))) == 1, "the centre back in front"
    behind = frame((0.62, 0.0), theirs=((1.0, 0.0), (0.50, 0.0)))
    assert O.defender_to_beat(behind) == -1, "only the keeper is in front: nobody to beat"
    # Two defenders in front of a ball at (0.62, 0): index 1 is 0.10 x-units ahead (5.4 m),
    # index 2 is 0.02 ahead and 0.07 across (5.9 m). Nearer in pitch units, further in metres.
    two = frame((0.62, 0.0), theirs=((1.0, 0.0), (0.72, 0.0), (0.64, 0.07)), roles=(0, 1, 1))
    assert O.defender_to_beat(two) == 1, "nearest in METRES"


def test_unit_beat_defender():
    with swap(O, carrier=LOOSE_IS_1, defender_to_beat=lambda s: 1):
        start = frame((0.62, 0.0), O.OURS, 1)
        assert O.beat_defender(start, frame((0.78, 0.0), O.OURS, 1)), "past the CB at 0.75"
        assert not O.beat_defender(start, frame((0.70, 0.0), O.OURS, 1)), "not past him yet"
        assert not O.beat_defender(start, frame((0.78, 0.0), O.OURS, 2)), "player 2 has it: a pass"
        assert not O.beat_defender(start, frame((0.78, 0.0), O.NOBODY)), "ball loose"
        ran_back = frame((0.78, 0.0), O.OURS, 1, theirs=((1.0, 0.0), (0.85, 0.0)))
        assert not O.beat_defender(start, ran_back), "he ran back to 0.85: compare with NOW"
    with swap(O, carrier=LOOSE_IS_1, defender_to_beat=lambda s: -1):
        assert not O.beat_defender(frame((0.62, 0.0)), frame((0.90, 0.0))), "nobody to beat"


def test_stack():
    # Your helpers, termination and replay_option, with the "actions" set (HORIZON 20).
    O.use_option_set("actions")
    try:
        z = O.SUBGOALS.index
        # Player 1 carries the ball 0.02 per step past the CB at 0.75 (x = 0.76 at step 7).
        dribble = [frame((0.62 + 0.02 * t, 0.0)) for t in range(12)]
        assert O.replay_option(dribble, z("dribble")) == [(0, 7, "reached"), (7, 11, "failed")], \
            O.replay_option(dribble, z("dribble"))
        # Player 1 holds it for 4 steps, it travels loose for 5, player 2 receives at step 9.
        passing = [frame((0.62, 0.0))] * 4 + [frame((0.64, 0.04 * k), O.NOBODY) for k in range(5)] \
            + [frame((0.70, 0.2), O.OURS, 2)] * 4
        assert O.replay_option(passing, z("pass")) == [(0, 9, "reached"), (9, 12, "failed")], \
            O.replay_option(passing, z("pass"))
        # Standing still with the CB in front: the actions set's clock is 20 steps.
        still = [frame((0.62, 0.0))] * 25
        assert O.replay_option(still, z("dribble")) == [(0, 20, "timeout"), (20, 24, "failed")], \
            O.replay_option(still, z("dribble"))
    finally:
        O.use_option_set("zones")
