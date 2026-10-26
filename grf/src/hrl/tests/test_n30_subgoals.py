"""Node 30 · the six subgoals and the option reward r_o. Your code: options.to_metres, lane,
we_own_ball, is_shot, subgoal_reached, option_reward."""

import numpy as np

from hrl import options as O
from hrl.hints import RawObs
from hrl.tests._support import swap

NODE, TITLE, LAYER, NEEDS = 30, "subgoals and option reward", 3, ()


def obs(ball_xy, owner=O.OURS, ball_dir=(0.0, 0.0)) -> RawObs:
    return {
        "ball": np.array([*ball_xy, 0.1]),
        "ball_owned_team": owner,
        "ball_direction": np.array([*ball_dir, 0.0]),
    }


def z(name):
    return O.SUBGOALS.index(name)


# Fakes for the helpers, so subgoal_reached is tested on its own.
FAKES = dict(
    to_metres=lambda xy: (
        np.asarray(xy, float).reshape(2) * [54.4, 83.6]
    ),  # (x, y) pairs only, like the real one
    lane=lambda y: "left" if y < -0.14 else ("right" if y > 0.14 else "centre"),
    we_own_ball=lambda o: o["ball_owned_team"] == O.OURS,
    is_shot=lambda prev, o, action=None: action == O.SHOT_ACTION,
)


def test_unit_geometry():
    assert np.allclose(O.to_metres([1.0, 0.42]), [54.4, 35.112], atol=0.01), "to_metres"
    assert [O.lane(y) for y in (-0.2, -0.14, 0.0, 0.14, 0.2)] == [
        "left",
        "centre",
        "centre",
        "centre",
        "right",
    ], "lane"
    assert O.we_own_ball(obs((0, 0))) and not O.we_own_ball(obs((0, 0), O.THEIRS)), (
        "we_own_ball"
    )


def test_unit_is_shot():
    with swap(O, we_own_ball=FAKES["we_own_ball"]):
        prev = obs((0.85, 0.02))
        assert O.is_shot(prev, obs((0.85, 0.02)), action=O.SHOT_ACTION), (
            "agent pressed shot with the ball"
        )
        assert O.is_shot(prev, obs((0.90, 0.02), O.NOBODY, (0.05, -0.001))), (
            "fast, at the goal mouth"
        )
        assert not O.is_shot(prev, obs((0.86, 0.02), O.NOBODY, (0.01, 0.0))), (
            "too slow: a touch"
        )
        assert not O.is_shot(prev, obs((0.88, 0.1), O.NOBODY, (0.05, 0.08))), (
            "fast, but heading wide"
        )
        assert O.is_shot(prev, obs((0.90, 0.0), O.NOBODY, (0.05, 0.0235))), (
            "aimed at y = 0.047: inside the margin"
        )
        assert not O.is_shot(prev, obs((0.90, 0.0), O.NOBODY, (0.05, 0.026))), (
            "aimed at y = 0.052: a firm pass"
        )
        assert not O.is_shot(prev, obs((0.90, 0.02), O.NOBODY, (-0.05, 0.0))), (
            "fast, but backwards"
        )
        assert not O.is_shot(
            obs((0.85, 0.02), O.THEIRS), obs((0.90, 0.02), O.NOBODY, (0.05, 0.0))
        ), "their ball, not our shot"
        assert not O.is_shot(
            obs((0.85, 0.02), O.THEIRS),
            obs((0.85, 0.02), O.THEIRS),
            action=O.SHOT_ACTION,
        ), "pressed shot without the ball"


def test_unit_subgoal_reached():
    with swap(O, **FAKES):
        s, prev = O.OptionStart(z("left_wing"), 0, 0.0), obs((0.0, -0.2))
        assert O.subgoal_reached(s, prev, obs((0.10, -0.2)), 5), (
            "5.44 m forward in the left lane"
        )
        assert not O.subgoal_reached(s, prev, obs((0.05, -0.2)), 5), (
            "only 2.72 m forward"
        )
        assert not O.subgoal_reached(s, prev, obs((0.10, 0.0)), 5), (
            "forward, but in the centre lane"
        )
        assert not O.subgoal_reached(s, prev, obs((0.10, -0.2), O.NOBODY), 5), (
            "ball loose"
        )
        b = O.OptionStart(z("enter_box"), 0, 0.5)
        assert O.subgoal_reached(b, prev, obs((0.75, 0.1)), 3), "in the box"
        assert not O.subgoal_reached(b, prev, obs((0.60, 0.1)), 3), "short of the box"
        assert not O.subgoal_reached(b, prev, obs((0.75, 0.30)), 3), "beside the box"
        k = O.OptionStart(z("keep"), 10, 0.0)
        assert O.subgoal_reached(k, prev, obs((0, 0)), 40) and not O.subgoal_reached(
            k, prev, obs((0, 0)), 39
        ), "keep"
        sh = O.OptionStart(z("shoot"), 0, 0.8)
        assert O.subgoal_reached(sh, prev, obs((0.8, 0)), 2, action=O.SHOT_ACTION), (
            "shoot uses is_shot"
        )


def test_unit_option_reward():
    with swap(O, subgoal_reached=lambda s, p, o, step, action=None: step == 7):
        s, o = O.OptionStart(0, 0, 0.0), obs((0, 0))
        assert (
            O.option_reward(s, o, o, 7) == O.OPTION_REWARD
            and O.option_reward(s, o, o, 6) == 0.0
        )


def test_stack():
    # Your helpers, not the fakes: the same subgoal cases must still come out right.
    s, prev = O.OptionStart(z("right_wing"), 0, 0.0), obs((0.0, 0.2))
    assert O.subgoal_reached(s, prev, obs((0.10, 0.2)), 5), (
        "right wing with your lane and to_metres"
    )
    assert O.subgoal_reached(
        O.OptionStart(z("shoot"), 0, 0.8),
        obs((0.85, 0.02)),
        obs((0.90, 0.02), O.NOBODY, (0.05, 0.0)),
        1,
    ), "shoot with your is_shot, no action given"
    assert (
        O.option_reward(
            O.OptionStart(z("enter_box"), 0, 0.5), prev, obs((0.75, 0.0)), 1
        )
        == O.OPTION_REWARD
    )
