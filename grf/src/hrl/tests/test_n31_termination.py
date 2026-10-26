"""Node 31 · beta: reached, failed or timeout, with the clock or the geometric form.
Your code: options.termination."""

import numpy as np

from hrl import options as O
from hrl.hints import RawObs
from hrl.tests._support import swap

NODE, TITLE, LAYER, NEEDS = 31, "termination (clock and geometric)", 3, (30,)


def obs(ball_xy, owner=O.OURS, ball_dir=(0.0, 0.0)) -> RawObs:
    return {"ball": np.array([*ball_xy, 0.1]), "ball_owned_team": owner,
            "ball_direction": np.array([*ball_dir, 0.0])}


NEVER = lambda s, p, o, step, action=None: False           # a subgoal that is never reached
AT_30 = lambda s, p, o, step, action=None: step - s.step >= 30


def test_unit_clock():
    with swap(O, subgoal_reached=NEVER):
        s, o = O.OptionStart(0, 0, 0.0), obs((0, 0))
        assert O.termination(s, o, obs((0, 0), O.THEIRS), 4, False) == "failed", "they won the ball"
        assert O.termination(s, o, o, 7, True) == "failed", "the attack ended"
        assert O.termination(s, o, o, 30, False) == "timeout", "clock, step 30"
        assert O.termination(s, o, o, 29, False) is None, "step 29: keep going"


def test_unit_order():
    with swap(O, subgoal_reached=AT_30):
        s, o = O.OptionStart(0, 0, 0.0), obs((0, 0))
        assert O.termination(s, o, o, 30, False) == "reached", "reached beats timeout on the same step"
        assert O.termination(s, o, o, 30, True) == "reached", "reached beats failed (a goal ends the attack)"


def test_unit_geometric():
    with swap(O, subgoal_reached=NEVER):
        s, o, rng, lengths = O.OptionStart(0, 0, 0.0), obs((0, 0)), np.random.default_rng(0), []
        for _ in range(4000):
            t = 1
            while O.termination(s, o, o, t, False, mode="geometric", rng=rng) is None:
                t += 1
            lengths.append(t)
        assert abs(np.mean(lengths) - O.HORIZON) < 1.5, f"mean length {np.mean(lengths):.1f}, expected ~{O.HORIZON}"
        assert max(lengths) > 2 * O.HORIZON, "geometric termination has no hard limit"


def test_stack():
    # A synthetic attack: we carry the ball straight up the middle, 0.02 per step, for 40
    # steps, then they win it. replay_option restarts each option whenever it ends.
    frames = [obs((i * 0.02, 0.0), ball_dir=(0.02, 0.0)) for i in range(41)] + [obs((0.8, 0.0), O.THEIRS)]
    runs = lambda name: O.replay_option(frames, O.SUBGOALS.index(name))
    assert runs("centre") == [(i, i + 5, "reached") for i in range(0, 40, 5)] + [(40, 41, "failed")], runs("centre")
    assert runs("keep") == [(0, 30, "reached"), (30, 41, "failed")], runs("keep")
    # enter_box's subgoal is ALREADY true once the ball is in the box, so the restarted option
    # ends after 1 step, four times running. This is the design question node 30 raises.
    # (The ball is first past BOX_X = 0.708 at step 36, x = 0.72.)
    assert runs("enter_box") == [(0, 30, "timeout"), (30, 36, "reached")] + \
        [(t, t + 1, "reached") for t in range(36, 40)] + [(40, 41, "failed")], runs("enter_box")
