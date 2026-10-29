"""Node 37d · initiation sets for the action options (--initiation). Written by Claude: options.Option.initiation."""

import numpy as np

from hrl.tests._support import needs

NODE, TITLE, LAYER, NEEDS = "37d", "initiation sets (--initiation)", 5, ("37a", 36)


def _frame(**kw):
    f = dict(ball=np.array([0.62, 0.0, 0.1]), ball_owned_team=0, ball_owned_player=1,
             left_team=np.array([[-1.0, 0.0], [0.6, 0.0], [0.7, 0.2], [0.7, -0.2]]),
             right_team=np.array([[1.0, 0.0], [0.75, 0.0]]), right_team_roles=np.array([0, 1]))
    f.update(kw)
    return f


def test_unit_initiation():
    from hrl import options as O
    O.use_option_set("actions")
    try:
        O.use_initiation(True)
        assert [o.initiation(_frame()) for o in O.OPTIONS] == [True, True, True], "kick-off, ball owned: all three"
        assert [o.initiation(_frame(ball_owned_team=-1)) for o in O.OPTIONS] == [False] * 3, "loose ball: none"
        far = _frame(ball=np.array([0.2, 0.0, 0.1]))     # 43.5 m from goal
        assert [o.initiation(far) for o in O.OPTIONS] == [True, True, False], "shoot needs <= 25 m"
        no_def = _frame(right_team=np.array([[1.0, 0.0]]), right_team_roles=np.array([0]))
        assert [o.initiation(no_def) for o in O.OPTIONS] == [True, False, True], "dribble needs a defender to beat"
        O.use_initiation(False)
        assert all(o.initiation(_frame(ball_owned_team=-1)) for o in O.OPTIONS), "off: available everywhere"
    finally:
        O.use_initiation(False)
        O.use_option_set("zones")


def test_stack():
    needs("gymnasium")
    from hrl import execution as E, options as O
    from hrl.toy import KnownAnswerPitch, RIGHT
    O.use_option_set("actions")
    O.use_initiation(True)
    try:
        env = E.SMDPEnv(KnownAnswerPitch(x0=0.0), lambda o: RIGHT, 0.9, with_primitives=True)
        env.reset()
        _, _, _, _, info = env.step(1)                  # dribble: the toy has no defender to beat
        assert (info["tau"], info["option_end"]) == (1, "unavailable"), info
        learn = E.OptionLearningEnv(KnownAnswerPitch(x0=0.0), rng=np.random.default_rng(0))
        for _ in range(20):                              # x = 0 is 54 m out: only pass can start
            learn.reset()
            assert learn.start.z == 0, learn.start
    finally:
        O.use_initiation(False)
        O.use_option_set("zones")
