"""Node 32 · call-and-return: one step of mu runs a whole option. Your code: execution.SMDPEnv.step
(the loop)."""

import numpy as np

from hrl.tests._support import needs, swap

NODE, TITLE, LAYER, NEEDS = 32, "SMDPEnv.step (call-and-return)", 4, (28, 31)


def test_unit_loop():
    needs("gymnasium")
    from hrl import execution as E, options as O, smdp
    from hrl.toy import KnownAnswerPitch, RIGHT
    seen = []
    # Fakes: beta ends every option after exactly 3 steps; option_return is a marker (42) so
    # the test can see that your loop called it; the option policy records what it was shown.
    beta3 = lambda start, prev, o, step, done, action=None, mode="clock", rng=None: "reached" if step - start.step >= 3 else None
    with swap(O, termination=beta3), swap(smdp, option_return=lambda rewards, gamma: 42.0):
        env = E.SMDPEnv(KnownAnswerPitch(), lambda o: (seen.append(o), RIGHT)[1], gamma=0.9)
        env.reset()
        obs, r, term, trunc, info = env.step(O.SUBGOALS.index("centre"))
    assert (info["tau"], info["option_end"], r) == (3, "reached", 42.0), info
    assert np.isclose(obs[0], 0.06), f"3 steps right should put the ball at x=0.06, got {obs[0]}"
    assert len(seen) == 3 and seen[0].shape == (2 + E.N_OPTIONS,), "the option policy sees obs + one-hot z"
    assert seen[0][2 + O.SUBGOALS.index("centre")] == 1.0 and seen[0][2:].sum() == 1.0, "one-hot at z"


def test_stack():
    needs("gymnasium")
    from hrl import execution as E, options as O
    from hrl.toy import KnownAnswerPitch, RIGHT
    env = E.SMDPEnv(KnownAnswerPitch(), lambda o: RIGHT, gamma=0.9)
    env.reset()
    _, r, term, _, info = env.step(O.SUBGOALS.index("centre"))       # 5 m = 5 steps of 0.02
    assert (info["tau"], info["option_end"], r, term) == (5, "reached", 0.0, False), info
    _, _, _, _, info = env.step(O.SUBGOALS.index("keep"))            # keep: reached on the clock
    assert (info["tau"], info["option_end"]) == (O.HORIZON, "reached"), info
    env = E.SMDPEnv(KnownAnswerPitch(x0=0.7), lambda o: O.SHOT_ACTION, gamma=0.9)
    env.reset()
    _, r, term, _, info = env.step(O.SUBGOALS.index("shoot"))        # scores on step 1
    assert (info["tau"], info["option_end"], r, term) == (1, "reached", 1.0, True), info
