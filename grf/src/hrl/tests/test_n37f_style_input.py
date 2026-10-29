"""Node 37f · style weights as mu's input (--style-input). Written by Claude: execution.StyleConditionedEnv."""

import numpy as np

from hrl.tests._support import needs

NODE, TITLE, LAYER, NEEDS = "37f", "style weights as mu's input", 5, (32,)


def test_unit_style_bonus():
    needs("gymnasium")
    from hrl import execution as E, options as O
    from hrl.toy import KnownAnswerPitch
    shoot = O.SUBGOALS.index("shoot")
    for w_shoot, expect in [(1.0, 0.3), (0.0, 0.0)]:
        w = np.full(E.N_OPTIONS, (1.0 - w_shoot) / (E.N_OPTIONS - 1)); w[shoot] = w_shoot
        env = E.StyleConditionedEnv(E.SMDPEnv(KnownAnswerPitch(x0=0.7), lambda o: O.SHOT_ACTION, 0.9), bonus=0.3)
        env.set_style(w)
        obs, _ = env.reset()
        assert np.allclose(obs[-E.N_OPTIONS:], w), "w is the last N_OPTIONS numbers of mu's observation"
        _, r, _, _, info = env.step(shoot)              # scores on step 1, so shoot is "reached"
        assert info["option_end"] == "reached" and abs(info["style_bonus"] - expect) < 1e-9, info
        assert abs(r - (1.0 + expect)) < 1e-9, r


def test_stack():
    needs("gymnasium")
    from hrl import execution as E
    from hrl.toy import KnownAnswerPitch, RIGHT
    env = E.StyleConditionedEnv(E.SMDPEnv(KnownAnswerPitch(), lambda o: RIGHT, 0.9), rng=np.random.default_rng(0))
    ws = [env.reset()[0][-E.N_OPTIONS:] for _ in range(5)]
    assert all(abs(w.sum() - 1) < 1e-5 for w in ws) and not np.allclose(ws[0], ws[1]), "a new w each episode"
    assert env.observation_space.shape[0] == E.SMDPEnv(KnownAnswerPitch(), lambda o: RIGHT, 0.9).observation_space.shape[0] + E.N_OPTIONS
