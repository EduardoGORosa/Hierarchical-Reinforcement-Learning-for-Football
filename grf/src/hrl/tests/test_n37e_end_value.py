"""Node 37e · reward-respecting options (--end-value-from). Written by Claude: OptionLearningEnv(end_value=...)."""

import numpy as np

from hrl.tests._support import needs

NODE, TITLE, LAYER, NEEDS = "37e", "end value paid when an option ends", 5, (34,)


def test_unit_end_value():
    needs("gymnasium")
    from hrl import execution as E, options as O
    from hrl.toy import KnownAnswerPitch, RIGHT
    O.use_option_set("actions")
    try:
        plain = E.OptionLearningEnv(KnownAnswerPitch(x0=0.0), rng=np.random.default_rng(3))
        paid = E.OptionLearningEnv(KnownAnswerPitch(x0=0.0), rng=np.random.default_rng(3),
                                   end_value=lambda o: 0.5, end_value_weight=2.0)
        plain.reset(); paid.reset()
        ends = 0
        for _ in range(60):
            _, r0, t0, _, i0 = plain.step(RIGHT)
            _, r1, t1, _, i1 = paid.step(RIGHT)
            assert i0["option_end"] == i1["option_end"]
            expect = 1.0 if (i1["option_end"] is not None and not t1) else 0.0
            assert abs((r1 - r0) - expect) < 1e-9, f"end value paid {r1 - r0}, expected {expect} ({i1['option_end']})"
            ends += expect > 0
            if t1:
                break
        assert ends >= 1, "the test must see at least one option end"
    finally:
        O.use_option_set("zones")


def test_stack():
    needs("gymnasium")
    from hrl import execution as E
    from hrl.toy import KnownAnswerPitch
    env = E.OptionLearningEnv(KnownAnswerPitch(x0=0.0))
    assert env.end_value is None, "default: rewards exactly as in node 33"
