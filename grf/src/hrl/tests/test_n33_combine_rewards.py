"""Node 33 · mixing option reward and task reward (survey 8.3). Your code: execution.combine_rewards."""

from hrl.tests._support import close, needs

NODE, TITLE, LAYER, NEEDS = 33, "combine_rewards", 4, ()


def test_unit_three_positions():
    needs("gymnasium")
    from hrl.execution import combine_rewards as c
    assert c(1.0, 1.0, "option") == 1.0 and c(1.0, 0.0, "option") == 0.0, "option only"
    close(c(1.0, 1.0, "both", 0.1), 1.1, 1e-12, "both, weight 0.1")
    close(c(0.0, 1.0, "both", 0.5), 0.5, 1e-12, "both, weight 0.5")
    assert c(0.0, 1.0, "env") == 0.0 and c(1.0, 1.0, "env") == 1.0, "task only"
