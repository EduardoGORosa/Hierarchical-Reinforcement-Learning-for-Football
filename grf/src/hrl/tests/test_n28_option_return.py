"""Node 28 · one transition of mu. Your code: smdp.option_return and smdp.smdp_target."""

from hrl import smdp
from hrl.tests._support import close

NODE, TITLE, LAYER, NEEDS = 28, "option_return, smdp_target", 2, (27,)


def test_unit_option_return():
    close(smdp.option_return([0.0, 0.0, 1.0], 0.9), 0.81, 1e-12, "goal on the option's 3rd step")
    close(smdp.option_return([1.0], 0.9), 1.0, 1e-12, "one-step option: not discounted")
    close(smdp.option_return([0.5, 0.0, 0.0, 2.0], 0.5), 0.75, 1e-12, "two rewards")


def test_unit_smdp_target():
    close(smdp.smdp_target(0.81, 3, 2.0, 0.9, False), 2.268, 1e-12, "bootstrap with 0.9^3")
    close(smdp.smdp_target(0.81, 3, 2.0, 0.9, True), 0.81, 1e-12, "attack over: nothing to bootstrap")


def test_stack():
    # A reward of 1 on every step of a tau-step option is worth exactly the geometric sum.
    for g, tau in [(0.9, 5), (0.993, 30)]:
        close(smdp.option_return([1.0] * tau, g), smdp.geometric_sum(g, tau), 1e-9,
              f"option_return of {tau} ones vs geometric_sum, gamma={g}")
