"""Node 25 · clipping, which is PPO. Your code: pg_basics.ppo_clip."""

import numpy as np

from hrl import pg_basics as pg
from hrl.tests._support import close

NODE, TITLE, LAYER, NEEDS = 25, "ppo_clip (and the surrogate)", 1, (22,)


CASES = [(1.5, +2.0, 2.4), (0.5, +2.0, 1.0), (0.5, -2.0, -1.6), (1.5, -2.0, -3.0), (1.1, +2.0, 2.2)]


def test_unit_five_cases():
    # the five (ratio, advantage, expected) in CASES: four outside the band, one inside.
    # Work each out by hand first. Every failing case is reported, not only the first.
    wrong = [f"ratio={r}, A={a:+}: got {float(pg.ppo_clip(r, a)):.4f}, expected {w}"
             for r, a, w in CASES if abs(float(pg.ppo_clip(r, a)) - w) > 1e-9]
    assert not wrong, f"{len(wrong)} of {len(CASES)} cases wrong:\n         " + "\n         ".join(wrong)


def test_unit_arrays_and_eps():
    got = pg.ppo_clip(np.array([1.5, 0.5]), np.array([2.0, 2.0]), eps=0.08)
    assert np.allclose(got, [2.16, 1.0]), f"element-wise with eps=0.08: got {got}"


def _batch():
    return pg.sample_batch(0.0, 40, np.random.default_rng(5))       # 40 episodes played at theta_old = 0


def _slope(f, x, h=1e-6):
    return (f(x + h) - f(x)) / h


def test_stack():
    # Inside the band the surrogate's slope IS your reweighted gradient (node 22); past it, 0.
    a, r = _batch()
    sur = lambda th: pg.surrogate(th, 0.0, a, r)
    close(_slope(sur, 0.1), pg.reweighted_gradient(0.1, 0.0, a, r), 1e-4,
          "theta = 0.1: every ratio is within 1 +- 0.2, so slope of surrogate = reweighted gradient")
    close(_slope(sur, 1.0), 0.0, 1e-9, "theta = 1: a1's ratio is 1.462 > 1.2, so the clip flattens it: slope 0")


def demo():
    """ppo_clip(ratio, A) for a row of ratios, once with A = +2 and once with A = -2."""
    ratios = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
    print("  ratio  " + " ".join(f"{x:6.2f}" for x in ratios))
    for adv in (2.0, -2.0):
        print(f"  A={adv:+.0f}   " + " ".join(f"{float(pg.ppo_clip(x, adv)):6.2f}" for x in ratios))
    a, r = _batch()
    print("\n  one batch of 40 episodes played at theta_old = 0 (R plays the advantage's role):")
    print(f"  {'theta':>6s} {'a1 ratio':>9s} {'surrogate':>10s} {'its slope':>10s} {'your reweighted_gradient':>26s}")
    for th in (0.0, 0.1, 0.5, 1.0):
        sur = lambda t: pg.surrogate(t, 0.0, a, r)
        ratio_a1 = pg.probs(th)[0] / pg.probs(0.0)[0]
        print(f"  {th:6.1f} {ratio_a1:9.3f} {sur(th):10.3f} {_slope(sur, th):10.3f} {pg.reweighted_gradient(th, 0.0, a, r):26.3f}")
