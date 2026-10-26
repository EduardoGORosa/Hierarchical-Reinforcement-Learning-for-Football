"""Node 26 · a baseline. Your code: pg_basics.episode_writes_with_baseline."""

from hrl import pg_basics as pg
from hrl.tests._support import close, swap, table

NODE, TITLE, LAYER, NEEDS = 26, "episode_writes_with_baseline", 1, (23,)

FAKE_GLP = table("grad_log_prob_by_nudge", {(1.0, 0): 0.2689414, (1.0, 1): -0.7310586},
                 key=lambda theta, action, h=None: (theta, action))


def test_unit_baseline():
    with swap(pg, grad_log_prob_by_nudge=FAKE_GLP):
        close(pg.episode_writes_with_baseline(1.0, 0, 10.0, 5.0), 1.344707, 1e-6, "a1, R=10, b=5")
        close(pg.episode_writes_with_baseline(1.0, 1, 0.0, 5.0), 3.655293, 1e-6, "a2, R=0, b=5")


def test_stack():
    J = pg.objective(1.0)                                  # 7.311
    for b in (0.0, 5.0, J):
        m, _ = pg.exact_mean_and_variance(1.0, b)
        close(m, 1.966119, 1e-3, f"mean with baseline {b:.3f}")
    v0, v_small, v_J = (pg.exact_mean_and_variance(1.0, b)[1] for b in (0.0, 2.689, J))
    assert v_small < v0 < v_J, (f"variances: b=0 -> {v0:.3f}, b=2.689 -> {v_small:.3f}, b=J -> {v_J:.3f}; "
                                "expected the middle one smallest and b=J largest")


def demo():
    """For five baselines b, at theta = 1: what an a1 episode and an a2 episode write
    ((R - b) x log-gradient, your episode_writes_with_baseline), then EXACTLY (no sampling,
    pg.exact_mean_and_variance) the average over episodes and the swing of one episode
    (square root of the variance)."""
    J = pg.objective(1.0)
    print("theta = 1; one episode; exact, no sampling\n")
    print(f"  {'baseline b':>12s} {'a1 writes':>10s} {'a2 writes':>10s} {'average':>9s} {'swing':>7s}")
    for label, b in [("0", 0.0), ("2.689", 2.689414), ("5", 5.0), ("7.311 (= J)", J), ("10", 10.0)]:
        w1 = pg.episode_writes_with_baseline(1.0, 0, 10.0, b)
        w2 = pg.episode_writes_with_baseline(1.0, 1, 0.0, b)
        m, v = pg.exact_mean_and_variance(1.0, b)
        print(f"  {label:>12s} {w1:10.3f} {w2:10.3f} {m:9.4f} {v ** 0.5:7.3f}")
