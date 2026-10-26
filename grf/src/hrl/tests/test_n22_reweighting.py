"""Node 22 · reusing a batch from an older policy. Your code: pg_basics.reweighted_gradient."""

import numpy as np

from hrl import pg_basics as pg
from hrl.tests._support import close, swap, table

NODE, TITLE, LAYER, NEEDS = 22, "reweighted_gradient", 1, (24,)

FAKE_WRITES = table("episode_writes", {(1.0, 0, 10.0): 2.689414, (1.0, 1, 0.0): 0.0, (1.0, 1, 4.0): -2.924234},
                    key=lambda theta, action, reward: (theta, int(action), float(reward)))


def test_unit_reweighted():
    with swap(pg, episode_writes=FAKE_WRITES):
        close(pg.reweighted_gradient(1.0, 0.0, np.array([0, 0, 1]), np.array([10.0, 10.0, 0.0])),
              2.621492, 1e-5, "3 episodes played at theta_old = 0")
        close(pg.reweighted_gradient(1.0, 0.0, np.array([0, 1, 1]), np.array([10.0, 0.0, 4.0])),
              0.786448, 1e-5, "3 other episodes (the reward 4 is made up)")


def test_stack():
    a, r = pg.sample_batch(0.0, 200_000, np.random.default_rng(1))            # old policy
    close(pg.batch_gradient(1.0, a, r), 1.345, 0.02, "naive average of a stale batch (WRONG on purpose: 1.345)")
    close(pg.reweighted_gradient(1.0, 0.0, a, r), 1.966, 0.03, "reweighted stale batch")


def demo():
    """2,000 small batches (20 episodes each) from three sources, all estimating dJ/dtheta at
    theta = 1 with your code. For each source it prints the average of the 2,000 estimates, and
    the swing: their standard deviation, i.e. how far ONE batch's estimate typically lands from
    that average."""
    rng = np.random.default_rng(0)
    sources = [("played at theta = 1 (on-policy), batch_gradient", 1.0),
               ("played at theta = 0, reweighted_gradient", 0.0),
               ("played at theta = -1, reweighted_gradient", -1.0)]
    print(f"true slope of J at theta = 1: {(pg.objective(1 + 1e-6) - pg.objective(1)) / 1e-6:.3f}\n")
    print(f"  {'batches of 20 episodes ...':50s} {'average':>8s} {'swing':>7s}")
    for label, theta_old in sources:
        est = []
        for _ in range(2000):
            a, r = pg.sample_batch(theta_old, 20, rng)
            est.append(pg.batch_gradient(1.0, a, r) if theta_old == 1.0 else pg.reweighted_gradient(1.0, theta_old, a, r))
        print(f"  {label:50s} {np.mean(est):8.3f} {np.std(est):7.3f}")
