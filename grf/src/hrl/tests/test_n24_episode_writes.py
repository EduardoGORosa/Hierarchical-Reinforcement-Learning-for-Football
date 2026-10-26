"""Node 24 · one episode writes reward x log-gradient. Your code: pg_basics.episode_writes and
batch_gradient."""

import numpy as np

from hrl import pg_basics as pg
from hrl.tests._support import close, swap, table

NODE, TITLE, LAYER, NEEDS = 24, "episode_writes, batch_gradient", 1, (23,)

# Your node-23 numbers at theta = 1, as a fake: the unit tests below don't depend on your nudge.
FAKE_GLP = table("grad_log_prob_by_nudge", {(1.0, 0): 0.2689414, (1.0, 1): -0.7310586},
                 key=lambda theta, action, h=None: (theta, action))


def test_unit_episode_writes():
    with swap(pg, grad_log_prob_by_nudge=FAKE_GLP):
        close(pg.episode_writes(1.0, 0, 10.0), 2.689414, 1e-6, "a1 played, reward 10")
        close(pg.episode_writes(1.0, 1, 0.0), 0.0, 1e-9, "a2 played, reward 0")
        close(pg.episode_writes(1.0, 1, 5.0), -3.655293, 1e-6, "a2 played, reward 5 (made up)")


def test_unit_batch_gradient():
    # A fake episode_writes with arbitrary but known values: this tests only the averaging.
    fake = table("episode_writes", {(1.0, 0, 1.0): 10.0, (1.0, 1, 2.0): 21.0, (1.0, 1, 3.0): 31.0},
                 key=lambda theta, action, reward: (theta, int(action), float(reward)))
    with swap(pg, episode_writes=fake):
        close(pg.batch_gradient(1.0, np.array([0, 1, 1]), np.array([1.0, 2.0, 3.0])),
              62.0 / 3, 1e-9, "average of 10, 21, 31 (batch_gradient must call episode_writes)")


def test_stack():
    a, r = pg.sample_batch(1.0, 200_000, np.random.default_rng(0))
    slope_of_J = (pg.objective(1.0 + 1e-6) - pg.objective(1.0)) / 1e-6       # 1.9661
    close(pg.batch_gradient(1.0, a, r), slope_of_J, 0.02, "on-policy batch vs the slope of J")


def demo():
    """Averages of the three candidates from node 24 over 200,000 episodes at theta = 1."""
    a, r = pg.sample_batch(1.0, 200_000, np.random.default_rng(0))
    p = pg.probs(1.0)
    print("200,000 one-action episodes at theta = 1; the batch average of what each writes down:\n")
    print(f"  Candidate R       {np.mean(r):.4f}")
    print(f"  Candidate piR     {np.mean(p[a] * r):.4f}")
    print(f"  Candidate R∇lnπ   {pg.batch_gradient(1.0, a, r):.4f}   (your episode_writes)")
    print(f"\n  slope of J, by nudging J:  {(pg.objective(1 + 1e-6) - pg.objective(1)) / 1e-6:.4f}")
