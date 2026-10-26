"""Node 23 · the log-gradient, measured by a nudge. Your code: pg_basics.grad_log_prob_by_nudge."""

from hrl import pg_basics as pg
from hrl.tests._support import close

NODE, TITLE, LAYER, NEEDS = 23, "grad_log_prob_by_nudge", 1, ()


def test_unit_nudge():
    close(pg.grad_log_prob_by_nudge(0.0, 0), 0.5, 1e-4, "theta=0, a1")
    close(pg.grad_log_prob_by_nudge(1.0, 0), 0.2689414, 1e-4, "theta=1, a1 (your Check 64)")
    close(pg.grad_log_prob_by_nudge(1.0, 1), -0.7310586, 1e-4, "theta=1, a2")


def test_stack():
    # Node 23's claim, checked with provided code only: the slope of p, divided by p, is
    # the slope of ln p. J = 10 p, so the slope of p is the slope of J divided by 10.
    h = 1e-6
    slope_p = (pg.objective(1.0 + h) - pg.objective(1.0)) / h / 10
    close(pg.grad_log_prob_by_nudge(1.0, 0), slope_p / pg.probs(1.0)[0], 1e-4, "slope of p / p")
