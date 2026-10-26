"""Node 46 · from five runs to a claim: IQM and a bootstrap interval. Your code: evaluate.iqm and
evaluate.bootstrap_ci."""

import numpy as np

from hrl import evaluate as ev

NODE, TITLE, LAYER, NEEDS = 46, "iqm, bootstrap_ci", 7, ()


def test_unit_iqm():
    assert abs(ev.iqm(np.arange(1, 9)) - 4.5) < 1e-12, "n=8: average of 3, 4, 5, 6"
    assert abs(ev.iqm([0.5, 0.6, 0.7, 0.8, 0.99]) - 0.7) < 1e-12, "n=5: drop one from each end"


def test_unit_bootstrap():
    lo, hi = ev.bootstrap_ci([0.7] * 5, rng=np.random.default_rng(0))
    assert lo == hi == 0.7, "no spread between runs -> zero-width interval"
    runs = [0.62, 0.55, 0.71, 0.48, 0.66]
    lo, hi = ev.bootstrap_ci(runs, rng=np.random.default_rng(0))
    assert lo < np.mean(runs) < hi and 0.48 <= lo and hi <= 0.71, (lo, hi)
    lo2, hi2 = ev.bootstrap_ci(runs * 4, rng=np.random.default_rng(0))
    assert hi2 - lo2 < hi - lo, "four times the runs must give a narrower interval"


def test_stack():
    runs = [0.62, 0.55, 0.71, 0.48, 0.66]
    lo, hi = ev.bootstrap_ci(runs, stat=ev.iqm, rng=np.random.default_rng(0))
    assert lo <= ev.iqm(runs) <= hi, f"IQM {ev.iqm(runs):.3f} outside its own interval ({lo:.3f}, {hi:.3f})"
