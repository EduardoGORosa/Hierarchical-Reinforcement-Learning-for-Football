"""Node 38 · softmax. Your code: style.softmax."""

import numpy as np

from hrl import style

NODE, TITLE, LAYER, NEEDS = 38, "softmax", 6, ()


def test_unit_softmax():
    s = style.softmax(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(s, [0.09003057, 0.24472847, 0.66524096]), f"got {s}"
    assert np.allclose(s, style.softmax(np.array([101.0, 102.0, 103.0]))), "adding a constant changes nothing"
    big = style.softmax(np.array([1000.0, 1001.0]))
    assert np.all(np.isfinite(big)) and np.allclose(big, style.softmax(np.array([0.0, 1.0]))), "overflow: subtract the max"
