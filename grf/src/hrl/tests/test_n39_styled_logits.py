"""Node 39 · the style rule. Your code: style.styled_logits."""

import numpy as np

from hrl import style

NODE, TITLE, LAYER, NEEDS = 39, "styled_logits", 6, (38,)


def test_unit_styled_logits():
    got = style.styled_logits(np.array([0.0, 1.0]), np.array([0.5, 0.25]), 2.0)
    assert np.allclose(got, [2 * np.log(0.5), 1 + 2 * np.log(0.25)]), f"got {got}"
    assert np.allclose(style.styled_logits(np.array([0.3, 0.1]), np.array([0.9, 0.1]), 0.0), [0.3, 0.1]), "lambda=0"


def test_stack():
    q = style.styled_probs(style.EXAMPLE_TRAINED, style.WING_PLAY, 1.0)
    want = np.array([0.2647, 0.0588, 0.3529, 0.1471, 0.1176, 0.0588])
    assert np.allclose(q, want, atol=1e-4), f"lambda=1 wing play: got {np.round(q, 4)}, expected {want}"
    assert np.allclose(style.styled_probs(style.EXAMPLE_TRAINED, np.full(6, 1 / 6), 3.0), style.EXAMPLE_TRAINED), \
        "equal preferences must change nothing"
    assert abs(style.share_of_choices_changed(q, style.EXAMPLE_TRAINED) - 0.268) < 1e-3, "about 27 in 100 choices change"
