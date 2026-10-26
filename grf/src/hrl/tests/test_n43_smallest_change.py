"""Node 43 · the tilt is the smallest KL change that reaches its style score. No new code: this
stacks your styled_logits (39), softmax (38) and kl (40)."""

import numpy as np

from hrl import style

NODE, TITLE, LAYER, NEEDS = 43, "the tilt is the smallest change", 6, (38, 39, 40)


def test_stack():
    p, pref = style.EXAMPLE_TRAINED, style.WING_PLAY
    q = style.styled_probs(p, pref, 1.0)
    # directions that keep both the total (1) and the style score fixed
    null = np.linalg.svd(np.stack([np.ones(6), np.log(pref)]))[2][2:]
    rng, base = np.random.default_rng(0), style.kl(q, p)
    for _ in range(200):
        d = rng.normal(size=len(null)) @ null
        other = q + 0.02 * d / np.abs(d).max()
        assert abs(style.style_score(other, pref) - style.style_score(q, pref)) < 1e-9
        assert style.kl(other, p) > base, "found a same-style distribution closer to the trained policy"
