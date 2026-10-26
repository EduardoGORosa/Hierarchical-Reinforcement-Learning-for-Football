"""Node 29 · GAE for mu, discounting by gamma^tau. Your code: smdp.smdp_gae."""

import numpy as np

from hrl import smdp

NODE, TITLE, LAYER, NEEDS = 29, "smdp_gae", 2, (28,)


def test_unit_tau_one_is_flat():
    rng = np.random.default_rng(0)
    r, v = rng.normal(size=6), rng.normal(size=6)
    d = np.array([0, 0, 1, 0, 0, 0.0])
    got = smdp.smdp_gae(r, np.ones(6, dtype=int), v, 0.3, d, 0.99, 0.95)
    assert np.allclose(got, smdp.flat_gae(r, v, 0.3, d, 0.99, 0.95)), "with every tau = 1 it must equal flat_gae"


def test_unit_two_decisions():
    # Decision 0 lasted 2 steps and paid 0; decision 1 lasted 3 and paid 1; then the attack ended.
    got = smdp.smdp_gae([0.0, 1.0], [2, 3], [0.5, 0.8], 0.0, [0, 1], 0.9, 1.0)
    assert np.allclose(got, [0.31, 0.2]), f"got {np.round(got, 4)}, expected [0.31, 0.2]"


def test_stack():
    # Three options, from their raw step rewards: your option_return turns each into one
    # reward, your smdp_target chains them backwards, and with lambda = 1 your smdp_gae must
    # give exactly (that return) - V.
    g, lists, taus, V = 0.9, [[0, 0], [0, 0, 0, 1], [0.5]], [2, 4, 1], [0.3, 0.6, 0.2]
    R = [smdp.option_return(l, g) for l in lists]
    G, nxt = [0.0] * 3, 0.0
    for t in reversed(range(3)):
        G[t] = nxt = smdp.smdp_target(R[t], taus[t], nxt, g, done=t == 2)
    assert np.allclose(G, [0.8562105, 1.05705, 0.5]), f"returns from option_return + smdp_target: {np.round(G, 6)}"
    adv = smdp.smdp_gae(R, taus, V, 0.0, [0, 0, 1], g, 1.0)
    assert np.allclose(adv, np.array(G) - V), f"smdp_gae (lambda=1) {np.round(adv, 6)} vs G - V {np.round(np.array(G) - V, 6)}"
