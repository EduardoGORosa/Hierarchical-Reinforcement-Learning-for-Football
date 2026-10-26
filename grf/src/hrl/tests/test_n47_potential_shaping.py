"""Node 47 · a possession value as a potential. Your code: evaluate.shaped_reward."""

from hrl import evaluate as ev
from hrl import smdp
from hrl.tests._support import close

NODE, TITLE, LAYER, NEEDS = 47, "shaped_reward", 7, (28,)


def test_unit_shaped():
    close(ev.shaped_reward(0.0, 0.2, 0.5, 0.9, False), 0.25, 1e-12, "0 + 0.9*0.5 - 0.2")
    close(ev.shaped_reward(1.0, 0.6, 0.9, 0.9, True), 0.4, 1e-12, "last step: Phi(s') counts as 0")


def test_stack():
    # Along any episode the shaping telescopes: discounted shaped return = discounted return - Phi(s0).
    # Checked with your option_return (node 28) as the discounted sum.
    r, phi, g = [0.0, 0.0, 0.3, 1.0], [0.1, 0.4, 0.2, 0.7, 0.0], 0.9
    shaped = [ev.shaped_reward(r[t], phi[t], phi[t + 1], g, done=t == 3) for t in range(4)]
    close(smdp.option_return(shaped, g), smdp.option_return(r, g) - phi[0], 1e-12, "telescoping sum")
