"""Node 40 · KL divergence. Your code: style.kl."""

from hrl import style
from hrl.tests._support import close

NODE, TITLE, LAYER, NEEDS = 40, "kl", 6, ()


def test_unit_kl():
    close(style.kl(style.EXAMPLE_TRAINED, style.EXAMPLE_TRAINED), 0.0, 1e-12, "KL of a distribution with itself")
    close(style.kl([0.5, 0.5], [0.25, 0.75]), 0.143841, 1e-6, "two outcomes")
    close(style.kl([0.25, 0.75], [0.5, 0.5]), 0.130812, 1e-6, "the other way round: not symmetric")
