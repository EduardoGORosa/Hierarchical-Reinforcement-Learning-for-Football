"""Node 27 · the finite geometric sum. Your code: smdp.geometric_sum."""

from hrl import smdp
from hrl.tests._support import close

NODE, TITLE, LAYER, NEEDS = 27, "geometric_sum", 2, ()


def test_unit_closed_form():
    for g, tau, want in [(0.9, 3, 2.71), (0.99, 30, 26.029962), (0.5, 1, 1.0), (0.993, 400, 134.255105)]:
        close(smdp.geometric_sum(g, tau), want, 1e-3, f"gamma={g}, tau={tau}")


def test_unit_gamma_one():
    close(smdp.geometric_sum(1.0, 7), 7.0, 1e-12, "gamma=1: the closed form divides by zero here")


def demo():
    """For options lasting tau steps at gamma = 0.993 (the GRF baseline's gamma): your
    geometric_sum (what a reward of 1 on every step of the option is worth), the loop that adds
    it term by term, and gamma**tau (the discount mu must apply to the state where the option
    ends). tau = 137 is the median built-in-AI episode on 3v1 (brief, section 3)."""
    g = 0.993
    print(f"gamma = {g}\n")
    print(f"  {'tau':>5s} {'geometric_sum':>14s} {'loop, term by term':>19s} {'gamma**tau':>11s}")
    for tau in (1, 3, 30, 137):
        loop = sum(g**k for k in range(tau))
        print(f"  {tau:5d} {smdp.geometric_sum(g, tau):14.4f} {loop:19.4f} {g**tau:11.4f}")
