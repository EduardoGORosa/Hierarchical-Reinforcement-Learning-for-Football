"""Node 44 · what PPO must store when training runs with a style on. Your code: style.styled_log_prob."""

import numpy as np

from hrl import style
from hrl.tests._support import close, swap

NODE, TITLE, LAYER, NEEDS = 44, "styled_log_prob", 6, (38, 39)


def test_unit_log_prob():
    with swap(style, styled_logits=lambda l, p, lam: np.array([9.0, 9.0, 9.0]),
              softmax=lambda x: np.array([0.1, 0.2, 0.7])):
        close(style.styled_log_prob(np.zeros(3), np.ones(3), 1.0, 2), np.log(0.7), 1e-12,
              "log of the softmax of the styled logits, at the chosen option")


def test_stack():
    logits = np.log(style.EXAMPLE_TRAINED)
    close(style.styled_log_prob(logits, style.WING_PLAY, 1.0, 2), np.log(0.3529), 1e-3, "right wing under wing play")
    close(style.styled_log_prob(logits, style.WING_PLAY, 0.0, 2), np.log(0.20), 1e-9, "lambda=0: the unstyled log-prob")
