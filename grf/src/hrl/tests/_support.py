"""Helpers the tests share. Short on purpose: read it once, then the tests read easily."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager


class Skip(Exception):
    """Raised when a test can't run here, e.g. it needs gymnasium and you're on the host."""


def needs(module: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise Skip(f"needs {module}: run it in the container, ./run.sh hrl check ...")


@contextmanager
def swap(module, **fakes):
    """Temporarily replace names inside `module` with fakes, then put the originals back.

    This isolates a unit. For the duration of the `with` block your code calls the fake
    instead of your real lower-node function:

        with swap(pg, grad_log_prob_by_nudge=fake):
            pg.episode_writes(1.0, 0, 10.0)      # uses fake, not your node-23 code

    A `from module import name` in another hrl module is a second copy of the name, so it is
    swapped too: import helpers either way.
    """
    saved = []
    try:
        for name, fake in fakes.items():
            original = getattr(module, name)
            for m in [module, *_holders(module, name, original)]:
                saved.append((m, name, original))
                setattr(m, name, fake)
        yield
    finally:
        for m, name, original in saved:
            setattr(m, name, original)


def _holders(module, name: str, original) -> list:
    """The other loaded hrl modules that hold `original` under `name`."""
    return [
        m for key, m in list(sys.modules.items())
        if key.startswith("hrl") and m is not None and m is not module
        and getattr(m, name, None) is original
    ]


def table(name: str, answers: dict, key=lambda *a, **k: a):
    """A fake that knows only the answers the test needs, and says so if asked anything else."""
    def fake(*args, **kwargs):
        k = key(*args, **kwargs)
        if k not in answers:
            raise AssertionError(f"fake {name} was called with {k!r}, which this test doesn't "
                                 f"cover. Is your code calling it with the arguments the node expects?")
        return answers[k]
    return fake


def close(got, want, tol, what):
    assert abs(float(got) - float(want)) <= tol, f"{what}: got {float(got):.6f}, expected {float(want):.6f} (tol {tol})"
