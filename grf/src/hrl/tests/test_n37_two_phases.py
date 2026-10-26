"""Node 37 · the two phases, on the toy pitch. Provided code: train_hrl.py. You read it; the tests
check the SB3 glue against your smdp_gae and run both phases for real."""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from hrl.tests._support import needs, swap

NODE, TITLE, LAYER, NEEDS = 37, "two phases (train_hrl.py, provided)", 5, (29, 32, 34)


def _buffer():
    import gymnasium
    import torch
    from hrl.train_hrl import _smdp_classes
    Buf, _ = _smdp_classes()
    b = Buf(4, gymnasium.spaces.Box(-1, 1, (2,)), gymnasium.spaces.Discrete(6), "cpu", gamma=0.9, gae_lambda=1.0, n_envs=1)
    for t, (r, tau, v, start) in enumerate([(0.0, 2, 0.5, 1), (1.0, 3, 0.8, 0), (0.0, 1, 0.1, 1), (0.5, 4, 0.2, 0)]):
        b.taus[b.pos] = tau
        b.add(np.zeros((1, 2)), np.array([[0]]), np.array([r]), np.array([start], dtype=bool),
              torch.tensor([v]), torch.tensor([0.0]))
    return b, torch


def test_unit_buffer_glue():
    needs("stable_baselines3")
    b, torch = _buffer()
    calls = []
    from hrl import smdp
    with swap(smdp, smdp_gae=lambda r, taus, v, lv, d, g, lam: (calls.append((list(taus), list(d), lv)), np.zeros(len(r)))[1]):
        b.compute_returns_and_advantage(torch.tensor([0.3]), np.array([False]))
    taus, ended, lv = calls[0]
    assert taus == [2, 3, 1, 4], f"taus reach smdp_gae: {taus}"
    # a new episode STARTS at decision 2, so one ENDED after decision 1
    assert ended == [0.0, 1.0, 0.0, 0.0], f"'episode ended after decision t' from SB3's episode_starts: {ended}"
    assert abs(lv - 0.3) < 1e-6


def test_stack():
    needs("stable_baselines3")
    b, torch = _buffer()
    b.compute_returns_and_advantage(torch.tensor([0.3]), np.array([False]))
    from hrl import smdp
    want = smdp.smdp_gae([0.0, 1.0, 0.0, 0.5], [2, 3, 1, 4], [0.5, 0.8, 0.1, 0.2], 0.3, [0, 1, 0, 0], 0.9, 1.0)
    assert np.allclose(b.advantages[:, 0], want), "SMDPRolloutBuffer uses your smdp_gae"
    # both phases, for real, on the toy pitch
    src = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as d:
        base = [sys.executable, "-m", "hrl.train_hrl", "--level", "toy", "--n-envs", "2", "--logdir", d]
        subprocess.run(base + ["options", "--timesteps", "1024", "--n-steps", "256", "--run-name", "o"],
                       cwd=src, check=True, capture_output=True)
        out = subprocess.run(base + ["mu", "--timesteps", "128", "--n-steps", "32", "--run-name", "m",
                                     "--option-policy", f"{d}/o/final_model.zip"],
                             cwd=src, check=True, capture_output=True, text=True).stdout
        assert (Path(d) / "m" / "final_model.zip").exists()
        steps = [int(l.split("|")[2]) for l in out.splitlines() if "env_steps" in l]
        assert steps and steps[-1] > 128, f"env steps ({steps}) must exceed the 128 decisions"
