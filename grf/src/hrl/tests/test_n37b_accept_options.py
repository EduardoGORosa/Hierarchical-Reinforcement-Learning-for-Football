"""Node 37b · train the "actions" option set and accept it with the diagnostic. Your code:
diagnose.accepted. Provided: diagnose.py, train_hrl.py --option-set."""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from hrl.tests._support import needs

NODE, TITLE, LAYER, NEEDS = "37b", "accept the options (diagnose.py)", 5, ("37a", 37)


def report(reached=(0.3, 0.3, 0.3), tv=0.4):
    rows = [dict(name=n, reached=r, failed=0.5, timeout=0.5 - r, reached_per_episode=r, goal_rate=0.3)
            for n, r in zip(("pass", "dribble", "shoot"), reached)]
    return dict(option_set="actions", episodes=10, options=rows, argmax_change=0.1, max_tv=tv,
                n_states=100)


def test_unit_accepted():
    from hrl import diagnose as D
    lo, tv = D.MIN_REACHED, D.MIN_MAX_TV
    assert D.accepted(report()), "every option reaches its subgoal and pi reads z"
    assert D.accepted(report((lo, lo, lo), tv)), "exactly at both thresholds counts"
    assert not D.accepted(report((0.3, 0.0, 0.3))), "dribble never reached"
    assert not D.accepted(report(tv=0.05)), "pi ignores z (the collapsed zone run: TV 0.05)"


def test_stack():
    needs("stable_baselines3")
    from hrl import diagnose as D, execution as E, options as O
    from hrl.toy import KnownAnswerPitch
    pitch = KnownAnswerPitch()
    pitch.reset()
    toy = pitch.raw_observation()
    O.completed_pass(toy, toy), O.beat_defender(toy, toy)  # TODO until node 37a is written
    O.use_option_set("actions")
    try:   # the option count flows into both envs
        assert E.OptionLearningEnv(KnownAnswerPitch()).observation_space.shape == (2 + 3,)
        assert E.SMDPEnv(KnownAnswerPitch(), lambda o: 0, 0.9).action_space.n == 3
    finally:
        O.use_option_set("zones")
    # --option-returns: a return stops where its option ends (row 1), as at an episode's end
    import gymnasium
    import torch
    from hrl.train_hrl import _option_return_classes
    Buf, _ = _option_return_classes()
    b = Buf(4, gymnasium.spaces.Box(-1, 1, (2,)), gymnasium.spaces.Discrete(3), "cpu", gamma=0.9,
            gae_lambda=1.0, n_envs=1)
    for r, option_end, start in [(0.0, 0, 1), (1.0, 1, 0), (0.0, 0, 0), (0.5, 0, 0)]:
        b.option_ends[b.pos] = option_end
        b.add(np.zeros((1, 2)), np.array([[0]]), np.array([r]), np.array([start], dtype=bool),
              torch.tensor([0.5]), torch.tensor([0.0]))
    b.compute_returns_and_advantage(torch.tensor([0.3]), np.array([False]))
    assert np.allclose(b.returns[:, 0], [0.9, 1.0, 0.693, 0.77]), b.returns[:, 0]
    src = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as d:
        cmd = ["options", "--level", "toy", "--n-envs", "2", "--logdir", d, "--run-name", "o",
               "--option-set", "actions", "--option-returns", "--timesteps", "1024", "--n-steps", "256"]
        run = subprocess.run([sys.executable, "-m", "hrl.train_hrl", *cmd], cwd=src, capture_output=True, text=True)
        assert run.returncode == 0, "train_hrl --option-set actions crashed:\n" + run.stderr[-1500:]
        rep = D.diagnose(Path(d) / "o", episodes=3)
    names = [r["name"] for r in rep["options"]]
    assert names == ["pass", "dribble", "shoot"], names
    for r in rep["options"]:
        assert np.isclose(r["reached"] + r["failed"] + r["timeout"], 1.0), r
    # The toy has one player of ours and only their keeper: pass and dribble can't happen.
    assert rep["options"][0]["reached"] == 0 and rep["options"][1]["reached"] == 0, rep["options"]
    assert not D.accepted(rep), "an option that is never reached can't be accepted"
