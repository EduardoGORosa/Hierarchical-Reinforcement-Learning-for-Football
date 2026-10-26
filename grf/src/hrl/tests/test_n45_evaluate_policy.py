"""Node 45 · win rate over a fixed list of game seeds. Your code: evaluate.evaluate_policy."""

from hrl import evaluate as ev
from hrl.tests._support import needs

NODE, TITLE, LAYER, NEEDS = 45, "evaluate_policy", 7, ()


class ScriptedEnv:
    """A fake env: even seeds are won on step 2, odd seeds lost on step 3. Records closing."""
    closed = 0

    def __init__(self, seed):
        self.win, self.t = seed % 2 == 0, 0

    def reset(self, seed=None, options=None):
        return [0.0], {}

    def step(self, action):
        self.t += 1
        done = self.t == (2 if self.win else 3)
        return [0.0], float(done and self.win), done, False, {}

    def close(self):
        ScriptedEnv.closed += 1


def test_unit_scripted():
    ScriptedEnv.closed = 0
    assert ev.evaluate_policy(lambda obs: 0, ScriptedEnv, seeds=range(10)) == 0.5
    assert ScriptedEnv.closed == 10, "close every env"


def test_stack():
    needs("gymnasium")
    from hrl.toy import KnownAnswerPitch
    from hrl import options as O
    # toy: seed s starts the ball at x = s/100; always shooting scores iff x > 0.69, so seeds
    # 70..99 are wins: exactly 30 of 100
    make_env = lambda seed: _Seeded(KnownAnswerPitch(x0=None), seed)
    assert ev.evaluate_policy(lambda obs: O.SHOT_ACTION, make_env, seeds=range(100)) == 0.30


class _Seeded:
    """Passes the game seed to reset, as GRF's make_eval_env does through its config."""

    def __init__(self, env, seed):
        self.env, self.seed = env, seed

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=self.seed)

    def step(self, a):
        return self.env.step(a)

    def close(self):
        pass
