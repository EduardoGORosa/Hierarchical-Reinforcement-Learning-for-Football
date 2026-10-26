"""Node 34 · the data option policies learn from: z as an input, mu uniform, rewards mixed.
Your code: execution.OptionLearningEnv.step."""

import numpy as np

from hrl.tests._support import needs, swap

NODE, TITLE, LAYER, NEEDS = 34, "OptionLearningEnv.step", 4, (31, 33)


def test_unit_step():
    needs("gymnasium")
    from hrl import execution as E, options as O
    from hrl.toy import KnownAnswerPitch, RIGHT
    every4 = lambda start, prev, o, step, done, action=None, mode="clock", rng=None: "reached" if step - start.step >= 4 else None
    with swap(O, termination=every4, option_reward=lambda s, p, o, step, action=None: 1.0 if step % 4 == 0 else 0.0), \
            swap(E, combine_rewards=lambda r_env, r_o, mix, weight=0.1: 1000 * r_o + r_env):
        env = E.OptionLearningEnv(KnownAnswerPitch(), rng=np.random.default_rng(3))
        obs, _ = env.reset()
        zs, rewards = [int(np.argmax(obs[2:]))], []
        for _ in range(12):
            obs, r, *_ = env.step(RIGHT)
            rewards.append(r)
            zs.append(int(np.argmax(obs[2:])))
    assert rewards == [0, 0, 0, 1000] * 3, f"rewards through your combine_rewards call: {rewards}"
    assert [n for _, _, n in env.log] == [4, 4, 4] and all(w == "reached" for _, w, _ in env.log), env.log
    assert len(env.log) == 3 and env.log[0][0] == O.SUBGOALS[zs[0]], "the log names the option that ran"


def test_stack():
    needs("gymnasium")
    from hrl import execution as E, options as O
    from hrl.toy import KnownAnswerPitch, RIGHT
    env = E.OptionLearningEnv(KnownAnswerPitch(), mix="option", rng=np.random.default_rng(0))
    obs, _ = env.reset()
    assert obs.shape == (2 + E.N_OPTIONS,) and obs[2:].sum() == 1.0, "obs = base + one-hot z"
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, _ = env.step(RIGHT)
        total, done = total + r, term or trunc
    reached = sum(w == "reached" for _, w, _ in env.log)
    assert reached > 0, "running right for 100 steps must reach some subgoal"
    assert total == reached * O.OPTION_REWARD, "mix='option': the reward is exactly the option rewards"
