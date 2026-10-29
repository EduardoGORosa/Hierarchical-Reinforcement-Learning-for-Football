"""Node 37c · a deliberation cost on mu (Harb et al. 2018). Written by Claude: execution.SMDPEnv(deliberation_cost=...)."""

from hrl.tests._support import needs

NODE, TITLE, LAYER, NEEDS = "37c", "deliberation cost (once per decision)", 5, (36,)


def _pair(cost):
    from hrl import execution as E
    from hrl.toy import KnownAnswerPitch, RIGHT
    return E.SMDPEnv(KnownAnswerPitch(), lambda o: RIGHT, 0.9, with_primitives=True, deliberation_cost=cost), E, RIGHT


def test_unit_once_per_decision():
    needs("gymnasium")
    free, E, RIGHT = _pair(0.0)
    paid, _, _ = _pair(0.05)
    for env in (free, paid):
        env.reset()
    # an option: many env steps, charged once
    _, r0, _, _, i0 = free.step(0)
    _, r1, _, _, i1 = paid.step(0)
    assert i0["tau"] == i1["tau"] > 1, (i0, i1)
    assert abs((r0 - r1) - 0.05) < 1e-9, f"an option of {i1['tau']} steps must pay the cost once: {r0} vs {r1}"
    # a primitive: one env step, charged once too
    _, p0, _, _, _ = free.step(E.N_OPTIONS + RIGHT)
    _, p1, _, _, j1 = paid.step(E.N_OPTIONS + RIGHT)
    assert abs((p0 - p1) - 0.05) < 1e-9, (p0, p1)
    assert j1["env_reward"] == p0, "info['env_reward'] is the game's reward, before the cost"


def test_unit_episode_env_reward_resets():
    needs("gymnasium")
    env, E, RIGHT = _pair(0.05)
    env.reset()
    _, _, _, _, info = env.step(E.N_OPTIONS + RIGHT)
    assert "episode_env_reward" in info
    env.reset()
    assert env._episode_env_reward == 0.0


def test_stack():
    needs("gymnasium")
    env, E, RIGHT = _pair(0.0)
    env.reset()
    _, r, _, _, info = env.step(E.N_OPTIONS + RIGHT)
    assert info["tau"] == 1 and r == info["env_reward"], "cost 0 is the plain SMDP of nodes 32-36"
