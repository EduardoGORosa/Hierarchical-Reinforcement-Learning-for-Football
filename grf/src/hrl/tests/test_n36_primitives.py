"""Node 36 · mu over options AND primitives (survey 3.2.1). Your code: execution.action_meaning."""

from hrl.tests._support import needs

NODE, TITLE, LAYER, NEEDS = 36, "action_meaning (options + primitives)", 4, (32,)


def test_unit_meaning():
    needs("gymnasium")
    from hrl.execution import action_meaning as m, N_OPTIONS
    assert m(0, True) == ("option", 0) and m(N_OPTIONS - 1, True) == ("option", N_OPTIONS - 1)
    assert m(N_OPTIONS, True) == ("primitive", 0) and m(N_OPTIONS + 18, True) == ("primitive", 18)
    for bad, prim in [(N_OPTIONS + 19, True), (N_OPTIONS, False), (-1, True)]:
        try:
            m(bad, prim)
        except ValueError:
            continue
        raise AssertionError(f"action_meaning({bad}, {prim}) should raise ValueError")


def test_stack():
    needs("gymnasium")
    from hrl import execution as E
    from hrl.toy import KnownAnswerPitch, RIGHT
    env = E.SMDPEnv(KnownAnswerPitch(), lambda o: RIGHT, 0.9, with_primitives=True)
    assert env.action_space.n == E.N_OPTIONS + E.N_PRIMITIVES
    env.reset()
    obs, _, _, _, info = env.step(E.N_OPTIONS + RIGHT)
    assert info["tau"] == 1 and abs(obs[0] - 0.02) < 1e-6, (info, obs)
