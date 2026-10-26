"""Node 35 · interruption (survey 7.1.2). Your code: execution.interrupt_by_advantage, and the
interruption lines inside SMDPEnv.step's loop."""

from hrl.tests._support import needs

NODE, TITLE, LAYER, NEEDS = 35, "interruption", 4, (32,)


def test_unit_criterion():
    needs("gymnasium")
    from hrl.execution import interrupt_by_advantage as stop
    assert stop(0.2, 0.5) and not stop(0.5, 0.5) and not stop(0.7, 0.5)


def test_stack():
    needs("gymnasium")
    from hrl import execution as E, options as O
    from hrl.toy import KnownAnswerPitch, RIGHT
    env = E.SMDPEnv(KnownAnswerPitch(), lambda o: RIGHT, 0.9, should_interrupt=lambda o, z: True, interrupt_every=2)
    env.reset()
    _, _, _, _, info = env.step(O.SUBGOALS.index("centre"))
    assert (info["tau"], info["option_end"]) == (2, "interrupted"), info
    # With a made-up value table, via your criterion: continuing is worth less after x > 0.05
    q = lambda obs, z: 0.1 if obs[0] > 0.05 else 0.9
    env = E.SMDPEnv(KnownAnswerPitch(), lambda o: RIGHT, 0.9,
                    should_interrupt=lambda obs, z: E.interrupt_by_advantage(q(obs, z), 0.5))
    env.reset()
    _, _, _, _, info = env.step(O.SUBGOALS.index("centre"))
    assert (info["tau"], info["option_end"]) == (3, "interrupted"), info
