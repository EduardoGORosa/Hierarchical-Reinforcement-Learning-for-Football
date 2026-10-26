"""Which players do N learning agents control? Run with grf/scenarios/ma_*.py
bind-mounted into gfootball/scenarios/."""
import collections, numpy as np, gfootball.env as fe
for level, m in [("academy_3_vs_1_with_keeper", 3), ("academy_pass_and_shoot_with_keeper", 2), ("ma_3_vs_1_with_keeper", 3), ("ma_pass_and_shoot_with_keeper", 2)]:
    env = fe.create_environment(env_name=level, representation="raw", rewards="scoring",
                                number_of_left_players_agent_controls=m)
    rng = np.random.default_rng(0)
    switches, gk_ctrl, lens, assign = 0, 0, [], collections.Counter()
    for ep in range(30):
        obs = env.reset(); prev = tuple(o["active"] for o in obs); assign[prev] += 1
        done, L = False, 0
        while not done:
            obs, r, done, _ = env.step([int(rng.integers(19)) for _ in range(m)])
            cur = tuple(o["active"] for o in obs); L += 1
            switches += cur != prev; gk_ctrl += 0 in cur; prev = cur
        lens.append(L)
    print(level, "agents", m, "obs per agent", len(obs), "| initial assignment", dict(assign),
          "| steps with any reassignment", switches, "of", sum(lens), "| steps where GK (idx0) controlled", gk_ctrl,
          "| reward shape", np.shape(r), "| designated==active?", all(o["designated"]==o["active"] for o in obs))
    env.close()
