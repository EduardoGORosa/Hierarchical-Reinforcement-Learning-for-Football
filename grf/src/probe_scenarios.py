"""Characterise one GRF scenario: random vs scripted policy, horizon, effective obs size."""
import json, math, sys, time
import numpy as np
import gfootball.env as football_env
from gfootball.env.wrappers import Simple115StateWrapper

level, n_eps, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
env = football_env.create_environment(env_name=level, representation="raw", rewards="scoring",
                                      number_of_left_players_agent_controls=1)
DIRS = {0: 5, -1: 4, -2: 3, -3: 2, 4: 1, -4: 1, 3: 8, 2: 7, 1: 6}

def move_to(p, t):
    k = round(math.atan2(t[1] - p[1], t[0] - p[0]) / (math.pi / 4))
    return DIRS[k]

def scripted(o):
    me = o["left_team"][o["active"]]
    if o["ball_owned_team"] == 0 and o["ball_owned_player"] == o["active"]:
        return 12 if me[0] > 0.7 and abs(me[1]) < 0.25 else move_to(me, (1.0, 0.0))
    return move_to(me, o["ball"][:2])

res = {"level": level}
for name in ("random", "scripted"):
    rng = np.random.default_rng(0)
    goals, lens, conceded, obs115, steps, t0 = 0, [], 0, [], 0, time.time()
    for ep in range(n_eps):
        o = env.reset(); done = False; L = 0; R = 0.0
        while not done:
            obs115.append(Simple115StateWrapper.convert_observation(o, True)[0])
            a = int(rng.integers(19)) if name == "random" else scripted(o[0])
            o, r, done, _ = env.step(a); L += 1; R += r
        lens.append(L); steps += L
        goals += R > 0; conceded += R < 0
    X = np.asarray(obs115)
    res[name] = dict(episodes=n_eps, success_rate=goals / n_eps, conceded_rate=conceded / n_eps,
                     mean_len=float(np.mean(lens)), median_len=float(np.median(lens)),
                     max_len=int(np.max(lens)), steps_per_s=steps / (time.time() - t0),
                     varying_dims=int((X.std(0) > 1e-6).sum()),
                     padded_minus1_dims=int(((X == -1).all(0)).sum()))
o = env.reset()[0]
res["left_players"] = len(o["left_team"]); res["right_players"] = len(o["right_team"])
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res))
