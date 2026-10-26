"""Do two processes with the same actions replay the same episodes?

Usage: python3 check_determinism.py default|seeded   (run twice, compare hashes)
Default GRF config: no. With game_engine_random_seed fixed: yes, bit-for-bit.
"""
import hashlib, json, sys, numpy as np, gfootball.env as fe
mode = sys.argv[1]
opts = {} if mode == "default" else {"game_engine_random_seed": 123}
env = fe.create_environment(env_name="academy_3_vs_1_with_keeper", representation="raw",
                            rewards="scoring", other_config_options=opts)
rng = np.random.default_rng(0)
out = []
for ep in range(3):
    o = env.reset(); h = hashlib.sha1(); done = False; L = 0; R = 0
    while not done:
        o, r, done, _ = env.step(int(rng.integers(19))); L += 1; R += r
        x = o[0]; h.update(np.round(np.concatenate([np.ravel(x["ball"]), np.ravel(x["left_team"]), np.ravel(x["right_team"])]), 6).tobytes())
    out.append(f"ep{ep}:{L}steps:R{R:+.0f}:{h.hexdigest()[:10]}")
print(mode, " ".join(out))
