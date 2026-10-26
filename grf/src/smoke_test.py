"""Sanity checks for the GRF environment, run before any training.

Verifies that the C++ engine loads, that the Gymnasium wrapper reports coherent
spaces, that a random rollout produces sane transitions, and that the SB3 env
checker is satisfied. Exits non-zero on the first failure.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

LEVEL = "academy_empty_goal_close"


def check(label: str, fn):
    print(f"  {label} ... ", end="", flush=True)
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 - a smoke test wants the raw failure
        print(f"FAIL\n    {type(exc).__name__}: {exc}")
        raise
    print("ok" if not isinstance(result, str) else f"ok ({result})")
    return result


def main() -> int:
    print("[1/5] libraries")

    def _versions():
        import gym, gymnasium, numpy, stable_baselines3, torch
        return (f"gym={gym.__version__} gymnasium={gymnasium.__version__} "
                f"numpy={numpy.__version__} sb3={stable_baselines3.__version__} "
                f"torch={torch.__version__}")

    check("versions", _versions)

    def _cuda():
        import torch
        if not torch.cuda.is_available():
            return "no CUDA (CPU only)"
        name = torch.cuda.get_device_name(0)
        cap = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
        # A Blackwell card (sm_120) on a torch built only up to sm_90 imports
        # fine and then fails at the first kernel launch, so actually run one.
        torch.zeros(8, device="cuda").add_(1).sum().item()
        return f"{name} sm_{cap.replace('.', '')}, kernel launch ok"

    check("torch/cuda", _cuda)

    print("[2/5] engine + wrapper")
    from gfootball_env import GFootballEnv, NUM_ACTIONS

    # One engine instance for the whole test: gfootball keeps process-global
    # state, so two live environments in one process is not a supported setup
    # (which is also why training uses SubprocVecEnv rather than DummyVecEnv).
    env = check("create env", lambda: GFootballEnv(env_name=LEVEL))

    print("[3/5] spaces")
    obs_space, act_space = env.observation_space, env.action_space
    check("observation_space", lambda: f"{obs_space}")
    check("action_space", lambda: f"{act_space}")
    assert act_space.n == NUM_ACTIONS, f"expected {NUM_ACTIONS} actions, got {act_space.n}"

    print("[4/5] random rollout")

    def _rollout():
        obs, info = env.reset(seed=0)
        assert obs_space.contains(obs), f"reset obs outside space; shape={obs.shape} dtype={obs.dtype}"
        steps = 0
        episodes = 0
        total_reward = 0.0
        t0 = time.time()
        for _ in range(3000):
            obs, reward, terminated, truncated, info = env.step(act_space.sample())
            assert obs_space.contains(obs), "step obs outside declared space"
            total_reward += reward
            steps += 1
            if terminated or truncated:
                episodes += 1
                env.reset()
        dt = time.time() - t0
        return (f"{steps} steps, {episodes} episodes, reward sum {total_reward:+.1f}, "
                f"{steps / dt:.0f} steps/s single-process")

    check("3000 random steps", _rollout)

    print("[5/5] stable-baselines3 env checker")

    def _sb3_check():
        from stable_baselines3.common.env_checker import check_env
        check_env(env, warn=True, skip_render_check=True)

    check("check_env", _sb3_check)
    env.close()

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
