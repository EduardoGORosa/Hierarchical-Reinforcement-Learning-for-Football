"""PPO baseline on Google Research Football (Stable-Baselines3).

Hyperparameter defaults follow the tuned PPO configuration published with the
GRF paper (Kurach et al., 2019) in `gfootball/examples/repro_checkpoint_easy.sh`,
translated from OpenAI Baselines' PPO2 to SB3:

    nsteps        -> n_steps            nminibatches -> batch_size
    noptepochs    -> n_epochs           cliprange    -> clip_range
    lam (0.95)    -> gae_lambda         vf_coef      -> vf_coef (0.5)

Note that PPO is one of three baselines benchmarked in that paper (alongside
IMPALA and Ape-X DQN), not "the" standard GRF baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    env = p.add_argument_group("environment")
    env.add_argument("--level", default="academy_empty_goal_close",
                     help="GRF scenario name (e.g. academy_empty_goal_close, 11_vs_11_easy_stochastic)")
    env.add_argument("--representation", default="simple115v2",
                     choices=["simple115v2", "simple115", "extracted", "pixels", "pixels_gray"],
                     help="observation representation")
    env.add_argument("--stacked", action="store_true", help="stack 4 frames (used with 'extracted')")
    env.add_argument("--rewards", default="scoring,checkpoints",
                     help="reward function: 'scoring' or 'scoring,checkpoints'")
    env.add_argument("--n-envs", type=int, default=8, help="parallel environment processes")

    ppo = p.add_argument_group("PPO (defaults from Kurach et al. 2019)")
    ppo.add_argument("--timesteps", type=int, default=500_000, help="total environment steps")
    ppo.add_argument("--n-steps", type=int, default=512, help="rollout length per env")
    ppo.add_argument("--n-minibatches", type=int, default=8, help="minibatches per update")
    ppo.add_argument("--n-epochs", type=int, default=2, help="optimisation epochs per update")
    ppo.add_argument("--lr", type=float, default=0.000343)
    ppo.add_argument("--gamma", type=float, default=0.993)
    ppo.add_argument("--gae-lambda", type=float, default=0.95)
    ppo.add_argument("--clip-range", type=float, default=0.08)
    ppo.add_argument("--ent-coef", type=float, default=0.003)
    ppo.add_argument("--vf-coef", type=float, default=0.5)
    ppo.add_argument("--max-grad-norm", type=float, default=0.64)

    run = p.add_argument_group("run")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--logdir", default="/work/runs")
    run.add_argument("--run-name", default=None, help="defaults to <level>_seed<seed>_<timestamp>")
    run.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    run.add_argument("--checkpoint-freq", type=int, default=0,
                     help="save every N steps (0 = only at the end)")
    return p


def main() -> int:
    args = build_parser().parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.utils import set_random_seed

    from gfootball_env import GFootballEnv

    run_name = args.run_name or f"{args.level}_seed{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.logdir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Recorded so a run is reproducible and so watch.py knows which scenario and
    # representation a checkpoint belongs to without being told again.
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    # Pixel/SMM observations go through a CNN and want the GPU; simple115 is a
    # small MLP where the per-step transfer costs more than the matmuls save.
    is_visual = args.representation in ("extracted", "pixels", "pixels_gray")
    device = args.device
    if device == "auto":
        device = "cuda" if (is_visual and torch.cuda.is_available()) else "cpu"

    policy = "CnnPolicy" if is_visual else "MlpPolicy"

    def make_env(rank: int):
        def _init():
            e = GFootballEnv(
                env_name=args.level,
                representation=args.representation,
                rewards=args.rewards,
                stacked=args.stacked,
                logdir=str(run_dir / "dumps"),
            )
            e.reset(seed=args.seed + rank)
            return e
        return _init

    set_random_seed(args.seed)

    # 'spawn' rather than 'fork': each worker loads the gfootball C++/SDL engine,
    # and forking a parent that has already initialised it is not safe.
    venv = SubprocVecEnv([make_env(i) for i in range(args.n_envs)], start_method="spawn")
    venv = VecMonitor(venv, filename=str(run_dir / "monitor.csv"))

    rollout = args.n_envs * args.n_steps
    if rollout % args.n_minibatches != 0:
        print(f"[warn] rollout {rollout} not divisible by {args.n_minibatches} minibatches", file=sys.stderr)
    batch_size = max(1, rollout // args.n_minibatches)

    print(f"[config] level={args.level} repr={args.representation} policy={policy} device={device}")
    print(f"[config] n_envs={args.n_envs} n_steps={args.n_steps} rollout={rollout} batch_size={batch_size}")
    print(f"[config] logging to {run_dir}")

    model = PPO(
        policy,
        venv,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=device,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
    )

    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // args.n_envs),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
        ))

    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks or None, progress_bar=False)
    finally:
        model.save(str(run_dir / "final_model"))
        venv.close()

    print(f"[done] model saved to {run_dir / 'final_model.zip'}")
    print(f"[done] tensorboard --logdir {run_dir / 'tb'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
