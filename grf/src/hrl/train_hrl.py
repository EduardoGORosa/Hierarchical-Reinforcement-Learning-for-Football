"""Train the hierarchical agent as the survey's Algorithm 1, in two separate phases.

Node 37 (layer 5 of the build map). Provided code: node 37 reads it line by line. Its tests
check SMDPRolloutBuffer against your smdp.smdp_gae, and run both phases on the toy pitch:

    ./run.sh hrl check 37
    python3 -m hrl.train_hrl options --level toy --timesteps 20000 --n-envs 2   # seconds

Klissarov et al. 2025 (arXiv:2506.14045), Algorithm 1 and section 8.1:

    OptionSpecification   hrl/options.py, fixed by hand (not run here)
    OptionLearning        phase `options`: learn pi(a | s, z) for all six options, while mu
                          picks options uniformly at random
    PolicyImprovement     phase `mu`: freeze the option policy, learn mu(o | s) over whole
                          options (SMDP, gamma^tau)

    ./run.sh shell
    cd /work/src
    python3 -m hrl.train_hrl options --level academy_3_vs_1_with_keeper --rewards scoring \
        --timesteps 2000000 --reward-mix both
    python3 -m hrl.train_hrl mu --level academy_3_vs_1_with_keeper --rewards scoring \
        --timesteps 1000000 --option-policy /work/runs/<options run>/final_model.zip

PPO arguments are train_ppo.py's, with the same defaults (Kurach et al. 2019). Use
--rewards scoring: the checkpoint reward is not potential-based (brief, trap 5).

Each survey choice is a flag, so it can be reported and varied:
  --reward-mix {option,both,env}      survey 8.3, execution.combine_rewards
  --option-reward-weight              the weight for "both"
  --termination {clock,geometric}     survey Design Principle 2, options.termination
  --with-primitives                   mu over options AND primitives (survey 3.2.1 and 2.2)

Budget accounting (node 11): in phase `mu`, model.num_timesteps counts DECISIONS. The env
steps are the sum of tau, logged as hrl/env_steps. Learning curves must use env steps. A tiny
test run took 6,599 env steps for 256 decisions.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # grf/src

from train_ppo import build_parser   # noqa: E402

from hrl import smdp                  # noqa: E402


def _ppo_kwargs(args):
    rollout = args.n_envs * args.n_steps
    return dict(
        learning_rate=args.lr, n_steps=args.n_steps, batch_size=max(1, rollout // args.n_minibatches),
        n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_range=args.clip_range, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm, seed=args.seed, device="cpu", verbose=1,
    )


def _smdp_classes():
    """SB3 subclasses, built lazily so importing this module needs no SB3."""
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.callbacks import BaseCallback

    class SMDPRolloutBuffer(RolloutBuffer):
        """Stores each decision's duration tau and discounts by gamma^tau (smdp.smdp_gae).
        SB3's own buffer uses one scalar gamma for every step: right for the option policies,
        wrong for mu. Same flaw outside the buffer: on truncated=True, collect_rollouts adds
        gamma * V(final obs) to the reward. Safe only while GFootballEnv never truncates."""

        def reset(self):
            super().reset()
            self.taus = np.ones((self.buffer_size, self.n_envs), dtype=np.int64)

        def compute_returns_and_advantage(self, last_values, dones):
            last_values = last_values.clone().cpu().numpy().flatten()
            # SB3 marks an episode boundary as episode_starts[t + 1]; smdp_gae wants
            # dones[t] = "the episode ended after decision t".
            ended = np.zeros_like(self.rewards)
            ended[:-1] = self.episode_starts[1:]
            ended[-1] = dones.astype(np.float32)
            for e in range(self.n_envs):
                self.advantages[:, e] = smdp.smdp_gae(
                    self.rewards[:, e], self.taus[:, e], self.values[:, e],
                    last_values[e], ended[:, e], self.gamma, self.gae_lambda)
            self.returns = self.advantages + self.values

    class RecordTau(BaseCallback):
        """Copies info["tau"] into the buffer slot about to be written.
        (SB3 2.4 calls on_step() just before rollout_buffer.add().)"""

        def __init__(self):
            super().__init__()
            self.env_steps = 0

        def _on_step(self) -> bool:
            buf = self.model.rollout_buffer
            taus = [int(info["tau"]) for info in self.locals["infos"]]
            buf.taus[buf.pos] = taus
            self.env_steps += sum(taus)
            self.logger.record("hrl/env_steps", self.env_steps)
            return True

    return SMDPRolloutBuffer, RecordTau


def main() -> int:
    p = build_parser()
    p.add_argument("phase", choices=["options", "mu"])
    p.add_argument("--reward-mix", default="both", choices=["option", "both", "env"])
    p.add_argument("--option-reward-weight", type=float, default=0.1)
    p.add_argument("--termination", default="clock", choices=["clock", "geometric"])
    p.add_argument("--with-primitives", action="store_true")
    p.add_argument("--option-policy", help="phase-`options` model zip (required for `mu`)")
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    run_dir = Path(args.logdir) / (args.run_name or
                                   f"hrl_{args.phase}_{args.level}_seed{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    def make_env(rank):
        def _init():
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from gfootball_env import GFootballEnv
            from hrl.execution import OptionLearningEnv, SMDPEnv
            if args.level == "toy":                      # node 37: the known-answer pitch
                from hrl.toy import KnownAnswerPitch
                base = KnownAnswerPitch(x0=0.0)
            else:
                base = GFootballEnv(env_name=args.level, representation=args.representation, rewards=args.rewards)
            rng = np.random.default_rng(args.seed + rank)
            if args.phase == "options":
                env = OptionLearningEnv(base, args.reward_mix, args.option_reward_weight, args.termination, rng)
            else:
                pi = PPO.load(args.option_policy, device="cpu")
                env = SMDPEnv(base, lambda o: int(pi.predict(o, deterministic=False)[0]), args.gamma,
                              args.with_primitives, args.termination, rng=rng)
            env.reset(seed=args.seed + rank)
            return env
        return _init

    venv = VecMonitor(SubprocVecEnv([make_env(i) for i in range(args.n_envs)], start_method="spawn"),
                      filename=str(run_dir / "monitor.csv"))
    kwargs, callbacks = _ppo_kwargs(args), None
    if args.phase == "mu":
        assert args.option_policy, "--option-policy is required for phase mu"
        SMDPRolloutBuffer, RecordTau = _smdp_classes()
        kwargs["rollout_buffer_class"] = SMDPRolloutBuffer
        callbacks = [RecordTau()]

    model = PPO("MlpPolicy", venv, tensorboard_log=str(run_dir / "tb"), **kwargs)
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
    finally:
        model.save(str(run_dir / "final_model"))
        venv.close()
    print(f"[done] {run_dir / 'final_model.zip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
