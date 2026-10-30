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

The 3v1 option set (nodes 37a-37b), then diagnose.py before phase mu:

    python3 -m hrl.train_hrl options --level academy_3_vs_1_with_keeper --rewards scoring \
        --option-set actions --option-returns --reward-mix both --option-reward-weight 1.0 \
        --timesteps 2000000
    python3 -m hrl.diagnose /work/runs/<options run>
    python3 -m hrl.train_hrl mu ... --option-set actions --option-policy ...

PPO arguments are train_ppo.py's, with the same defaults (Kurach et al. 2019). Use
--rewards scoring: the checkpoint reward is not potential-based (brief, trap 5).

Each survey choice is a flag, so it can be reported and varied:
  --option-set {zones,actions}        which options exist (options.OPTION_SETS; node 37a)
  --reward-mix {option,both,env}      survey 8.3, execution.combine_rewards
  --option-reward-weight              the weight for "both"
  --termination {clock,geometric}     survey Design Principle 2, options.termination
  --with-primitives                   mu over options AND primitives (survey 3.2.1 and 2.2)
  --option-returns                    phase options: each option's return stops where it ends

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

        def __init__(self, max_env_steps: int = 0):
            super().__init__()
            self.env_steps = 0
            self.max_env_steps = max_env_steps

        def _on_step(self) -> bool:
            buf = self.model.rollout_buffer
            taus = [int(info["tau"]) for info in self.locals["infos"]]
            buf.taus[buf.pos] = taus
            self.env_steps += sum(taus)
            self.logger.record("hrl/env_steps", self.env_steps)
            # mu's budget is counted in decisions; this stops it on an env-step budget instead.
            return not (self.max_env_steps and self.env_steps >= self.max_env_steps)

    return SMDPRolloutBuffer, RecordTau


def _option_return_classes():
    """--option-returns: phase `options` with each option's return cut where the option ends.
    Without it, PPO's returns run on past beta into the NEXT options' rewards, so pi is paid for
    setting up later subgoals (node 37b: pi passed forever and stopped scoring)."""
    from stable_baselines3.common.callbacks import BaseCallback
    SMDPRolloutBuffer, _ = _smdp_classes()

    class OptionReturnBuffer(SMDPRolloutBuffer):
        """smdp_gae with every tau = 1 (plain GAE, node 29b), and an option's end treated as
        the end of its return, like an episode's end."""

        def reset(self):
            super().reset()
            self.option_ends = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        def compute_returns_and_advantage(self, last_values, dones):
            dones = np.maximum(dones, self.option_ends[-1])
            saved, self.episode_starts = self.episode_starts, self.episode_starts.copy()
            self.episode_starts[1:] = np.maximum(self.episode_starts[1:], self.option_ends[:-1])
            super().compute_returns_and_advantage(last_values, dones)
            self.episode_starts = saved

    class RecordOptionEnd(BaseCallback):
        """Copies "did an option end on this step" into the buffer slot about to be written."""

        def _on_step(self) -> bool:
            buf = self.model.rollout_buffer
            buf.option_ends[buf.pos] = [info.get("option_end") is not None for info in self.locals["infos"]]
            return True

    return OptionReturnBuffer, RecordOptionEnd


def main() -> int:
    p = build_parser()
    p.add_argument("phase", choices=["options", "mu"])
    p.add_argument("--option-set", default="zones", choices=["zones", "actions", "targeted"])
    p.add_argument("--reward-mix", default="both", choices=["option", "both", "env"])
    p.add_argument("--option-reward-weight", type=float, default=0.1)
    p.add_argument("--termination", default="clock", choices=["clock", "geometric"])
    p.add_argument("--with-primitives", action="store_true")
    p.add_argument("--option-returns", action="store_true",
                   help="phase options: cut each option's return where it ends (node 37b)")
    p.add_argument("--option-policy", help="phase-`options` model zip (required for `mu`)")
    p.add_argument("--deliberation-cost", type=float, default=0.0,
                   help="phase mu: subtracted from mu's reward once per decision (Harb et al. 2018)")
    p.add_argument("--max-env-steps", type=int, default=0,
                   help="phase mu: stop once this many environment steps are used (0 = stop on --timesteps only)")
    p.add_argument("--initiation", action="store_true",
                   help="both phases: options start only where their initiation set I(s) holds (options.Option.initiation)")
    p.add_argument("--end-value-from", default=None,
                   help="phase options: a trained mu zip; an option that ends also earns that mu's V(end state)")
    p.add_argument("--end-value-weight", type=float, default=1.0, help="scales --end-value-from")
    p.add_argument("--style-input", action="store_true",
                   help="phase mu: mu sees a style w (Dirichlet per episode) and earns --style-bonus x w[z] when z is reached")
    p.add_argument("--style-bonus", type=float, default=0.3)
    p.add_argument("--style-alpha", type=float, default=1.0, help="Dirichlet concentration for w (1 = uniform over styles)")
    p.add_argument("--z-from", default=None,
                   help="phase options: a trained options-only mu zip picks each z, instead of uniform "
                        "(joint fine-tune of pi under the mu that uses it; variation 6)")
    args = p.parse_args()
    assert not (args.initiation and args.phase == "mu" and not args.with_primitives), \
        "--initiation in phase mu needs --with-primitives: 3v1 starts on a loose ball, where no option can start"

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
            from hrl.execution import OptionLearningEnv, SMDPEnv, StyleConditionedEnv
            from hrl.options import use_initiation, use_option_set
            use_option_set(args.option_set)              # node 37a: this worker's options
            use_initiation(args.initiation)
            if args.level == "toy":                      # node 37: the known-answer pitch
                from hrl.toy import KnownAnswerPitch
                base = KnownAnswerPitch(x0=0.0)
            else:
                base = GFootballEnv(env_name=args.level, representation=args.representation, rewards=args.rewards)
            rng = np.random.default_rng(args.seed + rank)
            if args.phase == "options":
                end_value, z_policy = None, None
                if args.z_from:
                    mu_z = PPO.load(args.z_from, device="cpu")

                    def z_policy(o):
                        return int(mu_z.predict(o, deterministic=False)[0])
                if args.end_value_from:
                    import torch
                    mu = PPO.load(args.end_value_from, device="cpu")

                    def end_value(o):
                        with torch.no_grad():
                            return float(mu.policy.predict_values(torch.as_tensor(o[None], dtype=torch.float32))[0, 0])
                env = OptionLearningEnv(base, args.reward_mix, args.option_reward_weight, args.termination, rng,
                                        end_value=end_value, end_value_weight=args.end_value_weight,
                                        z_policy=z_policy)
            else:
                pi = PPO.load(args.option_policy, device="cpu")
                env = SMDPEnv(base, lambda o: int(pi.predict(o, deterministic=False)[0]), args.gamma,
                              args.with_primitives, args.termination, rng=rng,
                              deliberation_cost=args.deliberation_cost)
                if args.style_input:
                    env = StyleConditionedEnv(env, args.style_bonus, args.style_alpha, rng=rng)
            env.reset(seed=args.seed + rank)
            return env
        return _init

    def make_team_env(rank):
        """--agents > 1, phase mu: TeamSMDPEnv over the frozen multi-agent option policy."""
        def _init():
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from hrl.options import use_option_set
            use_option_set(args.option_set)
            from hrl.multiagent import MultiAgentGame, TeamSMDPEnv
            pi = PPO.load(args.option_policy, device="cpu")
            env = TeamSMDPEnv(MultiAgentGame(args.level, args.agents), lambda o: pi.predict(o, deterministic=False)[0],
                              args.gamma)
            env.reset(seed=args.seed + rank)
            return env
        return _init

    def team_option_wrap(game, rank):
        from hrl.options import use_option_set
        use_option_set(args.option_set)
        from hrl.multiagent import TeamOptionLearningEnv
        return TeamOptionLearningEnv(game, args.reward_mix, args.option_reward_weight,
                                     np.random.default_rng(args.seed + rank))

    # Phase mu: also log the game's own reward per episode, so goals stay countable when a
    # deliberation cost makes the monitored reward negative.
    if args.agents > 1 and args.phase == "options":
        from hrl.multiagent import make_games
        games = make_games(args.level, args.agents, args.n_envs, args.seed, team_option_wrap)
    else:
        games = SubprocVecEnv([(make_team_env if args.agents > 1 else make_env)(i) for i in range(args.n_envs)],
                              start_method="spawn")
    venv = VecMonitor(games, filename=str(run_dir / "monitor.csv"),
                      info_keywords=("episode_env_reward",) if args.phase == "mu" else ())
    kwargs, callbacks = _ppo_kwargs(args), None
    if args.phase == "options" and args.option_returns:
        OptionReturnBuffer, RecordOptionEnd = _option_return_classes()
        kwargs["rollout_buffer_class"] = OptionReturnBuffer
        callbacks = [RecordOptionEnd()]
    if args.phase == "mu":
        assert args.option_policy, "--option-policy is required for phase mu"
        SMDPRolloutBuffer, RecordTau = _smdp_classes()
        kwargs["rollout_buffer_class"] = SMDPRolloutBuffer
        callbacks = [RecordTau(args.max_env_steps)]

    model = PPO("MlpPolicy", venv, tensorboard_log=str(run_dir / "tb"), **kwargs)
    if args.init_from:
        # Weights only, into the model built above, so this run's buffer class and callbacks stay.
        # A cost from scratch collapses mu before it learns to score (dense cost, sparse goals).
        model.set_parameters(args.init_from, exact_match=True, device="cpu")
    try:
        model.learn(total_timesteps=args.timesteps * (args.agents if args.phase == "options" else 1),
                    callback=callbacks)
    finally:
        model.save(str(run_dir / "final_model"))
        venv.close()
    print(f"[done] {run_dir / 'final_model.zip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
