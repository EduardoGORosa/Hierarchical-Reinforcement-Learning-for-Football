#!/usr/bin/env bash
# Multi-agent hierarchy, two designs against the flat three-agent team (2026-10-30). Seed 0, 6M game
# steps per arm. Written for the 8-vCPU cloud runner; shuts the machine down when finished
# (SHUTDOWN=0 to keep it up). Two lanes, 8 games each, so PPO settings match the earlier runs.
#
#   lane 1  A: team plays with roles (options 2M, team mu 4M), then flat team 3M -> 6M (continued)
#   lane 2  B: per-agent mu (options 2M, per-agent mu 4M)
#   then    every final agent evaluated on eval seeds 1-4, 600 games each
set -uo pipefail
GRF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$GRF"
OUT=$GRF/runs/ma_hrl; mkdir -p "$OUT/logs"
stamp() { echo "$(date '+%F %T') $*" >> "$OUT/progress.txt"; }
run() { local n=$1; shift; stamp "start $n"; "$@" > "$OUT/logs/$n.log" 2>&1 && stamp "done  $n" || stamp "FAIL  $n"; }
MA=(--level ma_3_vs_1_with_keeper --rewards scoring --agents 3 --seed 0)
OPT=(--option-returns --reward-mix both --option-reward-weight 1.0)

lane1() {
  run maA_opt_s0 ./run.sh hrl train_hrl options "${MA[@]}" --ma-design roles --option-set plays "${OPT[@]}" \
    --timesteps 2000000 --run-name maA_opt_s0
  run maA_mu_s0 ./run.sh hrl train_hrl mu "${MA[@]}" --ma-design roles --option-set plays --timesteps 90000000 \
    --max-env-steps 4000000 --option-policy /work/runs/maA_opt_s0/final_model.zip --run-name maA_mu_s0
  if [ -f runs/v4_ma_flat_s0/final_model.zip ]; then
    run maflat6_s0 ./run.sh train "${MA[@]}" --timesteps 3000000 --init-from /work/runs/v4_ma_flat_s0/final_model.zip \
      --run-name maflat6_s0
  else  # no earlier 3M run on this machine: train from scratch
    run maflat6_s0 ./run.sh train "${MA[@]}" --timesteps 6000000 --run-name maflat6_s0
  fi
}

lane2() {
  run maB_opt_s0 ./run.sh hrl train_hrl options "${MA[@]}" --ma-design agents --option-set agent "${OPT[@]}" \
    --timesteps 2000000 --run-name maB_opt_s0
  run maB_mu_s0 ./run.sh hrl train_hrl mu "${MA[@]}" --ma-design agents --option-set agent --timesteps 90000000 \
    --max-env-steps 4000000 --option-policy /work/runs/maB_opt_s0/final_model.zip --run-name maB_mu_s0
}

evals() {
  for seed in 1 2 3 4; do
    for a in maflat6:ma_flat:maflat6_s0 maA:ma_hrl:maA_mu_s0:maA_opt_s0 maB:ma_hrl:maB_mu_s0:maB_opt_s0; do
      local label=${a%%:*} agent=${a#*:}
      run "ev_${label}_$seed" ./run.sh hrl evaluate_runs --agent "$agent" --level ma_3_vs_1_with_keeper \
        --episodes 600 --seed "$seed" --label "$label@$seed" --out /work/runs/ma_hrl/results.jsonl
    done
  done
}

stamp "ma_hrl start"
lane1 & lane2 & wait
stamp "training finished"
evals
stamp "ma_hrl finished"
[ "${SHUTDOWN:-1}" = 1 ] && sudo shutdown -h now
