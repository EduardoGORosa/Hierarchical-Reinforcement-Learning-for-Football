#!/usr/bin/env bash
# Seed-to-seed spread of the evaluator (2026-10-30): the key comparisons again with eval seeds 2-4.
# evaluate_runs is deterministic for a given seed, so repeats say nothing about noise; seeds do.
set -uo pipefail
GRF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$GRF"; OUT=$GRF/runs/variations
export DOCKER_EXTRA="--cpu-shares=512"
stamp() { echo "$(date '+%F %T') $*" >> "$OUT/progress.txt"; }
ev() { local label=$1 seed=$2; shift 2
  ./run.sh hrl evaluate_runs --episodes 600 --seed "$seed" --label "${label}@$seed" "$@" > "$OUT/logs/ev_${label}_$seed.log" 2>&1 \
    && stamp "done  ev_${label}@$seed" || stamp "FAIL  ev_${label}@$seed"; }
for seed in 2 3 4; do
  ev v2_mu_s0 $seed --agent hrl:m10_mu_s0:p3_opt_s0
  ev v2_const1_s0 $seed --agent const:p3_opt_s0:1
  ev v2_flat_s0 $seed --agent flat:ov10_flat_s0
  ev v1_cf_flat_s0 $seed --level cf_3_vs_1_with_keeper --agent flat:ov10_flat_s0
  ev v1_cf_mu_s0 $seed --level cf_3_vs_1_with_keeper --agent hrl:m10_mu_s0:p3_opt_s0
  ev v6_ctrl_s0 $seed --agent hrl:v6_ctrl_s0:p3_opt_s0
  ev v6_hient_s0 $seed --agent hrl:v6_hient_s0:p3_opt_s0
  ev v3_leicester_l4 $seed --agent hrl:m10_mu_s0:p3_opt_s0 --tilt 4:0.26,0.42,0.32
done
stamp "seeds finished"
