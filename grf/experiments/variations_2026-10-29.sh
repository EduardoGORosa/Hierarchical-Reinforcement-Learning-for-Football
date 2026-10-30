#!/usr/bin/env bash
# Quick tests of the six variations (2026-10-29). Seed 0, short budgets: a look at behaviour, not results.
# Shares the machine with the overnight main run, so every job runs at low CPU priority, and the two
# training lanes wait for the main run to free memory (each training lane holds ~7 GB).
#
#   lane E  evaluations of existing models (starts at once, 4 workers)
#   lane A  training: plateau fixes (V6), targeted options (V2), scenario transfer (V4)   after the main run
#   lane B  training: subgoal shaping (V5), CF roles (V1), multi-agent (V4)               after the main run
set -uo pipefail
GRF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN=${MAIN_PROGRESS:-/home/dudu/.claude/jobs/bc964cef/tmp/overnight/main_progress.txt}
OUT=$GRF/runs/variations; mkdir -p "$OUT/logs"
export DOCKER_EXTRA="--cpu-shares=256"
cd "$GRF"
stamp() { echo "$(date '+%F %T') $*" >> "$OUT/progress.txt"; }
# Wait until the main run has finished, or its queue script is gone (it failed or was stopped).
wait_main() { while ! grep -q "main finished" "$MAIN" 2>/dev/null && pgrep -f main_queue.sh > /dev/null; do sleep 120; done; }
run() { local n=$1; shift; stamp "start $n"; "$@" > "$OUT/logs/$n.log" 2>&1 && stamp "done  $n" || stamp "FAIL  $n"; }
ev() { local label=$1; shift; run "ev_$label" ./run.sh hrl evaluate_runs --episodes 600 --seed 1 --label "$label" "$@"; }
L3=(--level academy_3_vs_1_with_keeper --rewards scoring)
RPS=academy_run_pass_and_shoot_with_keeper
OPT=(--option-returns --reward-mix both --option-reward-weight 1.0)

laneE() {
  run plateau ./run.sh hrl plateau ov10_mu_s0 m10_mu_s0 ov10_flat_s0
  # V2: "always option z" against the trained mu and flat, seeds 0 and 1
  for s in 0 1; do
    for z in 0 1 2; do ev "v2_const${z}_s$s" --n-envs 4 --agent "const:p3_opt_s$s:$z"; done
    [ -f "runs/m10_mu_s$s/final_model.zip" ] && ev "v2_mu_s$s" --n-envs 4 --agent "hrl:m10_mu_s$s:p3_opt_s$s"
    ev "v2_flat_s$s" --n-envs 4 --agent "flat:ov10_flat_s$s"
  done
  # V1: the same frozen models with CF teammates
  ev v1_cf_flat_s0 --n-envs 4 --level cf_3_vs_1_with_keeper --agent flat:ov10_flat_s0
  ev v1_cf_mu_s0 --n-envs 4 --level cf_3_vs_1_with_keeper --agent hrl:m10_mu_s0:p3_opt_s0
  for z in 0 1; do ev "v1_cf_const${z}_s0" --n-envs 4 --level cf_3_vs_1_with_keeper --agent "const:p3_opt_s0:$z"; done
  # V3: real teams' profiles as tilts (grf/runs/variations/style_profiles.json, from analysis/statsbomb_profiles.py)
  while read -r team w; do for lam in 2 4; do
    ev "v3_${team}_l$lam" --n-envs 4 --agent hrl:m10_mu_s0:p3_opt_s0 --tilt "$lam:$w"; done; done <<'EOF'
barcelona 0.39,0.31,0.30
leicester 0.26,0.42,0.32
leverkusen 0.39,0.26,0.35
arsenal 0.31,0.31,0.37
EOF
  run discover ./run.sh hrl discover --run ov10_flat_s0 --episodes 400 --k 5
  # V4: zero-shot on another scenario (the 3v1 models never saw it)
  ev v4_zs_flat_s0 --n-envs 4 --level $RPS --agent flat:ov10_flat_s0
  ev v4_zs_mu_s0 --n-envs 4 --level $RPS --agent hrl:m10_mu_s0:p3_opt_s0
  ev v4_zs_const1_s0 --n-envs 4 --level $RPS --agent const:p3_opt_s0:1
  stamp "lane E finished"
}

mu_cont() {  # mu from m10_mu_s0 for 1M more env steps on options $2, extra flags after
  local n=$1 opts=$2; shift 2
  run "$n" ./run.sh hrl train_hrl mu "${L3[@]}" --option-set actions --timesteps 9000000 --max-env-steps 1000000 \
    --seed 0 --run-name "$n" --option-policy "/work/runs/$opts/final_model.zip" \
    --init-from /work/runs/m10_mu_s0/final_model.zip "$@"
  ev "$n" --agent "hrl:$n:$opts"
}

laneA() {
  wait_main
  # V6: plateau. Control, then more exploration, then pi fine-tuned under the mu that uses it
  mu_cont v6_ctrl_s0 p3_opt_s0
  mu_cont v6_hient_s0 p3_opt_s0 --ent-coef 0.03 --lr 0.001 --clip-range 0.2
  run v6_joint_opt_s0 ./run.sh hrl train_hrl options "${L3[@]}" --option-set actions "${OPT[@]}" \
    --z-from /work/runs/m10_mu_s0/final_model.zip --end-value-from /work/runs/m10_mu_s0/final_model.zip \
    --init-from /work/runs/p3_opt_s0/final_model.zip --timesteps 1000000 --seed 0 --run-name v6_joint_opt_s0
  mu_cont v6_joint_mu_s0 v6_joint_opt_s0
  # V2: dribble with a target
  run v2_targ_opt_s0 ./run.sh hrl train_hrl options "${L3[@]}" --option-set targeted "${OPT[@]}" \
    --timesteps 2000000 --seed 0 --run-name v2_targ_opt_s0
  run diag_v2_targ ./run.sh hrl diagnose /work/runs/v2_targ_opt_s0
  run v2_targ_mu_s0 ./run.sh hrl train_hrl mu "${L3[@]}" --option-set targeted --timesteps 9000000 \
    --max-env-steps 2000000 --seed 0 --run-name v2_targ_mu_s0 --option-policy /work/runs/v2_targ_opt_s0/final_model.zip
  ev v2_targ_mu_s0 --agent hrl:v2_targ_mu_s0:v2_targ_opt_s0
  # V4: options trained on 3v1, reused on another scenario; mu and flat both from scratch there
  run v4_tr_mu_s0 ./run.sh hrl train_hrl mu --level $RPS --rewards scoring --option-set actions --timesteps 9000000 \
    --max-env-steps 2000000 --seed 0 --run-name v4_tr_mu_s0 --option-policy /work/runs/p3_opt_s0/final_model.zip
  ev v4_tr_mu_s0 --level $RPS --agent hrl:v4_tr_mu_s0:p3_opt_s0
  run v4_tr_flat_s0 ./run.sh train --level $RPS --rewards scoring --timesteps 2000000 --seed 0 --run-name v4_tr_flat_s0
  ev v4_tr_flat_s0 --level $RPS --agent flat:v4_tr_flat_s0
  stamp "lane A finished"
}

laneB() {
  wait_main
  # V5: flat paid for the options' subgoals
  for w in 0.1 1.0; do n="v5_bonus${w/./}_s0"
    run "$n" ./run.sh train "${L3[@]}" --subgoal-bonus "$w" --timesteps 2000000 --seed 0 --run-name "$n"
    ev "$n" --agent "flat:$n"; done
  # V1: flat trained with CF teammates
  run v1_cf_flat_train_s0 ./run.sh train --level cf_3_vs_1_with_keeper --rewards scoring --timesteps 2000000 \
    --seed 0 --run-name v1_cf_flat_train_s0
  ev v1_cf_flat_train_s0 --level cf_3_vs_1_with_keeper --agent flat:v1_cf_flat_train_s0
  # V4: three learning agents, no AI teammates
  run v4_ma_flat_s0 ./run.sh train --level ma_3_vs_1_with_keeper --rewards scoring --agents 3 --timesteps 3000000 \
    --seed 0 --run-name v4_ma_flat_s0
  ev v4_ma_flat_s0 --level ma_3_vs_1_with_keeper --agent ma_flat:v4_ma_flat_s0
  run v4_ma_opt_s0 ./run.sh hrl train_hrl options --level ma_3_vs_1_with_keeper --rewards scoring --agents 3 \
    --option-set actions "${OPT[@]}" --timesteps 2000000 --seed 0 --run-name v4_ma_opt_s0
  run v4_ma_mu_s0 ./run.sh hrl train_hrl mu --level ma_3_vs_1_with_keeper --rewards scoring --agents 3 \
    --option-set actions --timesteps 9000000 --max-env-steps 2000000 --seed 0 --run-name v4_ma_mu_s0 \
    --option-policy /work/runs/v4_ma_opt_s0/final_model.zip
  ev v4_ma_mu_s0 --level ma_3_vs_1_with_keeper --agent ma_hrl:v4_ma_mu_s0:v4_ma_opt_s0
  stamp "lane B finished"
}

stamp "variations start"
laneE & laneA & laneB & wait
stamp "variations finished"
