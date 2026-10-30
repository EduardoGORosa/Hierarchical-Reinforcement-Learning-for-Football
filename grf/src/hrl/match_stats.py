"""Football events counted from the raw frame on every env step (2026-10-29 quick tests).

MatchStats   match statistics for evaluate_runs: passes, shots, carries past the defender,
             lane shares, and what the off-ball teammates do
SubgoalBonus variation 5: a FLAT agent paid w x (the same events the options are paid for), so
             the flat baseline gets the hierarchy's extra reward knowledge and nothing else
"""

from __future__ import annotations

from collections import Counter

import gymnasium
import numpy as np

EARLY_STEPS = 20  # "early" teammate movement: the first 2 s of the attack


class MatchStats(gymnasium.Wrapper):
    """Counts football events from the raw frame on every env step; at the episode's end puts
    info["match"] = {...}. Sits directly on GFootballEnv (or the multi-agent game), below options.

    pass       our owner changed to another of our players
    shot       the ball left our player towards goal (options.shot_released)
    carry      the player who has the ball got it past the field defender (ball x > his x)
               while keeping it, counted once per possession
    teammates  for our outfield players who don't have the ball: mean forward speed over the first
               EARLY_STEPS steps (x units/step, + is towards their goal), and the mean share of them
               ahead of the ball over the episode
    lanes      share of steps with the ball in each lane (left / centre / right)
    """

    def __init__(self, env):
        super().__init__(env)

    def _raw(self):
        return self.env.unwrapped.raw_observation()

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        raw = self._raw()
        self.t, self.prev = 0, raw
        self.owner = raw["ball_owned_player"] if raw["ball_owned_team"] == 0 else -1
        self.carried = False
        self.c = Counter()
        self.early_vx, self.ahead = [], []
        return out

    def step(self, action):
        from hrl.options import defender_to_beat, lane, shot_released
        obs, r, term, trunc, info = self.env.step(action)
        raw, prev = self._raw(), self.prev
        self.t += 1
        ours = raw["ball_owned_team"] == 0
        owner = raw["ball_owned_player"] if ours else -1
        if ours and self.owner not in (-1, owner):
            self.c["pass"] += 1
            self.carried = False
        if ours and owner != -1:
            if owner != self.owner:
                self.carried = False
            d = defender_to_beat(prev) if prev["ball_owned_team"] == 0 else -1
            if not self.carried and d != -1 and raw["ball"][0] > raw["right_team"][d][0] \
                    and prev["ball"][0] <= prev["right_team"][d][0]:
                self.c["carry"] += 1
                self.carried = True
            self.owner = owner
        if shot_released(prev, raw):
            self.c["shot"] += 1
        self.c["lane_" + lane(float(raw["ball"][1]))] += 1
        ball_x = float(raw["ball"][0])
        mates = [i for i in range(1, len(raw["left_team"])) if i != owner]
        if mates:
            if self.t <= EARLY_STEPS:
                self.early_vx.append(float(np.mean([raw["left_team_direction"][i][0] for i in mates])))
            self.ahead.append(float(np.mean([raw["left_team"][i][0] > ball_x for i in mates])))
        self.prev = raw
        if term or trunc:
            info = dict(info)
            info["episode_goal"] = float(raw["score"][0] > 0)  # for VecMonitor when the reward is shaped
            info["match"] = dict(
                goal=float(raw["score"][0] > 0), steps=self.t,
                passes=self.c["pass"], shots=self.c["shot"], carries=self.c["carry"],
                lanes={k: self.c["lane_" + k] / self.t for k in ("left", "centre", "right")},
                mate_early_vx=float(np.mean(self.early_vx)) if self.early_vx else 0.0,
                mate_ahead=float(np.mean(self.ahead)) if self.ahead else 0.0)
        return obs, r, term, trunc, info


class SubgoalBonus(MatchStats):
    """reward + weight for every pass, carry past the defender and shot on this step: the action
    option set's three subgoals (options.completed_pass / beat_defender / shot_released), read as
    events because a flat agent has no current option. The events are the ones MatchStats counts."""

    EVENTS = ("pass", "carry", "shot")

    def __init__(self, env, weight: float):
        super().__init__(env)
        self.weight = weight

    def step(self, action):
        before = sum(self.c[e] for e in self.EVENTS)
        obs, r, term, trunc, info = super().step(action)
        events = sum(self.c[e] for e in self.EVENTS) - before
        return obs, float(r) + self.weight * events, term, trunc, info
