# academy_3_vs_1_with_keeper with the three attackers as centre forwards (CF) instead of central
# midfielders (CM). Nothing else changes. Why: the engine's off-the-ball AI pulls a player back
# towards his formation spot with weight (1.3 - mindset - team possession)^0.7
# (offtheball/default_off.cpp), and mindset is 0.5 for CM, 1.0 for CF (AIfunctions.cpp:1196).
# At kick-off, with the ball loose, that weight drops from ~0.86 to ~0.43.
# Variation 1 of the 2026-10-29 quick tests: do the AI teammates stop running back?
from . import *


def build_scenario(builder):
  builder.config().game_duration = 400
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = True
  builder.SetBallPosition(0.62, 0.0)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(0.6, 0.0, e_PlayerRole_CF)
  builder.AddPlayer(0.7, 0.2, e_PlayerRole_CF)
  builder.AddPlayer(0.7, -0.2, e_PlayerRole_CF)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(-0.75, 0.0, e_PlayerRole_CB)
