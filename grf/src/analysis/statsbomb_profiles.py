"""Real teams' attacking style as preference vectors over the options (variation 3). Runs on the host.

    python3 grf/src/analysis/statsbomb_profiles.py <cache dir> <out.json> [matches per team]

Source: StatsBomb open data (github.com/statsbomb/open-data, free, attribution required). For each
team-season, over its first N matches: the team's own Pass, Dribble (a take-on, attempted 1v1) and
Shot events in the attacking third (x > 80 on StatsBomb's 120 x 80 pitch), since a 3v1 academy
attack is a final-third attack. The shares become the style's preference over pass / dribble / shoot,
used by the test-time tilt  logits + lam * ln(w).

Raw counts are dominated by passes (~85% everywhere), which would make every team "pass". What
tells teams apart is how they DIFFER from the average team, so the preference written out is
w = share / mean share over the teams in the list, renormalised: a team that dribbles twice as often
as the average gets twice the average's dribble weight. Both are written, `share` and `w`.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
TEAMS = [  # (label, competition_id, season_id, team name as StatsBomb spells it)
    ("Barcelona 2010/11", 11, 22, "Barcelona"),
    ("Leicester 2015/16", 2, 27, "Leicester City"),
    ("Leverkusen 2023/24", 9, 281, "Bayer Leverkusen"),
    ("Arsenal 2003/04", 2, 44, "Arsenal"),
]
KINDS = ("Pass", "Dribble", "Shot")


def fetch(cache: Path, rel: str):
    path = cache / rel
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{BASE}/{rel}", path)
    return json.loads(path.read_text())


def profile(cache: Path, comp: int, season: int, team: str, n_matches: int) -> dict:
    matches = [m for m in fetch(cache, f"matches/{comp}/{season}.json")
               if team in (m["home_team"]["home_team_name"], m["away_team"]["away_team_name"])]
    matches.sort(key=lambda m: m["match_date"])
    counts = {k: 0 for k in KINDS}
    for m in matches[:n_matches]:
        for e in fetch(cache, f"events/{m['match_id']}.json"):
            kind = e["type"]["name"]
            if kind in KINDS and e["team"]["name"] == team and e.get("location", [0])[0] > 80:
                counts[kind] += 1
    total = sum(counts.values())
    return dict(matches=min(n_matches, len(matches)), counts=counts,
                share=[counts[k] / total for k in KINDS])


def main(argv: list[str]) -> int:
    cache, out = Path(argv[0]), Path(argv[1])
    n = int(argv[2]) if len(argv) > 2 else 8
    rows = {label: profile(cache, c, s, t, n) for label, c, s, t in TEAMS}
    mean = [sum(r["share"][i] for r in rows.values()) / len(rows) for i in range(len(KINDS))]
    for r in rows.values():
        rel = [r["share"][i] / mean[i] for i in range(len(KINDS))]
        r["w"] = [v / sum(rel) for v in rel]
    result = dict(kinds=["pass", "dribble", "shoot"], mean_share=mean, teams=rows,
                  source="StatsBomb open data, attacking third (x > 80)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    for label, r in rows.items():
        print(f"{label:20s} n={r['matches']}  share " + " ".join(f"{v:.3f}" for v in r["share"])
              + "   w " + " ".join(f"{v:.2f}" for v in r["w"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
