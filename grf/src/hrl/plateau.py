"""Why does the hierarchy stop improving after ~5M env steps? (variation 6). Reads TensorBoard logs only.

    ./run.sh hrl plateau ov10_mu_s0 m10_mu_s0 ov10_flat_s0

For each run, per 250k env steps: the policy's entropy (SB3 logs train/entropy_loss = -entropy),
its approximate KL per update, the clip fraction, and the training reward. For mu runs the x-axis is
hrl/env_steps (mu's own counter counts decisions). An entropy near 0 with a flat reward means mu has
become nearly deterministic and stopped exploring; a healthy entropy with a flat reward points to a
ceiling set by the frozen options instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out: dict[str, list[tuple[int, float]]] = {}
    for f in sorted(run_dir.glob("tb/*/events.out.tfevents.*")):
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            out.setdefault(tag, []).extend((e.step, e.value) for e in ea.Scalars(tag))
    return out


def binned(run: str, bin_steps: int = 250_000) -> dict:
    s = scalars(Path("/work/runs") / run)
    steps_of = None
    if "hrl/env_steps" in s:  # mu: map decision counts to env steps
        d = np.asarray(s["hrl/env_steps"])
        steps_of = lambda step: float(np.interp(step, d[:, 0], d[:, 1]))  # noqa: E731
    rows: dict[int, dict[str, list[float]]] = {}
    for tag, key in (("train/entropy_loss", "entropy"), ("train/approx_kl", "kl"),
                     ("train/clip_fraction", "clip"), ("rollout/ep_rew_mean", "reward")):
        for step, v in s.get(tag, []):
            x = steps_of(step) if steps_of else step  # steps since this run started
            rows.setdefault(int(x // bin_steps), {}).setdefault(key, []).append(-v if key == "entropy" else v)
    return {"run": run, "bins": [dict(env_steps=(b + 0.5) * bin_steps,
                                      **{k: float(np.mean(v)) for k, v in rows[b].items()}) for b in sorted(rows)]}


def main(argv: list[str]) -> int:
    out = [binned(r) for r in argv]
    for r in out:
        print(r["run"])
        for b in r["bins"][:: max(1, len(r["bins"]) // 12)]:
            print("  " + "  ".join(f"{k}={v:.4g}" for k, v in b.items()))
    Path("/work/runs/variations").mkdir(parents=True, exist_ok=True)
    Path("/work/runs/variations/plateau.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
