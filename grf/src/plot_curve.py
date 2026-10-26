"""Plot episode-return learning curves from SB3 VecMonitor logs.

Usage: python3 plot_curve.py runs/<run_name> [runs/<other> ...] -o curve.png

Plots return against environment steps, not wall-clock or update count. The
hypothesis is about sample efficiency, so environment interaction is the only
x-axis that lets two agents be compared fairly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(run_dir: Path) -> pd.DataFrame:
    monitor = run_dir / "monitor.csv"
    if not monitor.exists():
        raise FileNotFoundError(f"no monitor.csv in {run_dir}")
    # VecMonitor writes a '#{json}' header line before the CSV header.
    df = pd.read_csv(monitor, skiprows=1)
    df["steps"] = df["l"].cumsum()
    return df


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if len(y) < window or window < 2:
        return y
    return pd.Series(y).rolling(window, min_periods=1).mean().to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", default="curve.png")
    ap.add_argument("--window", type=int, default=100, help="rolling-mean window in episodes")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
    for run in args.runs:
        df = load(run)
        ax.plot(df["steps"], smooth(df["r"].to_numpy(), args.window), label=run.name, lw=1.8)
        ax.plot(df["steps"], df["r"], lw=0.4, alpha=0.15,
                color=ax.lines[-1].get_color())
        print(f"{run.name}: {len(df)} episodes, {int(df['steps'].iloc[-1])} steps, "
              f"final return {smooth(df['r'].to_numpy(), args.window)[-1]:.3f}")

    ax.set_xlabel("environment steps")
    ax.set_ylabel(f"episode return (rolling mean, {args.window} ep)")
    ax.set_title("PPO on Google Research Football")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
