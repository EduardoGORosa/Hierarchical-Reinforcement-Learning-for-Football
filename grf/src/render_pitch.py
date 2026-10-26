"""Render a GRF episode dump as a top-down tactical video.

gfootball's built-in dump renderer draws a bare rectangle and a few letters,
and its 3D engine needs a GPU-backed GL context that a headless container does
not have. This draws the pitch properly from the trace instead: every player,
the ball, possession, and the controlled agent, in correct proportions.

GRF's normalised coordinates are not isotropic -- x spans [-1, 1] over the
pitch length and y spans [-0.42, 0.42] over its width -- so the two axes are
scaled by different factors to recover a real 105x68 m pitch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

PITCH_L, PITCH_W = 105.0, 68.0          # metres
X_RANGE, Y_RANGE = 2.0, 0.84            # GRF normalised extents

GRASS_DARK = (48, 105, 48)
GRASS_LIGHT = (56, 120, 56)
LINE = (232, 236, 232)
LEFT_TEAM = (210, 130, 40)              # BGR: blue
RIGHT_TEAM = (55, 55, 215)              # BGR: red
CONTROLLED = (0, 225, 255)              # BGR: yellow
BALL = (250, 250, 250)
HUD_BG = (28, 28, 28)


class Pitch:
    """Maps GRF normalised coordinates to pixels on a drawn pitch."""

    def __init__(self, scale: float = 10.0, margin: float = 4.0, hud: int = 46):
        self.s = scale
        self.m = margin
        self.hud = hud
        self.w = int((PITCH_L + 2 * margin) * scale)
        self.h = int((PITCH_W + 2 * margin) * scale) + hud

    def px(self, xm: float, ym: float) -> tuple[int, int]:
        """Metres (origin at pitch corner) -> pixels."""
        return int((xm + self.m) * self.s), int((ym + self.m) * self.s) + self.hud

    def obs(self, x: float, y: float) -> tuple[int, int]:
        """GRF normalised coordinates -> pixels."""
        return self.px((x + 1.0) / X_RANGE * PITCH_L, (y + 0.42) / Y_RANGE * PITCH_W)

    def base(self) -> np.ndarray:
        img = np.zeros((self.h, self.w, 3), np.uint8)
        img[:] = GRASS_DARK
        # Mowing stripes: purely cosmetic, but they make motion easier to follow.
        for i in range(10):
            x0, _ = self.px(i * PITCH_L / 10, 0)
            x1, _ = self.px((i + 1) * PITCH_L / 10, 0)
            if i % 2 == 0:
                img[self.hud:, x0:x1] = GRASS_LIGHT
        img[: self.hud] = HUD_BG

        t = max(1, int(self.s * 0.12))
        r = lambda a, b: cv2.rectangle(img, self.px(*a), self.px(*b), LINE, t)
        r((0, 0), (PITCH_L, PITCH_W))                                  # touchlines
        cv2.line(img, self.px(PITCH_L / 2, 0), self.px(PITCH_L / 2, PITCH_W), LINE, t)
        cv2.circle(img, self.px(PITCH_L / 2, PITCH_W / 2), int(9.15 * self.s), LINE, t)
        cv2.circle(img, self.px(PITCH_L / 2, PITCH_W / 2), max(2, int(0.4 * self.s)), LINE, -1)

        for side in (0, 1):
            near = lambda d: d if side == 0 else PITCH_L - d
            r((near(0), (PITCH_W - 40.32) / 2), (near(16.5), (PITCH_W + 40.32) / 2))
            r((near(0), (PITCH_W - 18.32) / 2), (near(5.5), (PITCH_W + 18.32) / 2))
            cv2.circle(img, self.px(near(11), PITCH_W / 2), max(2, int(0.4 * self.s)), LINE, -1)
            # Goal, drawn just outside the touchline.
            gx = near(0), near(-2.0)
            cv2.rectangle(img, self.px(gx[0], (PITCH_W - 7.32) / 2),
                          self.px(gx[1], (PITCH_W + 7.32) / 2), LINE, max(1, t))
        return img


def draw_frame(pitch: Pitch, base: np.ndarray, obs: dict, step: int, total: int,
               trail: list) -> np.ndarray:
    img = base.copy()
    s = pitch.s

    # Ball trail, oldest faintest, so direction of play reads at a glance.
    for i, (bx, by) in enumerate(trail):
        a = (i + 1) / max(1, len(trail))
        cv2.circle(img, pitch.obs(bx, by), max(1, int(0.25 * s)),
                   tuple(int(c * a) for c in (200, 200, 200)), -1)

    owned = obs.get("ball_owned_team", -1)
    controlled = set(obs.get("left_agent_controlled_player", []) or [])

    for team, key, colour in (("left", "left_team", LEFT_TEAM),
                              ("right", "right_team", RIGHT_TEAM)):
        players = np.asarray(obs[key])
        dirs = np.asarray(obs.get(f"{key}_direction", np.zeros_like(players)))
        for idx, (x, y) in enumerate(players):
            c = pitch.obs(float(x), float(y))
            rad = max(3, int(0.95 * s))
            is_agent = team == "left" and idx in controlled
            if is_agent:
                cv2.circle(img, c, rad + max(2, int(0.5 * s)), CONTROLLED, max(2, int(0.22 * s)))
            cv2.circle(img, c, rad, colour, -1)
            cv2.circle(img, c, rad, (20, 20, 20), max(1, int(0.1 * s)))
            # Velocity arrow, exaggerated: per-step displacement is tiny.
            dx, dy = float(dirs[idx][0]), float(dirs[idx][1])
            if abs(dx) + abs(dy) > 1e-4:
                tip = pitch.obs(float(x) + dx * 6, float(y) + dy * 6)
                cv2.arrowedLine(img, c, tip, (245, 245, 245), max(1, int(0.12 * s)), tipLength=0.35)
            cv2.putText(img, str(idx), (c[0] - int(0.35 * s), c[1] + int(0.35 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30 * s / 10, (255, 255, 255),
                        max(1, int(0.08 * s)), cv2.LINE_AA)

    ball = np.asarray(obs["ball"], dtype=float)
    bc = pitch.obs(ball[0], ball[1])
    height = max(0.0, float(ball[2]))
    # Shadow offset by ball height, so lofted passes are distinguishable.
    cv2.circle(img, (bc[0] + int(height * s * 0.4), bc[1] + int(height * s * 0.4)),
               max(2, int(0.45 * s)), (30, 60, 30), -1)
    cv2.circle(img, bc, max(2, int(0.45 * s + height * 0.3 * s)), BALL, -1)
    cv2.circle(img, bc, max(2, int(0.45 * s + height * 0.3 * s)), (20, 20, 20), 1)

    score = obs.get("score", [0, 0])
    poss = {0: "LEFT", 1: "RIGHT"}.get(owned, "loose")
    f, fs = cv2.FONT_HERSHEY_DUPLEX, 0.52 * s / 10
    cv2.putText(img, f"{score[0]} - {score[1]}", (int(1.2 * s), int(3.0 * s)),
                f, fs * 1.25, (255, 255, 255), max(1, int(0.13 * s)), cv2.LINE_AA)
    cv2.putText(img, f"step {step}/{total}   ball: {poss}", (int(9.5 * s), int(3.0 * s)),
                f, fs, (200, 200, 200), max(1, int(0.1 * s)), cv2.LINE_AA)
    cv2.putText(img, "agent", (pitch.w - int(11.5 * s), int(3.0 * s)),
                f, fs, CONTROLLED, max(1, int(0.1 * s)), cv2.LINE_AA)
    return img


def render(dump_path: Path, out_path: Path, fps: int = 10, scale: float = 10.0,
           trail_len: int = 14) -> Path:
    from gfootball.env import script_helpers

    frames = script_helpers.ScriptHelpers().load_dump(str(dump_path))
    pitch = Pitch(scale=scale)
    base = pitch.base()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (pitch.w, pitch.h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out_path}")

    trail: list = []
    for i, fr in enumerate(frames):
        obs = fr["observation"]
        trail.append((float(obs["ball"][0]), float(obs["ball"][1])))
        trail = trail[-trail_len:]
        writer.write(draw_frame(pitch, base, obs, i + 1, len(frames), trail))
    writer.release()
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+", type=Path)
    ap.add_argument("-o", "--outdir", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--scale", type=float, default=10.0, help="pixels per metre")
    args = ap.parse_args()

    for d in args.dumps:
        out = (args.outdir or d.parent) / (d.stem + ".mp4")
        render(d, out, fps=args.fps, scale=args.scale)
        print(f"{d.name} -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
