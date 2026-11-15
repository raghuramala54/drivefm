#!/usr/bin/env python
"""Watch trained models drive in DriveLite: live window and/or saved MP4.

One model:
  python scripts/watch.py --ckpt runs/focused/dinov2_flow_s0/best.pt --style 1 --track 1014

Two models side by side on the same track (e.g. DINOv2 vs PilotNet in desert):
  python scripts/watch.py --ckpt runs/focused/dinov2_flow_s0/best.pt runs/focused/pilotnet_flow_s0/best.pt \
      --style 1 --track 1014 --save videos/desert_1014.mp4

Styles: 0 meadow (training), 1 desert, 2 snow, 3 dusk.  Test tracks: 1000-1019 (never used in training).
Keys in the window: SPACE pause, q quit.  --no_show for machines without a display (MP4 only).
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drivefm.agent import DrivingAgent  # noqa: E402
from drivefm.envs.drivelite import DriveLiteEnv, STYLES  # noqa: E402
from drivefm.utils import get_device, load_policies  # noqa: E402

COLORS = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237)]   # BGR: blue, orange, green, yellow
BG, INK, GRAY, RED = (250, 250, 248), (20, 20, 20), (110, 110, 110), (30, 30, 200)
FONT = cv2.FONT_HERSHEY_SIMPLEX
CAM_W, CAM_H, PAD = 560, 280, 20


def put(img, txt, xy, s=0.55, c=INK, t=1):
    cv2.putText(img, txt, xy, FONT, s, c, t, cv2.LINE_AA)


def meter(img, x, y, w, val, label, col):
    put(img, label, (x, y - 6), 0.42, GRAY)
    cv2.rectangle(img, (x, y), (x + w, y + 10), (225, 225, 222), -1)
    if val is not None:
        cv2.rectangle(img, (x, y), (x + int(w * val), y + 10), col, -1)
    cv2.line(img, (x + w // 2, y - 2), (x + w // 2, y + 12), GRAY, 1)


class Runner:
    def __init__(self, ckpt, args, device, color):
        policy, cfg = load_policies([ckpt], device)
        self.name = os.path.basename(os.path.dirname(ckpt)) or cfg["name"]
        self.agent = DrivingAgent(policy, cfg, device, n_samples=cfg["model"].get("infer_samples", 16),
                                  max_speed=args.max_speed, min_speed=args.min_speed, gate=args.gate != "off",
                                  unc_source=args.gate if args.gate != "off" else "max")
        self.env = DriveLiteEnv(style=args.style)
        self.obs = self.env.reset(args.track)
        self.agent.reset()
        lap_steps = self.env.track.length / (args.min_speed * 0.44704 * 0.1)
        self.env.max_steps = int(1.3 * lap_steps)
        self.color, self.done, self.status, self.path, self.last = color, False, "", [], None

    def step(self):
        if self.done:
            return
        out = self.agent.act(self.obs["image"], self.obs["speed"])
        self.obs, self.done, info = self.env.step(out["steer"], out["throttle"])
        a = self.agent
        s = a._ramp(out["sigma"], a.sigma_lo, a.sigma_hi) if out["sigma"] > 0 else None
        o = a._ramp(out["ood"], a.ood_lo, a.ood_hi) if out["ood"] is not None and a.ood_lo is not None else None
        self.last = dict(out=out, info=info, s=s, o=o)
        self.path.append((self.env.x, self.env.y))
        if self.done:
            self.status = ("LAP COMPLETE" if info["success"] else "LEFT THE ROAD" if info["off_road"] else "TIME LIMIT")
            self.status += f" after {self.env.steps / 10:.1f} s"


def draw(runners, track, style_name, track_id):
    n = len(runners)
    W = PAD + n * (CAM_W + PAD)
    W = max(W, 900)
    H = 60 + CAM_H + 110 + 300
    img = np.full((H, W, 3), BG, np.uint8)
    put(img, f"DriveLite  |  style: {style_name}  |  track {track_id}", (PAD, 32), 0.75, INK, 2)
    for j, r in enumerate(runners):
        x = PAD + j * (CAM_W + PAD)
        cam = cv2.resize(cv2.cvtColor(r.obs["image"], cv2.COLOR_RGB2BGR), (CAM_W, CAM_H))
        if r.done:
            cam = (cam * 0.5 + 125).astype(np.uint8)
        img[50:50 + CAM_H, x:x + CAM_W] = cam
        cv2.rectangle(img, (x, 50), (x + CAM_W, 50 + CAM_H), r.color, 3)
        put(img, r.name, (x + 8, 74), 0.6, (255, 255, 255), 2)
        if r.status:
            put(img, r.status, (x + 20, 50 + CAM_H // 2), 0.9, RED if "LEFT" in r.status else (40, 120, 40), 2)
        if r.last:
            o, inf = r.last["out"], r.last["info"]
            y0 = 50 + CAM_H + 28
            put(img, f"t {r.env.steps / 10:5.1f} s  speed {inf['speed']:4.1f} mph  steer {o['steer']:+.2f}  "
                     f"from centre {inf['cte']:+.2f} m", (x, y0), 0.5)
            meter(img, x, y0 + 35, 250, r.last["s"], "action spread (sigma)" + ("" if r.last["s"] is not None else ": n/a for MSE"), r.color)
            meter(img, x + 290, y0 + 35, 250, r.last["o"], "feature novelty", r.color)
    # map
    xy = track.xy
    lo = xy.min(0) - 6
    size = 280
    sc = (size - 12) / (xy.max(0) + 6 - lo).max()

    def p(q):
        q = (np.atleast_2d(q) - lo) * sc + 6
        q[:, 1] = size - q[:, 1]
        return q.astype(np.int32)
    m = np.full((size, size, 3), BG, np.uint8)
    cv2.polylines(m, [p(xy)], True, (205, 210, 214), max(2, int(track.width * sc)), cv2.LINE_AA)
    for r in runners:
        if len(r.path) > 1:
            cv2.polylines(m, [p(np.array(r.path))], False, r.color, 2, cv2.LINE_AA)
            cv2.circle(m, tuple(p(r.path[-1])[0]), 5, r.color, -1)
    y = 50 + CAM_H + 100
    img[y:y + size, PAD:PAD + size] = m
    put(img, "track from above", (PAD + size + 15, y + 20), 0.45, GRAY)
    put(img, "meters: empty = like training data, middle line = alarm, full = 95th percentile or above", (PAD + size + 15, y + 45), 0.42, GRAY)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="1-4 checkpoints, each drives its own car on the same track")
    ap.add_argument("--style", type=int, default=1, help="0 meadow, 1 desert, 2 snow, 3 dusk")
    ap.add_argument("--track", type=int, default=1014, help="track seed (1000-1019 = unseen test tracks)")
    ap.add_argument("--gate", default="off", choices=["off", "sigma", "ood", "max"], help="risk-gated speed")
    ap.add_argument("--max_speed", type=float, default=20.0)
    ap.add_argument("--min_speed", type=float, default=10.0)
    ap.add_argument("--save", default=None, help="write an MP4 here")
    ap.add_argument("--fps", type=int, default=20, help="video / window speed (10 = real time)")
    ap.add_argument("--no_show", action="store_true", help="no window (headless), use with --save")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    runners = [Runner(c, args, device, COLORS[j % 4]) for j, c in enumerate(args.ckpt[:4])]
    track = runners[0].env.track
    vw = None
    paused = False
    show = not args.no_show
    while True:
        if not paused:
            for r in runners:
                r.step()
        frame = draw(runners, track, STYLES[args.style]["name"], args.track)
        if args.save and not paused:
            if vw is None:
                os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
                vw = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (frame.shape[1], frame.shape[0]))
            vw.write(frame)
        if show:
            try:
                cv2.imshow("DriveLite - watch", frame)
                k = cv2.waitKey(max(1, int(1000 / args.fps))) & 0xFF
            except cv2.error:
                print("[watch] no display available - continuing without a window (use --save to get a video)")
                show, k = False, 255
            if k == ord("q"):
                break
            if k == ord(" "):
                paused = not paused
        if all(r.done for r in runners) and not paused:
            for _ in range(args.fps * 2):
                if vw is not None:
                    vw.write(frame)
            break
    if show:
        cv2.waitKey(1500)
        cv2.destroyAllWindows()
    if vw is not None:
        vw.release()
        print(f"[watch] saved {args.save}")
    for r in runners:
        print(f"  {r.name}: {r.status}")


if __name__ == "__main__":
    main()
