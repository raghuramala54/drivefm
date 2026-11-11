#!/usr/bin/env python
"""Generate a tiny Udacity-format dataset (IMG/ + driving_log.csv) with a procedurally drawn
road whose geometry determines the steering label. Used for smoke tests and for checking the
pipeline end to end before recording real simulator data.

  python scripts/make_synthetic_data.py --out data/synth_track1 --frames 600
  python scripts/make_synthetic_data.py --out data/synth_track2 --frames 300 --style 2 --seed 1
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import cv2
import numpy as np

W, H, HORIZON = 320, 160, 62


def render(curv, offset, heading, style, rng):
    sky = (235, 206, 135) if style == 1 else (200, 200, 210)
    ground = (60, 140, 60) if style == 1 else (90, 110, 140)
    road_col = (95, 95, 95) if style == 1 else (60, 60, 70)
    img = np.zeros((H, W, 3), np.uint8)
    img[:HORIZON] = sky
    img[HORIZON:] = ground
    F, half_w = 160.0, 2.0
    for y in range(HORIZON + 1, H):
        z = F * 1.5 / (y - HORIZON)
        xc = 0.5 * curv * z * z - offset - heading * z
        cx = W / 2 + F * xc / z
        hw = F * half_w / z
        x0, x1 = int(max(0, cx - hw)), int(min(W, cx + hw))
        if x1 > x0:
            img[y, x0:x1] = road_col
        for edge in (cx - hw, cx + hw):
            e = int(edge)
            if 0 <= e < W:
                img[y, max(0, e - 1):e + 1] = (240, 240, 240)
        if (int(z * 2) % 2 == 0) and 0 <= int(cx) < W:
            img[y, max(0, int(cx) - 1):int(cx) + 1] = (230, 210, 40)
    noise = rng.normal(0, 4, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--style", type=int, default=1, choices=[1, 2])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    img_dir = os.path.join(args.out, "IMG")
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    t = datetime(2026, 1, 1, 12, 0, 0)
    per = args.frames // args.sessions
    for s in range(args.sessions):
        curv = off = head = 0.0
        for i in range(per):
            curv = 0.97 * curv + rng.normal(0, 0.004)
            off = 0.95 * off + rng.normal(0, 0.05)
            head = 0.9 * head + rng.normal(0, 0.01)
            steer = float(np.clip(6.0 * curv - 0.25 * off - 2.0 * head + rng.normal(0, 0.01), -1, 1))
            stamp = t.strftime("%Y_%m_%d_%H_%M_%S_") + f"{t.microsecond // 1000:03d}"
            names = []
            for cam, d_off in (("center", 0.0), ("left", -0.6), ("right", 0.6)):
                name = f"{cam}_{stamp}.jpg"
                cv2.imwrite(os.path.join(img_dir, name),
                            cv2.cvtColor(render(curv, off + d_off, head, args.style, rng), cv2.COLOR_RGB2BGR))
                names.append(os.path.join("C:\\Users\\sim\\data\\IMG", name))  # Windows-style abs path, as the sim writes
            rows.append(f"{names[0]}, {names[1]}, {names[2]}, {steer:.6f}, 0.5, 0, {20 + rng.normal(0, 1):.4f}")
            t += timedelta(milliseconds=66)
        t += timedelta(seconds=60)  # gap -> new recording segment
    with open(os.path.join(args.out, "driving_log.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"wrote {len(rows)} frames to {args.out}")


if __name__ == "__main__":
    main()
