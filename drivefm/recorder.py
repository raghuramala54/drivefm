"""Write simulator episodes in the Udacity format (IMG/ + driving_log.csv), so every simulator
feeds the same training pipeline. Extra columns after the first 7 are ignored by the loader but
kept for analysis (seed, cte, executed steering, sigma, ...)."""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta

import cv2
import numpy as np


class LogWriter:
    EXTRA = ["seed", "step", "cte", "exec_steer", "sigma"]

    def __init__(self, out_dir: str, dt: float = 0.1, jpeg_quality: int = 92):
        self.out_dir = out_dir
        self.img_dir = os.path.join(out_dir, "IMG")
        os.makedirs(self.img_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "driving_log.csv")
        self.dt, self.q = dt, jpeg_quality
        self.t = datetime(2026, 1, 1, 0, 0, 0)
        if os.path.exists(self.csv_path):          # appending: continue after the last timestamp
            n = sum(1 for _ in open(self.csv_path))
            self.t += timedelta(seconds=n * dt + 3600)
        self.buf = []
        self.frames_written = 0

    def begin_episode(self):
        self.buf = []

    def add(self, center, steer, throttle, speed, left=None, right=None, **extra):
        enc = lambda im: cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR),  # noqa: E731
                                      [cv2.IMWRITE_JPEG_QUALITY, self.q])[1].tobytes()
        self.buf.append({"c": enc(center), "l": enc(left) if left is not None else None,
                         "r": enc(right) if right is not None else None,
                         "steer": float(steer), "throttle": float(throttle), "speed": float(speed), "extra": extra})

    def end_episode(self, keep=None):
        """keep: None (all), an int (keep the first N frames) or a boolean mask. Kept frames that are not
        consecutive are separated by a time gap, so they become separate segments for the loader."""
        if keep is None:
            mask = np.ones(len(self.buf), bool)
        elif isinstance(keep, (int, np.integer)):
            mask = np.arange(len(self.buf)) < keep
        else:
            mask = np.asarray(keep, bool)
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            prev = -2
            for i, fr in enumerate(self.buf):
                if not mask[i]:
                    continue
                if i != prev + 1:
                    self.t += timedelta(seconds=60)
                prev = i
                stamp = self.t.strftime("%Y_%m_%d_%H_%M_%S_") + f"{self.t.microsecond // 1000:03d}"
                names = {}
                for cam in ("c", "l", "r"):
                    data = fr[cam] if fr[cam] is not None else fr["c"]
                    full = {"c": "center", "l": "left", "r": "right"}[cam]
                    name = f"{full}_{stamp}.jpg"
                    if fr[cam] is not None or cam == "c":
                        with open(os.path.join(self.img_dir, name), "wb") as g:
                            g.write(data)
                    else:
                        name = f"center_{stamp}.jpg"
                    names[cam] = "IMG/" + name
                ex = fr["extra"]
                w.writerow([names["c"], names["l"], names["r"], f"{fr['steer']:.6f}", f"{fr['throttle']:.4f}", 0,
                            f"{fr['speed']:.4f}"] + [ex.get(k, "") for k in self.EXTRA])
                self.t += timedelta(seconds=self.dt)
                self.frames_written += 1
        self.t += timedelta(seconds=60)
        self.buf = []
        del new

    def write_meta(self, meta: dict):
        path = os.path.join(self.out_dir, "meta.json")
        old = json.load(open(path)) if os.path.exists(path) else {}
        old.setdefault("chunks", []).append(meta)
        old["frames_total"] = self.frames_written + old.get("frames_total", 0)
        with open(path, "w") as f:
            json.dump(old, f, indent=2)
