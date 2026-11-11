#!/usr/bin/env python
"""Closed-loop metrics from drive.py telemetry logs.

  python scripts/closed_loop_report.py closed_loop/*.csv --out closed_loop/summary.json

Metrics
  autonomy_pct      NVIDIA's metric: (1 - interventions * 6 s / elapsed) * 100
  interventions     ENTER keypresses + automatically detected stalls (car stuck after a crash)
  km_per_int        distance driven per intervention
  steer_jerk        mean |second difference| of steering (comfort / smoothness)
  fail_auroc        how well the policy's risk score (sigma / feature novelty) predicts an intervention within the
                    next --horizon seconds (the "does uncertainty foresee failure?" result)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MPH_TO_MPS = 0.44704


def detect_stalls(t, speed, min_moving=5.0, stall_speed=1.0, stall_s=2.0):
    """Indices where a stall begins: car had been moving, then speed < stall_speed for >= stall_s."""
    starts, moving, below_since, in_stall = [], False, None, False
    for i in range(len(t)):
        if speed[i] > min_moving:
            moving, in_stall = True, False
        if speed[i] < stall_speed and moving and not in_stall:
            below_since = t[i] if below_since is None else below_since
            if t[i] - below_since >= stall_s:
                starts.append(int(np.searchsorted(t, below_since)))
                in_stall, moving, below_since = True, False, None
        elif speed[i] >= stall_speed:
            below_since = None
    return starts


def report(path, horizon=2.0, penalty_s=6.0):
    df = pd.read_csv(path)
    auto = df[df["mode"] == "auto"].copy()
    unc = "risk" if "risk" in auto.columns else "sigma"
    auto = auto.rename(columns={unc: "unc"})
    for c in ("speed", "steer", "unc", "latency_ms", "intervention"):
        auto[c] = pd.to_numeric(auto[c], errors="coerce")
    auto = auto.dropna(subset=["speed"]).reset_index(drop=True)
    if len(auto) < 3:
        return {"log": os.path.basename(path), "ticks": len(auto)}
    t = auto["wall_time"].values.astype(float)
    sp = auto["speed"].values
    elapsed = float(t[-1] - t[0])
    dt = np.diff(t, prepend=t[0])
    dist_km = float(np.sum(sp * MPH_TO_MPS * dt) / 1000)
    manual = list(np.flatnonzero(auto["intervention"].fillna(0).values > 0))
    stalls = detect_stalls(t, sp)
    events = sorted(set(manual) | set(stalls))
    n_int = len(events)
    st = auto["steer"].values
    res = {
        "log": os.path.basename(path), "ticks": len(auto), "elapsed_s": round(elapsed, 1),
        "distance_km": round(dist_km, 3), "mean_speed_mph": float(sp.mean()),
        "interventions": n_int, "manual_marks": len(manual), "stalls": len(stalls),
        "autonomy_pct": float(max(0.0, 1 - n_int * penalty_s / max(elapsed, 1e-6)) * 100),
        "km_per_int": float(dist_km / n_int) if n_int else float("inf"),
        "steer_abs_diff": float(np.mean(np.abs(np.diff(st)))),
        "steer_jerk": float(np.mean(np.abs(np.diff(st, 2)))) if len(st) > 2 else float("nan"),
        "latency_ms": float(auto["latency_ms"].mean()),
        "mean_risk": float(auto["unc"].mean()),
    }
    if events and auto["unc"].std() > 0:
        ev_t = t[events]
        label = np.zeros(len(t), bool)
        for et in ev_t:
            label |= (t >= et - horizon) & (t < et)
        mask = np.ones(len(t), bool)       # ignore ticks while the car is stuck after a stall
        for s in stalls:
            stop = np.searchsorted(t, t[s] + 2.0)
            mask[s:stop] = False
        y, sc = label[mask], auto["unc"].values[mask]
        if 0 < y.sum() < len(y):
            res["fail_auroc"] = float(roc_auc_score(y, sc))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--horizon", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [report(p, args.horizon) for p in args.logs]
    cols = ["log", "elapsed_s", "distance_km", "interventions", "autonomy_pct", "km_per_int", "steer_jerk",
            "mean_speed_mph", "fail_auroc", "latency_ms"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(f"{r.get(c, ''):.3f}" if isinstance(r.get(c), float) else str(r.get(c, ""))
                                for c in cols) + " |")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
