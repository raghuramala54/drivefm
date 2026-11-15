#!/usr/bin/env python
"""Summarise DAgger runs: expert labels vs. closed-loop success per round (label-efficiency table).

  python scripts/summarize_dagger.py runs/focused/dagger_* --out results/focused/dagger.md
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="DAgger output folders (each holds dagger_history.json)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    lines = ["| run | style | round | labels this round | frames visited | labelled % | cumulative labels | success | failure | timeout | completion | failures/km |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    style_name = {0: "meadow", 1: "desert", 2: "snow", 3: "dusk"}
    for d in args.runs:
        p = os.path.join(d, "dagger_history.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p, encoding="utf-8")):
            ro, cl = r["rollout"], r.get("closed_loop", {})
            pct = f"{100 * ro['labels'] / ro['frames_visited']:.0f}%" if ro["frames_visited"] else "–"
            sr = cl.get("success_rate", float("nan"))
            to = cl.get("timeout_rate", float("nan"))
            fa = cl.get("failure_rate", 1 - sr - (to if to == to else 0))
            lines.append(f"| {os.path.basename(d.rstrip('/'))} | {style_name.get(r.get('style', 2), r.get('style'))} | {r['iter']} | "
                         f"{ro['labels']} | {ro['frames_visited']} | {pct} | {r['labels_cumulative']} | {sr:.2f} | {fa:.2f} | {to:.2f} | "
                         f"{cl.get('route_completion', float('nan')):.2f} | {cl.get('failures_per_km', float('nan')):.2f} |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
