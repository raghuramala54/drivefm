#!/usr/bin/env python
"""Aggregate runs/*/metrics.json over seeds -> markdown + LaTeX table (mean ± std).

  python scripts/aggregate.py --runs runs --split test --dataset track2_eval
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

METRICS = [("mae_t0", "MAE$\\downarrow$"), ("rmse_t0", "RMSE$\\downarrow$"), ("jitter_pred", "Jitter$\\downarrow$"),
           ("nll_t0", "NLL$\\downarrow$"), ("cov90", "Cov@90"), ("auroc_err", "AUROC$\\uparrow$"),
           ("eaurc", "E-AURC$\\downarrow$")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--dataset", default=None, help="test-set name (folder name of the log); default: first")
    ap.add_argument("--out", default=None, help="write LaTeX table here")
    args = ap.parse_args()

    groups = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(args.runs, "*", "metrics.json"))):
        r = json.load(open(p))
        m = r[args.split]
        if args.split == "test":
            if not m:
                continue
            m = m[args.dataset] if args.dataset else next(iter(m.values()))
        groups[r["name"]].append((m, r.get("params_trainable", 0)))

    if not groups:
        raise SystemExit("no metrics.json found")
    head = ["model", "seeds", "trainable M"] + [k for k, _ in METRICS]
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    tex = ["\\begin{tabular}{l" + "c" * (len(METRICS) + 1) + "}", "\\toprule",
           "Model & Params (M) & " + " & ".join(lbl for _, lbl in METRICS) + " \\\\", "\\midrule"]
    for name, items in groups.items():
        cells, tcells = [], []
        for k, _ in METRICS:
            vals = np.array([m[k] for m, _ in items if k in m and m[k] == m[k]])
            if len(vals) == 0:
                cells.append("-")
                tcells.append("--")
            else:
                cells.append(f"{vals.mean():.4f} ± {vals.std():.4f}")
                tcells.append(f"{vals.mean():.3f}$\\pm${vals.std():.3f}")
        pm = items[0][1] / 1e6
        print(f"| {name} | {len(items)} | {pm:.2f} | " + " | ".join(cells) + " |")
        tex.append(f"{name.replace('_', '-')} & {pm:.2f} & " + " & ".join(tcells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(tex) + "\n")
        print(f"[latex] -> {args.out}")


if __name__ == "__main__":
    main()
