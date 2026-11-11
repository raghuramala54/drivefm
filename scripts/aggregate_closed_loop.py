#!/usr/bin/env python
"""Aggregate closed_loop_eval summaries over training seeds -> markdown + LaTeX (mean ± std).

  python scripts/aggregate_closed_loop.py --dir results/closed_loop --out results/table_closed_loop.tex
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

COLS = [("success_rate", "Success$\\uparrow$", 100), ("route_completion", "Compl.$\\uparrow$", 100),
        ("failures_per_km", "Fail/km$\\downarrow$", 1), ("mean_abs_cte", "|CTE| (m)$\\downarrow$", 1),
        ("steer_jerk", "Jerk$\\downarrow$", 1000), ("fail_auroc", "Fail-AUROC$\\uparrow$", 1),
        ("fail_auroc_sigma", "F-AUROC$_\\sigma$", 1), ("fail_auroc_ood", "F-AUROC$_{ood}$", 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/closed_loop")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    groups = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(args.dir, "*", "summary.json"))):
        r = json.load(open(p))
        model = re.sub(r"_s\d+$", "", r["model"])
        groups[(model + ("_ens" if len(r["ckpts"]) > 1 else ""), r["tag"], r["gate"])].append(r["summary"])
    if not groups:
        raise SystemExit("no summary.json found")
    head = ["model", "test set", "gate", "n"] + [c for c, _, _ in COLS]
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    tex = ["\\begin{tabular}{llc" + "c" * len(COLS) + "}", "\\toprule",
           "Model & Test & Gate & " + " & ".join(l for _, l, _ in COLS) + " \\\\", "\\midrule"]
    for (model, tag, gate), items in sorted(groups.items()):
        cells, tcells = [], []
        for key, _, mult in COLS:
            v = np.array([it[key] for it in items if key in it and it[key] == it[key]], float) * mult
            if len(v) == 0:
                cells.append("-")
                tcells.append("--")
            else:
                cells.append(f"{v.mean():.3f} ± {v.std():.3f}")
                tcells.append(f"{v.mean():.2f}$\\pm${v.std():.2f}" if len(v) > 1 else f"{v.mean():.2f}")
        g = gate if isinstance(gate, str) else ("sigma" if gate else "off")
        print(f"| {model} | {tag} | {g} | {len(items)} | " + " | ".join(cells) + " |")
        mark = g
        tex.append(f"{model.replace('_', '-')} & {tag} & {mark} & " + " & ".join(tcells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(tex) + "\n")
        print(f"[latex] -> {args.out}")


if __name__ == "__main__":
    main()
