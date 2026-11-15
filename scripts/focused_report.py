#!/usr/bin/env python
"""Build the focused-study report (REPORT.md + figures) from training runs and closed-loop results.

  python scripts/focused_report.py --runs runs/focused --cl results/focused/closed_loop --out results/focused

Tables
  1. closed-loop generalisation (no speed gate): success / route completion per test style
  2. offline: steering MAE and domain-shift detection AUROC (sigma vs. feature novelty)
  3. failure prediction pooled over all styles (no gate): AUROC + alarm recall / false alarms per signal
  4. effect of the speed gate (off / sigma / ood / max): failure rate, timeout rate, failures per km
  5. smoothness control: is lower jerk due to the flow head or to 8-step action chunks?
  6. DAgger label efficiency (prefers the redesigned dagger2_* runs; round 0 = starting policy)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ALL_STYLES = ["shape", "desert", "snow", "dusk", "fog", "night", "dust"]
STYLES = ALL_STYLES[:4]   # updated in main() to the styles that have results
STYLE_LABEL = {"shape": "meadow (seen style)", "desert": "desert", "snow": "snow", "dusk": "dusk",
               "fog": "fog", "night": "night", "dust": "dust"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # fixed order, never cycled
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HORIZON = 20   # 2 s at 10 Hz


def ms(v, pct=False, nd=2):
    v = np.asarray([x for x in v if x is not None and x == x], float)
    if len(v) == 0:
        return "–"
    k = 100 if pct else 1
    if len(v) == 1:
        return f"{v[0] * k:.{nd if not pct else 0}f}"
    return f"{v.mean() * k:.{nd if not pct else 0}f} ± {v.std() * k:.{nd if not pct else 0}f}"


def load_cl(cl_dir):
    rows = []
    for p in glob.glob(os.path.join(cl_dir, "*", "*", "summary.json")):
        r = json.load(open(p, encoding="utf-8"))
        r["_dir"] = os.path.dirname(p)
        r["_seed"] = os.path.basename(os.path.dirname(r["_dir"]))
        r["_ens"] = len(r["ckpts"]) > 1
        r["_key"] = r["model"] + (f" ({len(r['ckpts'])}-seed ensemble)" if r["_ens"] else "")
        rows.append(r)
    return rows


def ramp(u, lo, hi):
    if lo is None or hi is None:
        return np.zeros_like(u)
    return np.clip((u - lo) / max(hi - lo, 1e-6), 0, 1)


def failure_signals(r):
    """Per-step labels (failure within HORIZON steps) and normalised signals for one closed-loop run."""
    tel = pd.read_csv(os.path.join(r["_dir"], "telemetry.csv"))
    out = {"y": [], "sigma": [], "ood": [], "max": []}
    for ep in r["episodes"]:
        t = tel[tel["seed"] == ep["seed"]]
        n = len(t)
        y = np.zeros(n, bool)
        if ep["off_road"] or ep["crash"]:
            y[max(0, n - HORIZON):] = True
        rs = ramp(t["sigma"].to_numpy(float), r["sigma_lo"], r["sigma_hi"])
        ro = ramp(t["ood"].to_numpy(float), r.get("ood_lo"), r.get("ood_hi"))
        out["y"].append(y)
        out["sigma"].append(rs)
        out["ood"].append(ro)
        out["max"].append(np.maximum(rs, ro))
    return {k: np.concatenate(v) if v else np.array([]) for k, v in out.items()}, r["episodes"]


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/focused")
    ap.add_argument("--cl", default="results/focused/closed_loop")
    ap.add_argument("--out", default="results/focused")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    md = ["# Focused study: results", "",
          "All closed-loop numbers are on **unseen** DriveLite tracks. Models were trained on meadow-style tracks only.",
          "Mean ± std over training seeds; ensembles combine the seeds.", ""]

    cl = load_cl(args.cl)
    global STYLES
    present = {r["tag"] for r in cl}
    STYLES = [x for x in ALL_STYLES if x in present] or ALL_STYLES[:4]
    models = sorted({r["_key"] for r in cl}, key=lambda k: (("ensemble" in k), k))

    # ---------------- 1. generalisation (gate off)
    md += ["## 1. Closed-loop generalisation (no speed gate)", "",
           "| model | " + " | ".join(f"{STYLE_LABEL[s]}: success" for s in STYLES) + " | " +
           " | ".join(f"{s}: completion" for s in STYLES) + " |", "|---" * (1 + 2 * len(STYLES)) + "|"]
    succ = defaultdict(dict)
    for m in models:
        cells_s, cells_c = [], []
        for s in STYLES:
            rs = [r for r in cl if r["_key"] == m and r["tag"] == s and r["gate"] == "off"]
            sv = [r["summary"]["success_rate"] for r in rs]
            succ[m][s] = sv
            cells_s.append(ms(sv, pct=True) + ("%" if sv else ""))
            cells_c.append(ms([r["summary"]["route_completion"] for r in rs], pct=True) + ("%" if rs else ""))
        md.append(f"| {m} | " + " | ".join(cells_s) + " | " + " | ".join(cells_c) + " |")
    md.append("")

    # ---------------- 2. offline
    md += ["## 2. Offline: steering error and domain-shift detection", "",
           "Shift AUROC: can the signal tell test frames apart from familiar validation frames? 0.5 = blind, 1.0 = perfect.", "",
           "| model | MAE meadow (unseen tracks) | MAE snow | shift AUROC snow: σ | shift AUROC snow: novelty |",
           "|---|---|---|---|---|"]
    off = defaultdict(list)
    for p in glob.glob(os.path.join(args.runs, "*_s[0-9]*", "metrics.json")):
        r = json.load(open(p, encoding="utf-8"))
        off[r["name"]].append(r["test"])
    for m in sorted(off):
        T = off[m]
        g = lambda ds, k: [t.get(ds, {}).get(k) for t in T]  # noqa: E731
        md.append(f"| {m} | {ms(g('dl_test_shape', 'mae_t0'), nd=3)} | {ms(g('dl_test_snow', 'mae_t0'), nd=3)} | "
                  f"{ms(g('dl_test_snow', 'shift_auroc_sigma'))} | {ms(g('dl_test_snow', 'shift_auroc_ood'))} |")
    md.append("")

    # ---------------- 3. failure prediction (pooled over styles, gate off)
    md += ["## 3. Does uncertainty predict crashes? (no gate, pooled over all 4 styles)", "",
           f"Label = the car leaves the road within the next {HORIZON / 10:.0f} s. Signals are normalised so that the "
           "validation median maps to 0 and the 95th percentile to 1. Alarm = signal ≥ 0.5.", "",
           "| model | signal | failure AUROC | alarm recall (failures preceded by an alarm) | false-alarm rate (steps of successful laps) |",
           "|---|---|---|---|---|"]
    fp = defaultdict(lambda: defaultdict(list))
    for m in models:
        for r in [r for r in cl if r["_key"] == m and r["gate"] == "off"]:
            sig, eps = failure_signals(r)
            fp[m]["runs"].append((r, sig, eps))
    auroc_plot = {}
    for m in models:
        runs = fp[m]["runs"]
        if not runs:
            continue
        # group by training seed (= summary.ckpts) so seeds give a spread
        by_ck = defaultdict(list)
        for r, sig, eps in runs:
            by_ck[tuple(r["ckpts"])].append((sig, eps))
        auroc_plot[m] = {}
        for k in ("sigma", "ood", "max"):
            aucs, recs, fas = [], [], []
            for ck, items in by_ck.items():
                y = np.concatenate([s["y"] for s, _ in items])
                x = np.concatenate([s[k] for s, _ in items])
                if 0 < y.sum() < len(y) and np.std(x) > 0:
                    aucs.append(roc_auc_score(y, x))
                rec, fa = [], []
                for s, eps in items:
                    start = 0
                    for ep in eps:
                        n = ep["steps"]
                        seg = s[k][start:start + n]
                        if ep["off_road"] or ep["crash"]:
                            rec.append(seg[-HORIZON:].max() >= 0.5 if n else False)
                        elif ep["success"]:
                            fa.append((seg >= 0.5).mean())
                        start += n
                if rec:
                    recs.append(np.mean(rec))
                if fa:
                    fas.append(np.mean(fa))
            if k == "sigma" and not any(np.std(np.concatenate([s["sigma"] for s, _ in it])) > 0 for it in by_ck.values()):
                continue   # deterministic head: no sigma
            auroc_plot[m][k] = aucs
            name = {"sigma": "σ (action spread)", "ood": "feature novelty", "max": "max(σ, novelty)"}[k]
            md.append(f"| {m} | {name} | {ms(aucs)} | {ms(recs, pct=True)}% | {ms(fas, pct=True)}% |")
    md.append("")

    # ---------------- 4. gate effect
    fr = lambda x: x.get("failure_rate", 1 - x["success_rate"] - x.get("timeout_rate", 0))  # noqa: E731
    md += ["## 4. Does slowing down when uncertain help?", "",
           "Failure = left the road or crashed. Timeout = still driving when the time limit ran out (neither success nor failure).", "",
           "| model | test style | gate | failure rate | timeout rate | failures / km | completion | mean speed (mph) |",
           "|---|---|---|---|---|---|---|---|"]
    for m in models:
        for s in STYLES:
            for g in ("off", "sigma", "ood", "max"):
                rs = [r for r in cl if r["_key"] == m and r["tag"] == s and r["gate"] == g]
                if not rs:
                    continue
                S = [r["summary"] for r in rs]
                md.append(f"| {m} | {s} | {g} | {ms([fr(x) for x in S], pct=True)}% | "
                          f"{ms([x.get('timeout_rate', float('nan')) for x in S], pct=True)}% | "
                          f"{ms([x['failures_per_km'] for x in S], nd=1)} | {ms([x['route_completion'] for x in S], pct=True)}% | "
                          f"{ms([x['mean_speed_mph'] for x in S], nd=1)} |")
    md.append("")

    # ---------------- 5. smoothness control (head vs. chunking)
    ctrl = [m for m in ("dinov2_mse", "dinov2_mse_chunk8", "dinov2_flow", "pilotnet_mse", "pilotnet_flow") if m in models]
    if "dinov2_mse_chunk8" in ctrl:
        desc = {"dinov2_mse": "MSE, 1 step", "dinov2_mse_chunk8": "MSE, 8-step chunks",
                "dinov2_flow": "flow, 8-step chunks", "pilotnet_mse": "MSE, 1 step", "pilotnet_flow": "flow, 8-step chunks"}
        md += ["## 5. Smoothness: flow head or action chunking?", "",
               "Steering jerk = mean |second difference of steering| per step (lower is smoother). Gate off, all styles pooled.", "",
               "| model | head / output | steering jerk (×10⁻³) | success | mean abs. cross-track error (m) |",
               "|---|---|---|---|---|"]
        for m in ctrl:
            per_seed = defaultdict(list)
            for r in cl:
                if r["_key"] == m and r["gate"] == "off":
                    per_seed[r.get("_seed", r["_dir"])].append(r["summary"])
            agg = lambda k: [float(np.nanmean([x[k] for x in S])) for S in per_seed.values()]  # noqa: E731
            md.append(f"| {m} | {desc[m]} | {ms([1e3 * v for v in agg('steer_jerk')], nd=1)} | "
                      f"{ms(agg('success_rate'), pct=True)}% | {ms(agg('mean_abs_cte'), nd=2)} |")
        md.append("")

    # ---------------- 6. DAgger
    dag = sorted(glob.glob(os.path.join(args.runs, "dagger2_*", "dagger_history.json"))) or \
        sorted(glob.glob(os.path.join(args.runs, "dagger_*", "dagger_history.json")))
    if dag:
        style_name = {0: "meadow", 1: "desert", 2: "snow", 3: "dusk"}
        st0 = json.load(open(dag[0], encoding="utf-8"))[-1].get("style", 2)
        md += [f"## 6. Adapting to {style_name.get(st0, st0)} with DAgger", "",
               "Round 0 is the starting policy before any expert labels. Scored on unseen tracks without the speed gate.", "",
               "| run | round | labels this round (% of frames) | cumulative labels | success | failure rate | completion | failures / km |",
               "|---|---|---|---|---|---|---|---|"]
        for p in dag:
            for r in json.load(open(p, encoding="utf-8")):
                ro, c = r["rollout"], r.get("closed_loop", {})
                lab = f"{ro['labels']} ({100 * ro['labels'] / ro['frames_visited']:.0f}%)" if ro["frames_visited"] else "–"
                md.append(f"| {os.path.basename(os.path.dirname(p))} | {r['iter']} | {lab} | {r['labels_cumulative']} | "
                          f"{100 * c.get('success_rate', float('nan')):.0f}% | {100 * fr(c) if c else float('nan'):.0f}% | "
                          f"{100 * c.get('route_completion', float('nan')):.0f}% | {c.get('failures_per_km', float('nan')):.1f} |")
        md.append("")

    # ---------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2})

    single = [m for m in models if "ensemble" not in m]
    if single:
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        w = 0.8 / len(single)
        x = np.arange(len(STYLES))
        for i, m in enumerate(single):
            mean = [np.mean(succ[m][s]) * 100 if succ[m][s] else np.nan for s in STYLES]
            sd = [np.std(succ[m][s]) * 100 if len(succ[m][s]) > 1 else 0 for s in STYLES]
            ax.bar(x + (i - (len(single) - 1) / 2) * w, mean, w * 0.92, yerr=sd, color=SERIES[i], label=m,
                   error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        ax.set_xticks(x, [STYLE_LABEL[s] for s in STYLES])
        ax.set_ylabel("closed-loop success (%)")
        ax.set_ylim(0, 105)
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=min(4, len(single)))
        ax.set_title("Unseen tracks: success by visual style (trained on meadow only)", color=INK, fontsize=10, loc="left",
                     pad=22)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "fig_generalisation.png"), dpi=200)
        plt.close(fig)

    if auroc_plot:
        sigs = ["sigma", "ood", "max"]
        lab = {"sigma": "σ (action spread)", "ood": "feature novelty", "max": "max of both"}
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        ms_ = list(auroc_plot)
        w = 0.8 / len(sigs)
        x = np.arange(len(ms_))
        for i, k in enumerate(sigs):
            mean = [np.mean(auroc_plot[m][k]) if auroc_plot[m].get(k) else np.nan for m in ms_]
            sd = [np.std(auroc_plot[m][k]) if len(auroc_plot[m].get(k, [])) > 1 else 0 for m in ms_]
            ax.bar(x + (i - 1) * w, mean, w * 0.92, yerr=sd, color=SERIES[i], label=lab[k],
                   error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
        ax.text(-0.45, 0.51, "chance", color=INK2, fontsize=8, ha="left", va="bottom")
        ax.set_xticks(x, [re.sub(r" \(\d+-seed ensemble\)", "\n(ensemble)", m) for m in ms_])
        ax.set_ylabel("failure-prediction AUROC")
        ax.set_ylim(0, 1.05)
        style_axes(ax)
        ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
        ax.set_title("Which signal sees a crash coming? (2 s horizon, all styles)", color=INK, fontsize=10, loc="left",
                     pad=22)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "fig_failure_prediction.png"), dpi=200)
        plt.close(fig)

    md += ["## Figures", "", "![generalisation](fig_generalisation.png)", "",
           "![failure prediction](fig_failure_prediction.png)", ""]
    with open(os.path.join(args.out, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
