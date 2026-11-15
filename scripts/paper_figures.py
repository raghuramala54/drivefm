#!/usr/bin/env python
"""Make the paper's driving figures on YOUR machine, by driving your trained models live in DriveLite.

  python scripts/paper_figures.py
  python scripts/paper_figures.py --style 2 --track 1005          # e.g. snow instead of desert

Output folder (default results/focused/paper_figs):
  fig_drive.pdf / .png     map of both laps + camera frames (paper Fig. 3)
  fig_styles.pdf / .png    one test track in the four visual styles (paper Fig. 1)
  drive_stats.json         lap result, time and distance for each model (used in the caption)
  frames/                  every camera frame of each lap as PNG (for slides / videos)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drivefm.agent import DrivingAgent  # noqa: E402
from drivefm.envs.drivelite import DriveLiteEnv, STYLES  # noqa: E402
from drivefm.utils import get_device, load_policies, seed_everything  # noqa: E402

S1, S2, INK2 = "#2a78d6", "#eb6834", "#52514e"
LABEL = {"dinov2_flow": "DINOv2 + flow", "dinov2_mse": "DINOv2 + MSE", "dinov2_mse_chunk8": "DINOv2 + MSE (8-step)",
         "pilotnet_flow": "PilotNet + flow", "pilotnet_mse": "PilotNet + MSE"}


def drive(ckpt, style, track, device, min_speed=10.0):
    policy, cfg = load_policies([ckpt], device)
    agent = DrivingAgent(policy, cfg, device, n_samples=cfg["model"].get("infer_samples", 16), max_speed=20.0,
                         min_speed=min_speed, gate=False)
    env = DriveLiteEnv(style=style)
    obs = env.reset(track)
    agent.reset()
    env.max_steps = int(1.3 * env.track.length / (min_speed * 0.44704 * 0.1))
    frames, path = [obs["image"]], [(env.x, env.y)]
    while True:
        out = agent.act(obs["image"], obs["speed"])
        obs, done, info = env.step(out["steer"], out["throttle"])
        frames.append(obs["image"]); path.append((env.x, env.y))
        if done:
            break
    res = "complete" if info["success"] else "left the road" if info["off_road"] else "time limit"
    return dict(name=cfg["name"], frames=frames, path=np.array(path), env=env, result=res,
                seconds=env.steps / 10, completion=info["route_completion"])


def styles_figure(track, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    titles = ["meadow (training style)", "desert", "snow", "dusk", "fog", "night", "dust"]
    fig, axs = plt.subplots(2, 4, figsize=(7.16, 2.05))
    for st in range(8):
        ax = axs[st // 4, st % 4]
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if st == 7:
            ax.text(0.05, 0.5, "rows: original styles (top),\nharder styles with reduced\nvisibility / contrast (bottom).\nOnly meadow is used for training.",
                    fontsize=6.8, va="center", transform=ax.transAxes, color="#52514e")
            continue
        env = DriveLiteEnv(style=st)
        obs = env.reset(track)
        for _ in range(70):
            obs, _, _ = env.step(env.expert_steer(), 0.5)
        ax.imshow(obs["image"])
        ax.set_title(titles[st], fontsize=7.5, pad=2)
    plt.subplots_adjust(left=0.005, right=0.995, top=0.91, bottom=0.01, wspace=0.03, hspace=0.22)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_styles.{ext}"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/focused")
    ap.add_argument("--models", nargs=2, default=["dinov2_flow_s0", "pilotnet_flow_s0"],
                    help="two run folders under --runs: the first is drawn in blue, the second in orange")
    ap.add_argument("--style", type=int, default=1, help="0 meadow, 1 desert, 2 snow, 3 dusk")
    ap.add_argument("--track", type=int, default=1014, help="unseen test track (1000-1019)")
    ap.add_argument("--styles_track", type=int, default=1003, help="track used for the four-styles figure")
    ap.add_argument("--out", default="results/focused/paper_figs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--styles_only", action="store_true", help="only redraw fig_styles (no models needed)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
    if args.styles_only:
        styles_figure(args.styles_track, args.out)
        print(f"[done] {args.out}/fig_styles.pdf")
        return
    device = get_device(args.device)
    seed_everything(args.seed)

    # ---------------- drive both models live
    laps = []
    for m in args.models:
        ck = os.path.join(args.runs, m, "best.pt")
        print(f"[drive] {m} on {STYLES[args.style]['name']} track {args.track} ...", flush=True)
        lap = drive(ck, args.style, args.track, device)
        print(f"        -> {lap['result']} after {lap['seconds']:.1f} s ({100 * lap['completion']:.0f}% of the lap)")
        d = os.path.join(args.out, "frames", m)
        os.makedirs(d, exist_ok=True)
        for k, f in enumerate(lap["frames"]):
            cv2.imwrite(os.path.join(d, f"{k:04d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        laps.append(lap)
    stats = {m: {k: lap[k] for k in ("name", "result", "seconds", "completion")} for m, lap in zip(args.models, laps)}
    stats["style"], stats["track"] = STYLES[args.style]["name"], args.track
    json.dump(stats, open(os.path.join(args.out, "drive_stats.json"), "w"), indent=1)

    # ---------------- fig_drive: map + four camera frames
    A, B = laps
    tr = A["env"].track
    nrm = np.stack([-tr.tangent[:, 1], tr.tangent[:, 0]], 1)
    Ledge, Redge = tr.xy + nrm * tr.width / 2, tr.xy - nrm * tr.width / 2
    fig = plt.figure(figsize=(7.16, 2.35))
    axm = fig.add_axes([0.0, 0.02, 0.30, 0.86])
    for e in (Ledge, Redge):
        axm.plot(*np.vstack([e, e[:1]]).T, color=INK2, lw=0.5)
    labA, labB = LABEL.get(A["name"], A["name"]), LABEL.get(B["name"], B["name"])
    axm.plot(*A["path"].T, color=S1, lw=1.3, label=f"{labA}: {A['result']}")
    axm.plot(*B["path"].T, color=S2, lw=1.8, label=f"{labB}: {B['result']}")
    for lap, c in ((A, S1), (B, S2)):
        if lap["result"] == "left the road":
            axm.plot(*lap["path"][-1], "x", color=c, ms=7, mew=2)
    axm.plot(*A["path"][0], "o", color="black", ms=4)
    axm.annotate("start", A["path"][0], xytext=(6, -10), textcoords="offset points", fontsize=7)
    axm.set_aspect("equal"); axm.axis("off")
    axm.legend(loc="upper left", bbox_to_anchor=(0.0, 1.17), fontsize=6.5, frameon=False, handlelength=1.2)
    nA, nB = len(A["frames"]), len(B["frames"])
    picks = [(A, int(nA * 0.12), ""), (A, int(nA * 0.55), ""), (B, max(0, nB - 30), ""), (B, nB - 1, ", final frame")]
    for j, (lap, k, note) in enumerate(picks):
        ax = fig.add_axes([0.315 + (j % 2) * 0.343, 0.50 - (j // 2) * 0.48, 0.335, 0.40])
        ax.imshow(lap["frames"][k]); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(S1 if lap is A else S2); s.set_linewidth(1.5)
        ax.set_title(f"{LABEL.get(lap['name'], lap['name'])}, t = {k / 10:.1f} s{note}", fontsize=6.8, pad=2, loc="left")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"fig_drive.{ext}"), dpi=200)
    plt.close(fig)

    # ---------------- fig_styles: the same unseen track in all seven visual styles (expert driving)
    styles_figure(args.styles_track, args.out)
    print(f"[done] figures in {args.out}:  fig_drive.pdf/png, fig_styles.pdf/png, drive_stats.json")


if __name__ == "__main__":
    main()
