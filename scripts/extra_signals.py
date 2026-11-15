#!/usr/bin/env python
"""Extra failure-prediction baselines, computed on the logged closed-loop test laps (gate off).

For every lap already in results/.../closed_loop, the lap is replayed exactly (drivefm.replay) to recover the
camera frames the policy saw, and three more signals are computed per time step:

  knn   : distance of the policy's context feature to its 10 nearest training features (cosine);
          a strong non-parametric novelty detector (Sun et al., ICML 2022).
  mcd   : MC-dropout spread of the steering command (Gal & Ghahramani 2016); DINOv2 MSE models only
          (they are trained with dropout; flow models already have the action spread sigma).
  ae    : reconstruction error of a convolutional autoencoder trained on training-style frames,
          the signal used by SelfOracle (Stocco et al., ICSE 2020). One autoencoder, shared by all models.

Each signal is normalised like the paper's signals (validation median -> 0, 95th percentile -> 1).
Output: <out>/<model>_s<seed>/<model>_<style>.npz  and  <out>/extra_signals.md (summary tables).

  python scripts/extra_signals.py                      # all models, seeds 0-2, all styles that have results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drivefm.data import build_datasets, load_logs, load_rgb  # noqa: E402
from drivefm.envs.drivelite import STYLES  # noqa: E402
from drivefm.replay import replay_lap  # noqa: E402
from drivefm.utils import get_device, load_policies, seed_everything  # noqa: E402

STYLE_ID = {"shape": 0, **{v["name"]: k for k, v in STYLES.items() if k > 0}}
AE_H, AE_W = 64, 160
H_FAIL = 20


# --------------------------------------------------------------------------- frame preparation (worker side)
def prep(img, crop_top, crop_bottom, size, color):
    h = img.shape[0]
    x = cv2.resize(img[crop_top: h - crop_bottom], (size[1], size[0]), interpolation=cv2.INTER_AREA)
    if color == "yuv":
        x = cv2.cvtColor(x, cv2.COLOR_RGB2YUV)
    return x


def prep_ae(img, crop_top=60, crop_bottom=25):
    h = img.shape[0]
    return cv2.resize(img[crop_top: h - crop_bottom], (AE_W, AE_H), interpolation=cv2.INTER_AREA)


def replay_job(job):
    steer, tgt, style, track, pp = job
    frames, _ = replay_lap(steer, tgt, style, track,
                           frame_fn=lambda f: (prep(f, *pp), prep_ae(f)))
    return np.stack([a for a, _ in frames]), np.stack([b for _, b in frames])


# --------------------------------------------------------------------------- autoencoder baseline
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        c = [3, 32, 64, 128, 128]
        enc, dec = [], []
        for i in range(4):
            enc += [nn.Conv2d(c[i], c[i + 1], 4, 2, 1), nn.ReLU()]
        for i in range(4, 0, -1):
            dec += [nn.ConvTranspose2d(c[i], c[i - 1], 4, 2, 1), nn.ReLU() if i > 1 else nn.Sigmoid()]
        self.enc, self.dec = nn.Sequential(*enc), nn.Sequential(*dec)
        self.fc1, self.fc2 = nn.Linear(128 * 4 * 10, 64), nn.Linear(64, 128 * 4 * 10)

    def forward(self, x):
        z = self.fc1(self.enc(x).flatten(1))
        return self.dec(self.fc2(z).view(-1, 128, 4, 10))


def ae_err(ae, frames_u8, device, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(frames_u8), bs):
            x = torch.from_numpy(frames_u8[i:i + bs]).to(device).permute(0, 3, 1, 2).float() / 255
            out.append(((ae(x) - x) ** 2).mean((1, 2, 3)).cpu().numpy())
    return np.concatenate(out)


def train_ae(train_log, device, out_path, epochs=10, max_frames=30000, seed=0):
    df = load_logs([train_log], 0.5)
    rng = np.random.default_rng(seed)
    paths = df["center"].values
    # hold out every 10th 200-frame block for calibration
    blk = (np.arange(len(paths)) // 200) % 10 == 0
    tr_idx, va_idx = np.flatnonzero(~blk), np.flatnonzero(blk)
    tr_idx = rng.choice(tr_idx, min(max_frames, len(tr_idx)), replace=False)
    load = lambda ids: np.stack([prep_ae(load_rgb(paths[i])) for i in ids])  # noqa: E731
    print(f"[ae] loading {len(tr_idx)} training and {len(va_idx)} validation frames", flush=True)
    Xtr, Xva = load(tr_idx), load(va_idx)
    ae = AE().to(device)
    opt = torch.optim.Adam(ae.parameters(), 1e-3)
    T = torch.from_numpy(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(len(T))
        tot = 0.0
        for i in range(0, len(T), 128):
            x = T[perm[i:i + 128]].to(device).permute(0, 3, 1, 2).float() / 255
            loss = ((ae(x) - x) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(x)
        print(f"[ae] epoch {ep + 1}/{epochs} loss {tot / len(T):.5f}", flush=True)
    ae.eval()
    ev = ae_err(ae, Xva, device)
    q = {"p50": float(np.quantile(ev, 0.5)), "p95": float(np.quantile(ev, 0.95))}
    torch.save({"state_dict": ae.state_dict(), "q": q}, out_path)
    return ae, q


# --------------------------------------------------------------------------- policy-side signals
def frame_ctx(policy, x):
    """x: (N,3,H,W) float -> context vectors (N,d). Assumes one context frame (all study models)."""
    return policy.encode(x[:, None])


@torch.no_grad()
def policy_signals(policy, frames_u8, device, bank, mc=0, bs=256):
    knn, mcd = [], []
    drops = [m for m in policy.modules() if isinstance(m, nn.Dropout) and m.p > 0]
    for i in range(0, len(frames_u8), bs):
        x = torch.from_numpy(frames_u8[i:i + bs]).to(device).permute(0, 3, 1, 2).float() / 255
        ctx = frame_ctx(policy, x).float()
        z = torch.nn.functional.normalize(ctx, dim=1)
        sim = z @ bank.T
        knn.append((1 - sim.topk(10, dim=1).values).mean(1).cpu().numpy())
        if mc and drops:
            tok = policy.backbone(x)
            for d in drops:
                d.train()
            acts = []
            for _ in range(mc):
                c = policy.fuse(policy.frame_proj(tok))
                acts.append(policy.head.predict(c, 1)["action"][:, 0].float())
            for d in drops:
                d.eval()
            mcd.append(torch.stack(acts, 1).std(1).cpu().numpy())
    return np.concatenate(knn), (np.concatenate(mcd) if mcd else None)


@torch.no_grad()
def calib(policy, cfg, seed, device, mc, n_bank=20000, n_val=3000, chunk=512):
    """Feature bank (training frames) and validation quantiles, streamed in chunks to keep RAM low."""
    train_ds, val_ds, _, _, _ = build_datasets(cfg, seed)
    rng = np.random.default_rng(seed)

    def chunks(ds, n):
        ds.train = False
        idx = rng.choice(len(ds), min(n, len(ds)), replace=False)
        for a in range(0, len(idx), chunk):
            X = np.stack([ds[i]["images"][0].numpy() for i in idx[a:a + chunk]])   # (c,3,H,W) float in [0,1]
            yield (X.transpose(0, 2, 3, 1) * 255).round().astype(np.uint8)
    print(f"[calib] feature bank from {min(n_bank, len(train_ds))} training frames", flush=True)
    zs = []
    for u8 in chunks(train_ds, n_bank):
        for b in range(0, len(u8), 256):
            x = torch.from_numpy(u8[b:b + 256]).to(device).permute(0, 3, 1, 2).float() / 255
            zs.append(torch.nn.functional.normalize(frame_ctx(policy, x).float(), dim=1))
    bank = torch.cat(zs)
    K, M = [], []
    for u8 in chunks(val_ds, n_val):
        k, m = policy_signals(policy, u8, device, bank, mc)
        K.append(k)
        if m is not None:
            M.append(m)
    k = np.concatenate(K)
    q = {"knn": [float(np.quantile(k, .5)), float(np.quantile(k, .95))]}
    if M:
        m = np.concatenate(M)
        q["mcd"] = [float(np.quantile(m, .5)), float(np.quantile(m, .95))]
    return bank, q


def ramp(u, q):
    return np.clip((u - q[0]) / max(q[1] - q[0], 1e-9), 0, 1)


# --------------------------------------------------------------------------- summary
def summarize(out_dir, models, styles):
    from sklearn.metrics import roc_auc_score
    rows, alarm_rows = [], []
    for m in models:
        for sig in ("knn", "mcd", "ae", "sigma", "ood"):
            per_seed, within = [], {}
            fa_all, rec_all = [], []
            seeds = sorted({os.path.basename(os.path.dirname(f)) for f in glob.glob(f"{out_dir}/{m}_s*/{m}_*.npz")})
            if not seeds:
                continue
            Y_by_style = {s: [] for s in styles}
            for sd in seeds:
                ys, xs = [], []
                for st in styles:
                    p = f"{out_dir}/{sd}/{m}_{st}.npz"
                    if not os.path.exists(p):
                        continue
                    z = np.load(p, allow_pickle=True)
                    if sig not in z.files:
                        continue
                    for y, x, fail, succ in zip(z["y"], z[sig], z["fail"], z["succ"]):
                        ys.append(y); xs.append(x)
                        Y_by_style[st].append((y, x))
                        x = np.asarray(x, float)
                        if fail:
                            rec_all.append(x[-H_FAIL:].max() >= .5)
                        elif succ:
                            fa_all.append((x >= .5).mean())
                if ys:
                    y, x = np.concatenate(ys).astype(bool), np.concatenate(xs).astype(float)
                    if 0 < y.sum() < len(y) and x.std() > 0:
                        per_seed.append(roc_auc_score(y, x))
            if not any(Y_by_style.values()):
                continue
            for st, L in Y_by_style.items():
                if not L:
                    continue
                y = np.concatenate([a for a, _ in L]).astype(bool); x = np.concatenate([b for _, b in L]).astype(float)
                within[st] = (roc_auc_score(y, x) if 0 < y.sum() < len(y) and x.std() > 0 else float("nan"),
                              float((x >= .5).mean()))
            rows.append((m, sig, np.mean(per_seed) if per_seed else float("nan"), np.std(per_seed) if per_seed else 0,
                         np.mean(rec_all) if rec_all else float("nan"), np.mean(fa_all) if fa_all else float("nan"), within))
    md = ["# Extra failure-prediction signals (gate off, all test laps)", "",
          "AUROC: pooled over styles, mean ± std over seeds. Recall: failures with an alarm in their last 2 s. "
          "FA: alarm rate on successful laps. Then per style: within-style AUROC / fraction of steps with an alarm.", "",
          "| model | signal | AUROC | recall | FA | " + " | ".join(styles) + " |",
          "|---" * (5 + len(styles)) + "|"]
    for m, sig, a, sd, rec, fa, within in rows:
        cells = []
        for st in styles:
            if st in within:
                au, al = within[st]
                cells.append(f"{'–' if au != au else f'{au:.2f}'} / {100 * al:.0f}%")
            else:
                cells.append("–")
        md.append(f"| {m} | {sig} | {a:.2f} ± {sd:.2f} | {100 * rec:.0f}% | {100 * fa:.0f}% | " + " | ".join(cells) + " |")
    open(os.path.join(out_dir, "extra_signals.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/focused")
    ap.add_argument("--cl", default="results/focused/closed_loop")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/focused/extra_signals")
    ap.add_argument("--models", nargs="+", default=["pilotnet_mse", "pilotnet_flow", "dinov2_mse", "dinov2_mse_chunk8", "dinov2_flow"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--mc", type=int, default=16, help="MC-dropout passes (0 = skip)")
    ap.add_argument("--ae_epochs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--summary_only", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    styles_all = ["shape", "desert", "snow", "dusk", "fog", "night", "dust"]
    styles = [s for s in styles_all if glob.glob(f"{args.cl}/*_s*/*_{s}_gate-off/summary.json")]
    if args.summary_only:
        summarize(args.out, args.models, styles)
        return
    device = get_device(args.device)
    seed_everything(0)

    ae_path = os.path.join(args.out, "ae.pt")
    if os.path.exists(ae_path):
        ck = torch.load(ae_path, map_location=device)
        ae = AE().to(device); ae.load_state_dict(ck["state_dict"]); ae.eval(); ae_q = ck["q"]
    else:
        ae, ae_q = train_ae(os.path.join(args.data, "dl_train", "driving_log.csv"), device, ae_path, args.ae_epochs)
    ae_q = [ae_q["p50"], ae_q["p95"]]

    pool = Pool(args.workers)
    for m in args.models:
        for sd in args.seeds:
            run = f"{m}_s{sd}"
            ck = os.path.join(args.runs, run, "best.pt")
            if not os.path.exists(ck):
                print(f"[skip] {ck} not found"); continue
            todo = [st for st in styles if not os.path.exists(f"{args.out}/{run}/{m}_{st}.npz")
                    and os.path.exists(f"{args.cl}/{run}/{m}_{st}_gate-off/telemetry.csv")]
            if not todo:
                continue
            policy, cfg = load_policies([ck], device)
            policy.eval()
            mc = args.mc if (cfg["model"]["head"] == "mse" and cfg["model"].get("dropout", 0) > 0) else 0
            bank, q = calib(policy, cfg, sd, device, mc)
            pp = (cfg["data"]["crop_top"], cfg["data"]["crop_bottom"], tuple(cfg["model"]["img_size"]), cfg["model"]["color"])
            os.makedirs(f"{args.out}/{run}", exist_ok=True)
            for st in todo:
                d = f"{args.cl}/{run}/{m}_{st}_gate-off"
                summ = json.load(open(f"{d}/summary.json"))
                tel = pd.read_csv(f"{d}/telemetry.csv")
                eps = summ["episodes"]
                jobs = []
                for ep in eps:
                    t = tel[tel.seed == ep["seed"]].sort_values("step")
                    jobs.append((t.steer.to_numpy(float), t.target_speed.to_numpy(float), STYLE_ID[st], ep["seed"], pp))
                res = {k: [] for k in ("y", "fail", "succ", "knn", "ae", "sigma", "ood")}
                if mc:
                    res["mcd"] = []
                for ep, (fr, fr_ae), (_, _, _, _, _) in zip(eps, pool.imap(replay_job, jobs), jobs):
                    t = tel[tel.seed == ep["seed"]].sort_values("step")
                    n = min(len(fr), len(t))
                    fr, fr_ae, t = fr[:n], fr_ae[:n], t.iloc[:n]
                    fail = bool(ep["off_road"] or ep["crash"])
                    y = np.zeros(n, bool)
                    if fail:
                        y[-H_FAIL:] = True
                    k, mcd = policy_signals(policy, fr, device, bank, mc)
                    res["y"].append(y); res["fail"].append(fail); res["succ"].append(bool(ep["success"]))
                    res["knn"].append(ramp(k, q["knn"]))
                    if mc:
                        res["mcd"].append(ramp(mcd, q["mcd"]))
                    res["ae"].append(ramp(ae_err(ae, fr_ae, device), ae_q))
                    sq = (summ["sigma_lo"], summ["sigma_hi"])
                    res["sigma"].append(ramp(t.sigma.to_numpy(float), sq) if summ["sigma_hi"] and cfg["model"]["head"] != "mse"
                                        else np.zeros(n))
                    res["ood"].append(ramp(t.ood.to_numpy(float), (summ["ood_lo"], summ["ood_hi"])))
                out = {k: np.array(v, dtype=object) for k, v in res.items()}
                if cfg["model"]["head"] == "mse":
                    out.pop("sigma")
                np.savez(f"{args.out}/{run}/{m}_{st}.npz", **out)
                print(f"[done] {run} {st}: {len(eps)} laps", flush=True)
            del policy
            torch.cuda.empty_cache()
    pool.close()
    summarize(args.out, args.models, styles)


if __name__ == "__main__":
    main()
