"""Offline evaluation: accuracy, smoothness, and uncertainty quality (selective prediction)."""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


@torch.no_grad()
def run_inference(policy, dataset, device, batch_size=64, n_samples=16, mc_dropout=False, num_workers=0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    policy.eval()
    P, S, G, I, O = [], [], [], [], []
    for b in loader:
        out = policy.predict(b["images"].to(device), b["speed"].to(device), n_samples, mc_dropout)
        P.append(out["action"].float().cpu())
        S.append(out["std"].float().cpu())
        G.append(b["action"])
        I.append(b["index"])
        if "ood" in out:
            O.append(out["ood"].float().cpu())
    res = {"pred": torch.cat(P).numpy(), "std": torch.cat(S).numpy(),
           "gt": torch.cat(G).numpy(), "index": torch.cat(I).numpy()}
    if O:
        res["ood"] = torch.cat(O).numpy()
    return res


def risk_coverage(err: np.ndarray, unc: np.ndarray):
    order = np.argsort(unc, kind="stable")
    e = err[order]
    risk = np.cumsum(e) / np.arange(1, len(e) + 1)
    cov = np.arange(1, len(e) + 1) / len(e)
    return cov, risk


def offline_metrics(res: dict, err_thr: float = 0.1) -> dict:
    p, s, g, idx = res["pred"], res["std"], res["gt"], res["index"]
    e0 = np.abs(p[:, 0] - g[:, 0])
    out = {
        "n": int(len(e0)),
        "mae_t0": float(e0.mean()),
        "rmse_t0": float(np.sqrt((e0 ** 2).mean())),
        "mae_chunk": float(np.abs(p - g).mean()),
    }
    consecutive = np.diff(idx) == 1
    if consecutive.any():
        out["jitter_pred"] = float(np.abs(np.diff(p[:, 0]))[consecutive].mean())
        out["jitter_gt"] = float(np.abs(np.diff(g[:, 0]))[consecutive].mean())
    if "ood" in res:   # feature-space novelty: does it flag large errors / domain shift?
        o = res["ood"]
        out["mean_ood"] = float(o.mean())
        lab = e0 > err_thr
        if 0 < lab.sum() < len(lab):
            out["auroc_err_ood"] = float(roc_auc_score(lab, o))
        if np.std(o) > 0:
            out["spearman_ood_err"] = float(spearmanr(o, e0).correlation)
    s0 = s[:, 0]
    if np.all(s0 <= 1e-8):
        return out
    sd = np.maximum(s0, 1e-3)
    out["nll_t0"] = float(np.mean(0.5 * ((p[:, 0] - g[:, 0]) / sd) ** 2 + np.log(sd) + 0.5 * np.log(2 * np.pi)))
    out["cov50"] = float(np.mean(e0 <= 0.674 * sd))
    out["cov90"] = float(np.mean(e0 <= 1.645 * sd))
    out["mean_std"] = float(s0.mean())
    if np.std(s0) > 0:
        out["spearman_unc_err"] = float(spearmanr(s0, e0).correlation)
    labels = e0 > err_thr
    if 0 < labels.sum() < len(labels):
        out["auroc_err"] = float(roc_auc_score(labels, s0))
    cov, risk = risk_coverage(e0, s0)
    out["aurc"] = float(risk.mean())
    _, oracle = risk_coverage(e0, e0)
    out["eaurc"] = float(risk.mean() - oracle.mean())   # excess AURC (lower is better)
    k = int(0.8 * len(e0))
    out["mae_at_cov80"] = float(risk[k - 1]) if k > 0 else float("nan")
    return out


def save_plots(res: dict, path_prefix: str, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p, s, g = res["pred"][:, 0], res["std"][:, 0], res["gt"][:, 0]
    n = min(len(p), 1500)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    x = np.arange(n)
    ax.plot(x, g[:n], lw=1.0, color="#444", label="human")
    ax.plot(x, p[:n], lw=1.0, color="#1f6feb", label="policy")
    if np.any(s > 0):
        ax.fill_between(x, p[:n] - 1.645 * s[:n], p[:n] + 1.645 * s[:n], color="#1f6feb", alpha=0.18, lw=0,
                        label="90% interval")
    ax.set_xlabel("frame")
    ax.set_ylabel("steering")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path_prefix + "_trace.png", dpi=150)
    plt.close(fig)

    if np.any(s > 0):
        e = np.abs(p - g)
        cov, risk = risk_coverage(e, s)
        _, oracle = risk_coverage(e, e)
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.plot(cov, risk, color="#1f6feb", label="uncertainty")
        ax.plot(cov, oracle, color="#444", ls="--", label="oracle")
        ax.axhline(e.mean(), color="#999", lw=0.8, label="no rejection")
        ax.set_xlabel("coverage")
        ax.set_ylabel("MAE of kept frames")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path_prefix + "_risk_coverage.png", dpi=150)
        plt.close(fig)
