#!/usr/bin/env python
"""Train a driving policy.

  python scripts/train.py --config configs/dinov2_flow.yaml --seed 0
  python scripts/train.py --config configs/pilotnet_mse.yaml --set train.epochs=5 data.num_workers=0
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drivefm.config import load_config  # noqa: E402
from drivefm.data import build_datasets, worker_init_fn  # noqa: E402
from drivefm.evaluation import offline_metrics, run_inference, save_plots  # noqa: E402
from drivefm.models import Policy, count_params, fit_feature_density  # noqa: E402
from drivefm.utils import (EMA, CSVLogger, cosine_with_warmup, dump_yaml, get_device,  # noqa: E402
                           param_groups, save_checkpoint, seed_everything)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", default=None, help="one or more YAML files, merged in order")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="run directory (default runs/<name>_s<seed>)")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=10")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = cfg["seed"]
    run_dir = args.out or cfg["out_dir"] or os.path.join("runs", f"{cfg['name']}_s{seed}")
    os.makedirs(run_dir, exist_ok=True)
    seed_everything(seed)
    device = get_device(cfg["train"]["device"])
    tcfg, dcfg = cfg["train"], cfg["data"]

    train_ds, val_ds, test_sets, stats, scale = build_datasets(cfg, seed)
    print(f"[data] {stats}  action_scale={scale:.4f}")
    model = Policy(cfg, action_scale=scale).to(device)
    if tcfg["init_ckpt"]:
        ck0 = torch.load(tcfg["init_ckpt"], map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck0["state_dict"], strict=False)
        print(f"[init] warm start from {tcfg['init_ckpt']} (missing {len(missing)}, unexpected {len(unexpected)})")
    n_total, n_train = count_params(model)
    print(f"[model] {cfg['model']['backbone']} + {cfg['model']['head']} head | params {n_total/1e6:.2f}M "
          f"(trainable {n_train/1e6:.2f}M) | device {device}")
    dump_yaml(cfg, os.path.join(run_dir, "config.yaml"))

    nw = dcfg["num_workers"]
    loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=len(train_ds) > tcfg["batch_size"],
                        num_workers=nw, worker_init_fn=worker_init_fn if nw > 0 else None,
                        pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    steps_per_epoch = tcfg["steps_per_epoch"] or len(loader)
    total = steps_per_epoch * tcfg["epochs"]
    opt = torch.optim.AdamW(param_groups(model, tcfg["lr"], tcfg["backbone_lr_mult"], tcfg["weight_decay"]),
                            lr=tcfg["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_with_warmup(total, tcfg["warmup_epochs"] * steps_per_epoch))
    use_amp = tcfg["amp"] and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    autocast = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if use_amp else nullcontext
    ema = EMA(model, tcfg["ema_decay"])
    log = CSVLogger(os.path.join(run_dir, "log.csv"))
    n_inf = cfg["model"]["infer_samples"]

    best, best_epoch, bad = float("inf"), -1, 0
    step = 0
    for epoch in range(tcfg["epochs"]):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        it = iter(loader)
        for _ in range(steps_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast():
                loss = model.loss(batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            run_loss += loss.item()
            n += 1
            step += 1
        row = {"epoch": epoch + 1, "step": step, "train_loss": run_loss / max(n, 1),
               "lr": opt.param_groups[0]["lr"], "time_s": round(time.time() - t0, 1)}
        if (epoch + 1) % tcfg["val_every"] == 0 or epoch + 1 == tcfg["epochs"]:
            res = run_inference(ema.module, val_ds, device, tcfg["batch_size"], n_inf)
            vm = offline_metrics(res)
            row.update({f"val_{k}": v for k, v in vm.items()})
            score = vm[tcfg["select_metric"]]
            if score < best:
                best, best_epoch, bad = score, epoch + 1, 0
                sig = res["std"][:, 0]
                # validation uncertainty quantiles -> default thresholds of the closed-loop speed gate
                sigma_q = {"p50": float(np.quantile(sig, 0.5)), "p95": float(np.quantile(sig, 0.95))}
                save_checkpoint(os.path.join(run_dir, "best.pt"), ema.module, cfg,
                                {"epoch": epoch + 1, "val": vm, "action_scale": scale, "sigma_q": sigma_q})
            else:
                bad += 1
        log.log(row)
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()
                         if k in ("epoch", "train_loss", "val_mae_t0", "val_rmse_t0", "val_cov90", "val_auroc_err",
                                  "time_s")))
        if bad >= tcfg["patience"]:
            print(f"[early stop] no improvement for {bad} validations")
            break
    save_checkpoint(os.path.join(run_dir, "last.pt"), ema.module, cfg, {"epoch": epoch + 1})

    # ---- final evaluation with the best checkpoint
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    ema.module.load_state_dict(ck["state_dict"])
    # fit the feature-density novelty detector on un-augmented training frames, calibrate on validation
    plain = copy.copy(train_ds)
    plain.train = False
    n_fit = fit_feature_density(ema.module, DataLoader(plain, batch_size=tcfg["batch_size"], shuffle=False,
                                                         num_workers=nw), device)
    val_res = run_inference(ema.module, val_ds, device, tcfg["batch_size"], n_inf)
    ck["extra"]["val"] = offline_metrics(val_res)
    ck["extra"]["ood_q"] = {"p50": float(np.quantile(val_res["ood"], 0.5)),
                            "p95": float(np.quantile(val_res["ood"], 0.95))}
    save_checkpoint(os.path.join(run_dir, "best.pt"), ema.module, cfg, ck["extra"])
    print(f"[novelty] fitted on {n_fit} frames; val ood p50={ck['extra']['ood_q']['p50']:.3f} "
          f"p95={ck['extra']['ood_q']['p95']:.3f}")
    result = {"name": cfg["name"], "seed": seed, "best_epoch": best_epoch, "params_total": n_total,
              "params_trainable": n_train, "data": stats, "val": ck["extra"]["val"], "test": {}}
    for tname, tds in test_sets.items():
        res = run_inference(ema.module, tds, device, tcfg["batch_size"], n_inf)
        result["test"][tname] = offline_metrics(res)
        # domain-shift detection: can each uncertainty signal tell test frames from validation frames?
        lab = np.r_[np.zeros(len(val_res["ood"])), np.ones(len(res["ood"]))]
        result["test"][tname]["shift_auroc_ood"] = float(roc_auc_score(lab, np.r_[val_res["ood"], res["ood"]]))
        if np.std(np.r_[val_res["std"][:, 0], res["std"][:, 0]]) > 0:
            result["test"][tname]["shift_auroc_sigma"] = float(roc_auc_score(
                lab, np.r_[val_res["std"][:, 0], res["std"][:, 0]]))
        save_plots(res, os.path.join(run_dir, f"test_{tname}"), f"{cfg['name']} on {tname}")
        print(f"[test:{tname}] " + json.dumps({k: round(v, 4) for k, v in result["test"][tname].items()}))
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[done] best {tcfg['select_metric']}={best:.4f} at epoch {best_epoch} -> {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
