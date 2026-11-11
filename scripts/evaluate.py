#!/usr/bin/env python
"""Offline evaluation of one checkpoint or a deep ensemble on any Udacity log.

  python scripts/evaluate.py --ckpt runs/dinov2_flow_s0/best.pt --logs data/track2_eval/driving_log.csv
  python scripts/evaluate.py --ckpt runs/pilotnet_mse_s{0,1,2}/best.pt --logs ... --out runs/ens_pilotnet
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drivefm.data import build_eval_dataset  # noqa: E402
from drivefm.evaluation import offline_metrics, run_inference, save_plots  # noqa: E402
from drivefm.utils import get_device, load_policies  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="one checkpoint, or several = deep ensemble")
    ap.add_argument("--logs", nargs="*", default=None, help="driving_log.csv files (default: config test_logs)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--mc_dropout", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args(argv)

    device = get_device(args.device)
    policy, cfg = load_policies(args.ckpt, device)
    logs = args.logs if args.logs else cfg["data"]["test_logs"]
    if not logs:
        raise SystemExit("No logs given and config has no test_logs")
    out_dir = args.out or os.path.dirname(os.path.abspath(args.ckpt[0]))
    os.makedirs(out_dir, exist_ok=True)
    n = args.samples or cfg["model"]["infer_samples"]
    tag = "ens" if len(args.ckpt) > 1 else "single"
    results = {"ckpts": args.ckpt, "kind": tag, "mc_dropout": args.mc_dropout, "metrics": {}}
    for log in logs:
        name = os.path.basename(os.path.dirname(os.path.abspath(log)))
        ds = build_eval_dataset(cfg, log)
        res = run_inference(policy, ds, device, args.batch_size, n, args.mc_dropout)
        m = offline_metrics(res)
        results["metrics"][name] = m
        save_plots(res, os.path.join(out_dir, f"eval_{tag}_{name}"), f"{cfg['name']} ({tag}) on {name}")
        print(f"[{name}] " + json.dumps({k: round(v, 4) for k, v in m.items()}))
    with open(os.path.join(out_dir, f"eval_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
