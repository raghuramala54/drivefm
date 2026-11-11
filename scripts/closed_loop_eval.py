#!/usr/bin/env python
"""Automatic closed-loop evaluation on UNSEEN tracks/maps (no human in the loop).

  python scripts/closed_loop_eval.py --env drivelite --style 0 --seeds 1000-1019 \
      --ckpt runs/dl_dinov2_flow_s0/best.pt --out results/closed_loop --tag shape
  python scripts/closed_loop_eval.py ... --no_gate            # ablation: constant speed
  python scripts/closed_loop_eval.py ... --ckpt runs/x_s0/best.pt runs/x_s1/best.pt runs/x_s2/best.pt  # ensemble

Per episode: success, route completion, off-road / crash, mean |cross-track error|, steering jerk.
Across steps: fail_auroc = how well policy uncertainty predicts a failure within the next --horizon s;
expert_disagreement = |policy - expert| steering on the states the POLICY visits (closed-loop error).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collect import add_env_args, env_kwargs  # noqa: E402

from drivefm.agent import DrivingAgent  # noqa: E402
from drivefm.envs import make_env, parse_seeds  # noqa: E402
from drivefm.utils import get_device, load_policies  # noqa: E402


def summarize(episodes, steps, horizon_steps):
    from sklearn.metrics import roc_auc_score

    E = episodes
    res = {
        "episodes": len(E),
        "success_rate": float(np.mean([e["success"] for e in E])),
        "route_completion": float(np.mean([e["route_completion"] for e in E])),
        "failure_rate": float(np.mean([e["off_road"] or e["crash"] for e in E])),
        "mean_abs_cte": float(np.nanmean([e["mean_abs_cte"] for e in E])),
        "steer_jerk": float(np.nanmean([e["steer_jerk"] for e in E])),
        "mean_speed_mph": float(np.mean([e["mean_speed"] for e in E])),
        "expert_disagreement": float(np.nanmean([e["expert_disagreement"] for e in E])),
        "km_driven": float(np.sum([e["km"] for e in E])),
    }
    fails = sum(e["off_road"] or e["crash"] for e in E)
    res["failures_per_km"] = float(fails / max(res["km_driven"], 1e-6))
    # failure prediction: label = a failure happens within the next horizon; score = each uncertainty signal
    y = []
    for e, st in zip(E, steps):
        n = len(st["sigma"])
        lab = np.zeros(n, bool)
        if e["off_road"] or e["crash"]:
            lab[max(0, n - horizon_steps):] = True
        y.append(lab)
    y = np.concatenate(y)
    for key, out_key in (("sigma", "fail_auroc_sigma"), ("ood", "fail_auroc_ood"), ("risk", "fail_auroc")):
        sc = np.concatenate([np.asarray(st[key], float) for st in steps])
        ok = np.isfinite(sc)
        if 0 < y[ok].sum() < ok.sum() and np.std(sc[ok]) > 0:
            res[out_key] = float(roc_auc_score(y[ok], sc[ok]))
    res["mean_risk"] = float(np.mean(np.concatenate([st["risk"] for st in steps])))
    # alarm view (risk >= 0.5): did an alarm precede each failure? how often does it fire on good episodes?
    failed = [st for e, st in zip(E, steps) if e["off_road"] or e["crash"]]
    good = [st for e, st in zip(E, steps) if e["success"]]
    if failed:
        res["alarm_recall"] = float(np.mean([np.max(np.asarray(st["risk"])[-horizon_steps:]) >= 0.5 for st in failed]))
    if good:
        res["false_alarm_rate"] = float(np.mean(np.concatenate([np.asarray(st["risk"]) >= 0.5 for st in good])))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_env_args(ap)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--seeds", default="1000-1019")
    ap.add_argument("--out", default="results/closed_loop")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--mc_dropout", action="store_true")
    ap.add_argument("--max_speed", type=float, default=20.0)
    ap.add_argument("--min_speed", type=float, default=10.0)
    ap.add_argument("--sigma_lo", type=float, default=None)
    ap.add_argument("--sigma_hi", type=float, default=None)
    ap.add_argument("--no_gate", action="store_true")
    ap.add_argument("--unc_source", default="max", choices=["sigma", "ood", "max"],
                    help="risk signal for the speed gate: action spread, feature novelty, or the max of both")
    ap.add_argument("--no_temporal_ensemble", action="store_true")
    ap.add_argument("--horizon", type=float, default=2.0, help="failure-prediction horizon (s)")
    ap.add_argument("--no_expert", action="store_true", help="skip expert-disagreement metric (faster)")
    ap.add_argument("--video", type=int, default=0, help="save mp4 of the first N episodes")
    args = ap.parse_args(argv)

    device = get_device(args.device)
    policy, cfg = load_policies(args.ckpt, device)
    agent = DrivingAgent(policy, cfg, device, n_samples=args.samples or cfg["model"]["infer_samples"],
                         mc_dropout=args.mc_dropout, max_speed=args.max_speed, min_speed=args.min_speed,
                         sigma_lo=args.sigma_lo, sigma_hi=args.sigma_hi, gate=not args.no_gate,
                         temporal_ensemble=not args.no_temporal_ensemble, unc_source=args.unc_source)
    lo, hi = agent.sigma_lo, agent.sigma_hi
    gate_label = "off" if args.no_gate else args.unc_source
    env = make_env(args.env, **env_kwargs(args))
    name = f"{cfg['name']}{'_ens' if len(args.ckpt) > 1 else ''}_{args.tag}_gate-{gate_label}"
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)
    tele = open(os.path.join(out_dir, "telemetry.csv"), "w", newline="")
    tw = csv.writer(tele)
    tw.writerow(["seed", "step", "steer", "expert", "sigma", "ood", "risk", "target_speed", "speed", "cte",
                 "latency_ms"])

    episodes, steps_all = [], []
    t0 = time.time()
    for k, seed in enumerate(parse_seeds(args.seeds)):
        obs = env.reset(seed)
        agent.reset()
        st = {"sigma": [], "ood": [], "risk": [], "steer": [], "cte": [], "speed": [], "dis": []}
        vw = None
        if k < args.video:
            import cv2
            vw = cv2.VideoWriter(os.path.join(out_dir, f"seed{seed}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10,
                                 (obs["image"].shape[1], obs["image"].shape[0]))
        while True:
            t1 = time.time()
            out = agent.act(obs["image"], obs["speed"])
            lat = (time.time() - t1) * 1000
            exp_steer = np.nan if args.no_expert else env.expert_steer()
            if vw is not None:
                import cv2
                fr = cv2.cvtColor(obs["image"], cv2.COLOR_RGB2BGR)
                cv2.putText(fr, f"steer {out['steer']:+.2f} sigma {out['sigma']:.3f}", (5, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                vw.write(fr)
            obs, done, info = env.step(out["steer"], out["throttle"])
            ood = np.nan if out["ood"] is None else out["ood"]
            for key, val in (("sigma", out["sigma"]), ("ood", ood), ("risk", out["risk"]),
                             ("steer", out["steer"]), ("cte", info["cte"]),
                             ("speed", info["speed"]), ("dis", abs(out["steer"] - exp_steer))):
                st[key].append(val)
            tw.writerow([seed, len(st["sigma"]), f"{out['steer']:.5f}", f"{exp_steer:.5f}", f"{out['sigma']:.5f}",
                         f"{ood:.4f}", f"{out['risk']:.3f}",
                         f"{out['target_speed']:.2f}", f"{info['speed']:.2f}", f"{info['cte']:.4f}", f"{lat:.1f}"])
            if done:
                break
        if vw is not None:
            vw.release()
        s = np.asarray(st["steer"])
        ep = {"seed": seed, "steps": len(s), "success": info["success"], "off_road": info["off_road"],
              "crash": info["crash"], "route_completion": info["route_completion"],
              "mean_abs_cte": float(np.nanmean(np.abs(st["cte"]))),
              "steer_jerk": float(np.mean(np.abs(np.diff(s, 2)))) if len(s) > 2 else float("nan"),
              "mean_speed": float(np.mean(st["speed"])), "km": float(np.sum(st["speed"]) * 0.44704 * 0.1 / 1000),
              "expert_disagreement": float(np.nanmean(st["dis"])) if not args.no_expert else float("nan")}
        episodes.append(ep)
        steps_all.append(st)
        print(f"[{name}] seed {seed}: {'SUCCESS' if ep['success'] else 'FAIL' if ep['off_road'] or ep['crash'] else 'timeout'}"
              f" completion {ep['route_completion']:.2f} |cte| {ep['mean_abs_cte']:.2f} ({time.time() - t0:.0f}s)",
              flush=True)
    tele.close()
    env.close()
    summary = summarize(episodes, steps_all, int(args.horizon / 0.1))
    result = {"name": name, "model": cfg["name"], "ckpts": args.ckpt, "env": args.env, "tag": args.tag,
              "gate": gate_label, "sigma_lo": lo, "sigma_hi": hi, "ood_lo": agent.ood_lo, "ood_hi": agent.ood_hi, "summary": summary, "episodes": episodes}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}))
    return result


if __name__ == "__main__":
    main()
