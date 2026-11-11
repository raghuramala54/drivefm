#!/usr/bin/env python
"""DAgger (Ross et al., AISTATS 2011) with optional uncertainty-gated expert queries.

Behavioural cloning only sees states the expert visits; at test time the policy drifts into
states it never saw (covariate shift). DAgger lets the current policy drive, asks the expert
what it would have done in the visited states, adds those labels, and retrains.

--query all          label every visited frame (classic DAgger)
--query uncertainty  label only windows where the policy's risk score (action spread and/or feature
                     novelty, normalised to [0,1] by validation quantiles) exceeds --risk_tau (cf. SafeDAgger, Zhang & Cho
                     2017; EnsembleDAgger, Menda et al. 2019): far fewer expert labels. The label
                     budget vs. closed-loop success trade-off is a natural paper figure.

  python scripts/dagger.py --env drivelite --style 0 --init_ckpt runs/dl_dinov2_flow_s0/best.pt \
      --base_logs data/dl_train/driving_log.csv --iters 3 --seeds_per_iter 10 --query uncertainty \
      --eval_seeds 1000-1019 --out runs/dagger_flow_unc_s0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import closed_loop_eval  # noqa: E402
import train as train_script  # noqa: E402
from collect import add_env_args, env_kwargs  # noqa: E402

from drivefm.agent import DrivingAgent  # noqa: E402
from drivefm.envs import make_env  # noqa: E402
from drivefm.recorder import LogWriter  # noqa: E402
from drivefm.utils import dump_yaml, get_device, load_policy  # noqa: E402


def rollout_and_label(env, policy, cfg, device, seeds, beta, query, tau, out_dir, rng, max_speed, n_samples,
                      unc_source="max"):
    d = cfg["data"]
    ctx_span = (d["context_frames"] - 1) * d["frame_stride"]
    tail = d["chunk"] - 1
    agent = DrivingAgent(policy, cfg, device, n_samples=n_samples, max_speed=max_speed, gate=False,
                         unc_source=unc_source)
    log = LogWriter(out_dir)
    stats = {"frames_visited": 0, "labels": 0, "episodes": 0, "failures": 0}
    for seed in seeds:
        obs = env.reset(seed)
        agent.reset()
        log.begin_episode()
        sig = []
        while True:
            label = env.expert_steer()
            out = agent.act(obs["image"], obs["speed"])
            use_expert = rng.random() < beta
            steer = label if use_expert else out["steer"]
            log.add(obs["image"], label, out["throttle"], obs["speed"], seed=seed, step=len(sig),
                    exec_steer=f"{steer:.5f}", sigma=f"{out['sigma']:.5f}")
            sig.append(out["risk"])
            obs, done, info = env.step(steer, out["throttle"])
            if done:
                break
        sig = np.asarray(sig)
        n = len(sig)
        if query == "all":
            mask = np.ones(n, bool)
        else:
            trig = np.flatnonzero(sig >= tau)
            if info["off_road"] or info["crash"]:            # always learn from the frames before a failure
                trig = np.union1d(trig, np.arange(max(0, n - 20), n))
            mask = np.zeros(n, bool)
            for t in trig:
                mask[max(0, t - ctx_span): min(n, t + tail + 1)] = True
        log.end_episode(mask)
        stats["frames_visited"] += n
        stats["labels"] += int(mask.sum())
        stats["episodes"] += 1
        stats["failures"] += int(info["off_road"] or info["crash"])
        print(f"  [rollout seed {seed}] steps {n} labelled {int(mask.sum())} "
              f"{'FAIL' if info['off_road'] or info['crash'] else 'ok'}", flush=True)
    log.write_meta({"dagger": True, **stats})
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_env_args(ap)
    ap.add_argument("--init_ckpt", required=True, help="behaviour-cloning checkpoint to start from")
    ap.add_argument("--base_logs", nargs="+", required=True, help="original expert logs (kept in every round)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--seeds_per_iter", type=int, default=10)
    ap.add_argument("--seed_start", type=int, default=500, help="DAgger rollouts use fresh training seeds")
    ap.add_argument("--beta", type=float, default=0.5, help="expert mixing prob. in round 1, decays as beta^k")
    ap.add_argument("--query", default="all", choices=["all", "uncertainty"])
    ap.add_argument("--risk_tau", type=float, default=0.5, help="query the expert when risk >= tau (0..1)")
    ap.add_argument("--unc_source", default="max", choices=["sigma", "ood", "max"])
    ap.add_argument("--epochs", type=int, default=10, help="fine-tuning epochs per round")
    ap.add_argument("--max_speed", type=float, default=20.0)
    ap.add_argument("--eval_seeds", default=None, help="closed-loop eval after each round, e.g. 1000-1019")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--train_set", nargs="*", default=[], help="extra train.py overrides")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    device = get_device(args.device)
    rng = np.random.default_rng(args.seed)
    env = make_env(args.env, **env_kwargs(args))
    ckpt, logs, history = args.init_ckpt, list(args.base_logs), []
    total_labels = 0
    for k in range(1, args.iters + 1):
        policy, cfg = load_policy(ckpt, device)
        tau = args.risk_tau
        if args.query == "uncertainty" and cfg["model"]["head"] == "mse" and args.unc_source == "sigma":
            print("[warn] MSE head has no sigma; use --unc_source ood/max, an MDN/flow head, or an ensemble")
        beta = args.beta ** k
        seeds = list(range(args.seed_start + (k - 1) * args.seeds_per_iter, args.seed_start + k * args.seeds_per_iter))
        data_dir = os.path.join(args.out, f"data_iter{k}")
        print(f"[DAgger round {k}] beta={beta:.3f} query={args.query} tau={tau:.4f} seeds {seeds[0]}-{seeds[-1]}")
        st = rollout_and_label(env, policy, cfg, device, seeds, beta, args.query, tau, data_dir, rng,
                               args.max_speed, cfg["model"]["infer_samples"], args.unc_source)
        total_labels += st["labels"]
        if st["labels"] > 0:
            logs.append(os.path.join(data_dir, "driving_log.csv"))
        run_dir = os.path.join(args.out, f"iter{k}")
        cfg_path = os.path.join(args.out, "base_config.yaml")
        base = {k2: v for k2, v in cfg.items()}
        dump_yaml(base, cfg_path)
        train_script.main(["--config", cfg_path, "--out", run_dir, "--set", f"data.train_logs={json.dumps(logs)}",
                           f"train.init_ckpt={ckpt}", f"train.epochs={args.epochs}", "train.warmup_epochs=0",
                           f"name={cfg['name'].split('_dagger')[0]}_dagger_{args.query}", *args.train_set])
        ckpt = os.path.join(run_dir, "best.pt")
        rec = {"iter": k, "beta": beta, "rollout": st, "labels_cumulative": total_labels, "ckpt": ckpt}
        if args.eval_seeds:
            env.close()
            ev = closed_loop_eval.main(["--env", args.env, "--style", str(args.style), "--map_blocks",
                                        str(args.map_blocks), "--ckpt", ckpt, "--seeds", args.eval_seeds,
                                        "--out", os.path.join(args.out, "closed_loop"), "--tag", f"iter{k}",
                                        "--no_expert", "--device", str(device)])
            rec["closed_loop"] = ev["summary"]
            env = make_env(args.env, **env_kwargs(args))
        history.append(rec)
        with open(os.path.join(args.out, "dagger_history.json"), "w") as f:
            json.dump(history, f, indent=2)
    env.close()
    print(f"[done] {len(history)} rounds, {total_labels} expert labels -> {args.out}/dagger_history.json")


if __name__ == "__main__":
    main()
