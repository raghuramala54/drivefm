#!/usr/bin/env python
"""Automatic expert data collection with DART noise injection (Laskey et al., CoRL 2017).

The car EXECUTES expert steering + temporally correlated noise, but every frame is LABELLED with the
clean expert steering for the visited state. The noise makes the car drift, and the labels then
show how to recover -- learned recovery without the side-camera hack.

  # DriveLite: 60 training tracks, style 0 (meadow)
  python scripts/collect.py --env drivelite --out data/dl_train --seeds 0-59 --style 0 --noise 0.15
  # held-out offline test sets: unseen track shapes, and unseen shapes + unseen visual style
  python scripts/collect.py --env drivelite --out data/dl_test_shape --seeds 1000-1009 --style 0 --noise 0
  python scripts/collect.py --env drivelite --out data/dl_test_snow  --seeds 1000-1009 --style 2 --noise 0
  # MetaDrive (Python <= 3.11; on a headless server prefix with xvfb-run -a)
  python scripts/collect.py --env metadrive --out data/md_train --seeds 0-99 --map_blocks 4 --noise 0.15
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drivefm.agent import SpeedController  # noqa: E402
from drivefm.envs import make_env, parse_seeds  # noqa: E402
from drivefm.recorder import LogWriter  # noqa: E402


def env_kwargs(args):
    if args.env == "drivelite":
        return dict(style=args.style, side_cameras=getattr(args, "side_cameras", False), randomize_lighting=args.randomize_lighting,
                    max_steps=args.max_steps)
    return dict(map_blocks=args.map_blocks, traffic_density=args.traffic, max_steps=args.max_steps,
                expert=args.expert)


def add_env_args(ap):
    ap.add_argument("--env", default="drivelite", choices=["drivelite", "metadrive"])
    ap.add_argument("--style", type=int, default=0, help="DriveLite visual style 0..3")
    ap.add_argument("--randomize_lighting", action="store_true")
    ap.add_argument("--map_blocks", type=int, default=4, help="MetaDrive blocks per map")
    ap.add_argument("--traffic", type=float, default=0.0, help="MetaDrive traffic density")
    ap.add_argument("--expert", default="idm", choices=["idm", "ppo"], help="MetaDrive expert")
    ap.add_argument("--max_steps", type=int, default=1500)


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_env_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--side_cameras", action="store_true", help="DriveLite: also render left/right cameras")
    ap.add_argument("--noise", type=float, default=0.15, help="DART noise std on executed steering (0 = clean)")
    ap.add_argument("--noise_tau", type=float, default=1.0, help="noise correlation time (s)")
    ap.add_argument("--speed", default="15,25", help="target speed range in mph, sampled per episode")
    ap.add_argument("--drop_failure_s", type=float, default=2.0, help="drop frames this long before a failure")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    env = make_env(args.env, **env_kwargs(args))
    dt = 0.1
    log = LogWriter(args.out, dt=dt)
    lo, hi = (float(x) for x in args.speed.split(","))
    a = np.exp(-dt / max(args.noise_tau, 1e-3))
    stats = {"episodes": 0, "expert_failures": 0, "frames": 0}
    t0 = time.time()
    for seed in parse_seeds(args.seeds):
        obs = env.reset(seed)
        ctl = SpeedController()
        target = rng.uniform(lo, hi)
        n, steps, info = 0.0, 0, {}
        log.begin_episode()
        while True:
            label = env.expert_steer()
            n = a * n + np.sqrt(1 - a * a) * rng.normal(0, args.noise)
            exec_steer = float(np.clip(label + n, -1, 1))
            thr = ctl(target, obs["speed"])
            log.add(obs["image"], label, thr, obs["speed"], obs.get("left"), obs.get("right"),
                    seed=seed, step=steps, exec_steer=f"{exec_steer:.5f}")
            obs, done, info = env.step(exec_steer, thr)
            log.buf[-1]["extra"]["cte"] = f"{info['cte']:.4f}"
            steps += 1
            if done:
                break
        failed = info.get("off_road") or info.get("crash")
        keep = max(0, steps - int(args.drop_failure_s / dt)) if failed else None
        stats["expert_failures"] += int(bool(failed))
        log.end_episode(keep)
        stats["episodes"] += 1
        print(f"[seed {seed}] steps {steps} completion {info.get('route_completion', 0):.2f} "
              f"{'FAIL' if failed else 'ok'}  ({time.time() - t0:.0f}s)", flush=True)
    stats["frames"] = log.frames_written
    log.write_meta({"env": args.env, "args": vars(args), **stats})
    env.close()
    print(f"[done] {stats}  -> {args.out}")


if __name__ == "__main__":
    main()
