"""Exact replay of a logged DriveLite closed-loop lap.

The simulator is deterministic, so feeding back the logged steering command and the throttle that the
PI speed controller produced (recomputed from the logged target speed and the simulated speed) rebuilds
the lap exactly, including every camera frame the policy saw.
"""
from __future__ import annotations

import numpy as np

from .envs.drivelite import DriveLiteEnv


def replay_lap(steer, target_speed, style: int, track: int, kp=0.1, ki=0.002, frame_fn=None):
    """Yield (step, frame) for every frame the policy saw; frame_fn(frame) may transform it (e.g. crop/resize)."""
    env = DriveLiteEnv(style=style, max_steps=10 ** 9)
    obs = env.reset(track)
    integ, spd = 0.0, obs["speed"]
    out = []
    for k in range(len(steer)):
        f = obs["image"]
        out.append(frame_fn(f) if frame_fn else f)
        err = float(target_speed[k]) - spd
        integ = float(np.clip(integ + err, -200, 200))
        thr = float(np.clip(kp * err + ki * integ, -1, 1))
        obs, done, info = env.step(float(steer[k]), thr)
        spd = obs["speed"]
        if done:
            break
    return out, info
