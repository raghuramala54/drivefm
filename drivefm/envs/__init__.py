"""Closed-loop simulators sharing one API.

    env = make_env("drivelite", style=0)          # or make_env("metadrive", map_blocks=4)
    obs = env.reset(seed)                         # obs = {"image": RGB uint8 HxWx3, "speed": mph}
    obs, done, info = env.step(steer, throttle)   # steer + = right, both in [-1, 1]
    label = env.expert_steer()                    # expert steering for the CURRENT state (DART / DAgger)
    info: cte (m), off_road, crash, success, route_completion, timeout, speed
"""
from __future__ import annotations


def make_env(name: str, **kw):
    if name == "drivelite":
        from .drivelite import DriveLiteEnv
        return DriveLiteEnv(**kw)
    if name == "metadrive":
        from .metadrive_env import MetaDriveCamEnv
        return MetaDriveCamEnv(**kw)
    raise ValueError(f"unknown env '{name}' (drivelite | metadrive)")


def parse_seeds(spec: str):
    """'0-49' -> [0..49];  '1,5,9' -> [1,5,9];  '0-9,100-104' -> both ranges."""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out
