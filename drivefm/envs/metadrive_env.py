"""MetaDrive wrapper exposing the same API as DriveLiteEnv.

MetaDrive (Li et al., TPAMI 2022; github.com/metadriverse/metadrive) procedurally composes road
networks from blocks (straights, curves, ramps, roundabouts, intersections), so "train on map
seeds [0, N), test on unseen seeds [1000, ...)" is its standard generalisation protocol.

Install (needs Python 3.9-3.11 and a GPU/OpenGL display for camera rendering):
    pip install metadrive-simulator
Headless Linux without a display:  xvfb-run -a -s "-screen 0 1024x768x24" python ...

Conventions are converted to ours: steering + = RIGHT (MetaDrive uses + = left), images RGB uint8.
"""
from __future__ import annotations

import numpy as np

KMH_TO_MPH = 1 / 1.609344


class MetaDriveCamEnv:
    name = "metadrive"

    def __init__(self, map_blocks=4, traffic_density=0.0, img_size=(160, 320), max_steps=2000,
                 num_scenarios=20000, expert="idm", window=False, extra_config=None):
        from metadrive.component.sensors.rgb_camera import RGBCamera
        from metadrive.envs.metadrive_env import MetaDriveEnv

        H, W = img_size
        cfg = dict(
            use_render=window, image_observation=True, norm_pixel=False, stack_size=1,
            sensors=dict(rgb_camera=(RGBCamera, W, H)), vehicle_config=dict(image_source="rgb_camera"),
            num_scenarios=num_scenarios, start_seed=0, map=map_blocks, traffic_density=traffic_density,
            horizon=max_steps, log_level=50, out_of_road_done=True, crash_vehicle_done=True,
        )
        cfg.update(extra_config or {})
        self.env = MetaDriveEnv(cfg)
        self.expert_kind = expert
        self.side_cameras = False
        self.steps = 0
        self.max_steps = max_steps

    @property
    def agent(self):
        return self.env.agent

    def _obs(self, o):
        img = o["image"]
        if img.ndim == 4:
            img = img[..., -1]
        if img.dtype != np.uint8:
            img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        return {"image": np.ascontiguousarray(img[..., ::-1]),            # BGR -> RGB
                "speed": float(self.agent.speed_km_h) * KMH_TO_MPH}

    def reset(self, seed: int):
        from metadrive.policy.idm_policy import IDMPolicy

        o, _ = self.env.reset(seed=int(seed))
        self.seed, self.steps = int(seed), 0
        self.idm = IDMPolicy(self.agent, int(seed)) if self.expert_kind == "idm" else None
        return self._obs(o)

    def _cte(self):
        v = self.agent
        try:
            return float(v.navigation.current_lane.local_coordinates(v.position)[1])
        except Exception:
            return float("nan")

    def step(self, steer: float, throttle: float):
        o, _, term, trunc, info = self.env.step([-float(np.clip(steer, -1, 1)), float(np.clip(throttle, -1, 1))])
        self.steps += 1
        crash = bool(info.get("crash_vehicle") or info.get("crash_object") or info.get("crash_building")
                     or info.get("crash_human") or info.get("crash_sidewalk"))
        out = {"cte": self._cte(), "off_road": bool(info.get("out_of_road", False)), "crash": crash,
               "success": bool(info.get("arrive_dest", False)),
               "route_completion": float(info.get("route_completion", np.nan)),
               "timeout": bool(trunc and not info.get("arrive_dest", False)),
               "speed": float(self.agent.speed_km_h) * KMH_TO_MPH}
        return self._obs(o), bool(term or trunc), out

    def expert_steer(self) -> float:
        if self.idm is not None:
            a = self.idm.act(self.agent.id)
        else:
            from metadrive.policy.expert_policy import expert
            a = expert(self.agent, deterministic=True)
        return float(-np.clip(a[0], -1, 1))

    def close(self):
        self.env.close()
