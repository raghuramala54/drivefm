"""Closed-loop driving agent (simulator-independent, so it can be unit tested).

  * keeps a ring buffer of the last K frames for temporal policies;
  * temporal ensembling of overlapping action chunks (Zhao et al. 2023, ACT);
  * uncertainty-gated speed: the more uncertain the policy, the slower the target speed.
    This is the safety mechanism whose benefit the paper measures in closed loop.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .data import SPEED_NORM, preprocess


class SpeedController:
    """PI speed controller shared by the expert (data collection) and the learned agent."""

    def __init__(self, kp=0.1, ki=0.002):
        self.kp, self.ki, self.integral = kp, ki, 0.0

    def reset(self):
        self.integral = 0.0

    def __call__(self, target: float, speed: float) -> float:
        err = target - speed
        self.integral = float(np.clip(self.integral + err, -200, 200))
        return float(np.clip(self.kp * err + self.ki * self.integral, -1.0, 1.0))


class DrivingAgent:
    def __init__(self, policy, cfg, device="cpu", n_samples=16, mc_dropout=False,
                 max_speed=20.0, min_speed=8.0, sigma_lo=None, sigma_hi=None, gate=True,
                 kp=0.1, ki=0.002, temporal_ensemble=True, te_decay=0.5, unc_source="max",
                 ood_lo=None, ood_hi=None):
        self.policy, self.cfg, self.device = policy, cfg, torch.device(device)
        d = cfg["data"]
        self.K, self.stride, self.H = d["context_frames"], d["frame_stride"], d["chunk"]
        self.frames = deque(maxlen=(self.K - 1) * self.stride + 1)
        self.chunks = deque(maxlen=self.H)
        self.n_samples, self.mc_dropout = n_samples, mc_dropout
        self.max_speed, self.min_speed = max_speed, min_speed
        # thresholds default to validation quantiles stored in the checkpoint (p50 -> no slow-down, p95 -> full)
        ex = getattr(policy, "extra", {}) or {}
        sq, oq = ex.get("sigma_q", {}), ex.get("ood_q", {})
        self.sigma_lo = sigma_lo if sigma_lo is not None else sq.get("p50", 0.03)
        self.sigma_hi = sigma_hi if sigma_hi is not None else max(sq.get("p95", 0.12), self.sigma_lo + 1e-3)
        self.ood_lo = ood_lo if ood_lo is not None else oq.get("p50")
        self.ood_hi = ood_hi if ood_hi is not None else oq.get("p95")
        assert unc_source in {"sigma", "ood", "max"}, unc_source
        self.unc_source, self.gate = unc_source, gate
        self.kp, self.ki, self.integral = kp, ki, 0.0
        self.te, self.te_decay = temporal_ensemble and self.H > 1, te_decay
        self.step = 0

    def reset(self):
        self.frames.clear()
        self.chunks.clear()
        self.integral, self.step = 0.0, 0

    @staticmethod
    def _ramp(u, lo, hi):
        return float(np.clip((u - lo) / max(hi - lo, 1e-6), 0.0, 1.0))

    def risk(self, sigma: float, ood: float | None) -> float:
        """Normalised risk in [0,1]: 0 at/below the validation median, 1 at/above the 95th percentile.
        'sigma' = aleatoric (action spread), 'ood' = epistemic proxy (feature novelty), 'max' = either."""
        r_s = self._ramp(sigma, self.sigma_lo, self.sigma_hi)
        has_ood = ood is not None and self.ood_lo is not None
        r_o = self._ramp(ood, self.ood_lo, self.ood_hi) if has_ood else 0.0
        if self.unc_source == "sigma" or not has_ood:
            return r_s
        if self.unc_source == "ood":
            return r_o
        return max(r_s, r_o)

    def target_speed(self, risk: float) -> float:
        if not self.gate:
            return self.max_speed
        return float(self.max_speed - risk * (self.max_speed - self.min_speed))

    def throttle(self, target: float, speed: float) -> float:
        err = target - speed
        self.integral = float(np.clip(self.integral + err, -200, 200))
        return float(np.clip(self.kp * err + self.ki * self.integral, -1.0, 1.0))

    @torch.no_grad()
    def act(self, img_rgb: np.ndarray, speed: float) -> dict:
        self.frames.append(preprocess(img_rgb, self.cfg))
        while len(self.frames) < self.frames.maxlen:
            self.frames.appendleft(self.frames[0])
        ctx = np.stack([self.frames[i] for i in range(0, self.frames.maxlen, self.stride)])
        x = torch.from_numpy(ctx)[None].to(self.device)
        sp = torch.tensor([speed / SPEED_NORM], dtype=torch.float32, device=self.device)
        out = self.policy.predict(x, sp, self.n_samples, self.mc_dropout)
        chunk = out["action"][0].float().cpu().numpy()
        sigma = float(out["std"][0, 0].float().cpu())
        ood = float(out["ood"][0].float().cpu()) if "ood" in out else None
        self.chunks.append((self.step, chunk))
        if self.te:
            vals, ws = [], []
            for issued, c in self.chunks:
                age = self.step - issued
                if age < len(c):
                    vals.append(c[age])
                    ws.append(np.exp(-self.te_decay * age))
            steer = float(np.dot(vals, ws) / np.sum(ws))
        else:
            steer = float(chunk[0])
        steer = float(np.clip(steer, -1.0, 1.0))
        risk = self.risk(sigma, ood)
        tgt = self.target_speed(risk)
        thr = self.throttle(tgt, speed)
        self.step += 1
        return {"steer": steer, "throttle": thr, "sigma": sigma, "ood": ood, "risk": risk, "target_speed": tgt}
