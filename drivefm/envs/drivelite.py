"""DriveLite: a dependency-light procedural driving simulator with a forward RGB camera.

Why it exists: the Udacity simulator is a 2017 Unity build with two fixed tracks and no expert,
so it cannot measure generalisation to *unseen* roads or run automatic closed-loop evaluation.
DriveLite (numpy + scipy + OpenCV only, runs on any laptop, ~100+ FPS with rendering) gives

  * unlimited procedurally generated closed tracks (train on seeds 0..N, test on unseen seeds);
  * several visual styles (train on one, test on others -> visual domain shift);
  * a kinematic bicycle model with steering-actuator lag;
  * a pure-pursuit expert for automatic data collection, DART noise injection and DAgger labels;
  * exact metrics: cross-track error, off-road, route completion.

Conventions: steering command in [-1, 1], positive = turn RIGHT (same as Udacity);
images are RGB uint8, 160 x 320 by default (same as Udacity, so preprocessing is shared).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

MPS_TO_MPH = 2.23694

STYLES = {
    0: dict(name="meadow", sky_top=(80, 140, 225), sky_bot=(190, 215, 240), ground=(70, 130, 55), ground_var=22,
            road=(95, 95, 100), road_var=7, edge=(235, 235, 235), center=(230, 200, 40), center_line=True,
            fog=(200, 215, 230), fog_len=90.0, mountain=(105, 125, 150), shadows=0.35, sun=1.0),
    1: dict(name="desert", sky_top=(110, 155, 220), sky_bot=(235, 215, 175), ground=(200, 170, 120), ground_var=28,
            road=(70, 66, 66), road_var=9, edge=(240, 240, 240), center=(240, 240, 240), center_line=False,
            fog=(230, 215, 190), fog_len=110.0, mountain=(170, 120, 90), shadows=0.2, sun=1.1),
    2: dict(name="snow", sky_top=(150, 160, 178), sky_bot=(212, 216, 222), ground=(226, 229, 236), ground_var=10,
            road=(112, 112, 118), road_var=10, edge=(250, 250, 250), center=(200, 170, 60), center_line=True,
            fog=(215, 218, 225), fog_len=60.0, mountain=(160, 165, 182), shadows=0.1, sun=0.9),
    3: dict(name="dusk", sky_top=(35, 38, 90), sky_bot=(232, 122, 72), ground=(48, 78, 44), ground_var=18,
            road=(58, 58, 64), road_var=6, edge=(205, 205, 190), center=(210, 170, 40), center_line=True,
            fog=(120, 82, 84), fog_len=70.0, mountain=(40, 40, 62), shadows=0.5, sun=0.6),
}


# --------------------------------------------------------------------------- track generation
@dataclass
class Track:
    xy: np.ndarray        # (N,2) centreline, counter-clockwise
    tangent: np.ndarray   # (N,2) unit tangents
    kappa: np.ndarray     # (N,) signed curvature (+ = left)
    ds: float
    width: float
    seed: int

    @property
    def length(self):
        return len(self.xy) * self.ds


def make_track(seed: int, width: float = 8.0, ds: float = 0.5, min_radius: float = 15.0) -> Track:
    rng = np.random.default_rng(seed)
    for _ in range(200):
        n = int(rng.integers(8, 15))
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-0.25, 0.25, n) * (2 * np.pi / n)
        base = rng.uniform(55, 110)
        rad = base * (1 + rng.uniform(-0.38, 0.38, n))
        pts = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
        pts = np.vstack([pts, pts[:1]])
        chord = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        cs = CubicSpline(chord, pts, bc_type="periodic")
        dense_t = np.linspace(0, chord[-1], 20000, endpoint=False)
        dense = cs(dense_t)
        seg = np.linalg.norm(np.diff(np.vstack([dense, dense[:1]]), axis=0), axis=1)
        arc = np.concatenate([[0], np.cumsum(seg)])
        L = arc[-1]
        n_pts = int(L / ds)
        s_new = np.arange(n_pts) * (L / n_pts)
        t_new = np.interp(s_new, arc[:-1], dense_t)
        xy = cs(t_new)
        d1, d2 = cs(t_new, 1), cs(t_new, 2)
        speed = np.linalg.norm(d1, axis=1)
        tangent = d1 / speed[:, None]
        kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / speed ** 3
        if np.abs(kappa).max() > 1.0 / min_radius:
            continue
        # no near self-intersection: points closer than 2.5 road widths must be close in arc length
        tree = cKDTree(xy)
        pairs = np.array(list(tree.query_pairs(2.5 * width)), dtype=np.int64)
        if len(pairs):
            gap = np.abs(pairs[:, 0] - pairs[:, 1])
            gap = np.minimum(gap, n_pts - gap) * (L / n_pts)
            if (gap > 6 * width).any():
                continue
        return Track(xy, tangent, kappa, L / n_pts, width, seed)
    raise RuntimeError(f"could not generate a valid track for seed {seed}")


# --------------------------------------------------------------------------- camera / renderer
class Camera:
    """Pinhole camera on the car; ground rays are precomputed once (they are static in the car frame)."""

    def __init__(self, W=320, H=160, fov_deg=80.0, height=1.4, pitch_deg=3.0, forward=1.2, max_dist=160.0):
        self.W, self.H = W, H
        f = (W / 2) / np.tan(np.radians(fov_deg) / 2)
        u, v = np.meshgrid(np.arange(W) + 0.5 - W / 2, np.arange(H) + 0.5 - H / 2)
        xc, yc = u / f, v / f
        p = np.radians(pitch_deg)
        dF = np.cos(p) - yc * np.sin(p)     # forward
        dD = np.sin(p) + yc * np.cos(p)     # down
        dR = xc                             # right
        norm = np.sqrt(dF ** 2 + dD ** 2 + dR ** 2)
        ground = dD > 1e-4
        t = np.where(ground, height / np.where(ground, dD, 1), np.inf)
        X, Y = t * dF, t * dR
        ground &= X < max_dist
        self.g_idx = np.flatnonzero(ground.ravel())
        self.gX, self.gY = X.ravel()[self.g_idx], Y.ravel()[self.g_idx]
        self.g_dist = (t * norm).ravel()[self.g_idx]
        self.s_idx = np.flatnonzero(~ground.ravel())
        self.s_elev = np.arctan2(-dD, np.sqrt(dF ** 2 + dR ** 2)).ravel()[self.s_idx]   # + = above horizon
        self.s_az = np.arctan2(dR, dF).ravel()[self.s_idx]                               # + = right
        self.forward = forward
        hood = np.zeros((H, W), bool)
        yy, xx = np.mgrid[0:H, 0:W]
        hood |= (yy > H - 13) & (np.abs(xx - W / 2) < W * 0.42 + (yy - (H - 13)) * 3)
        self.hood = hood


class Renderer:
    def __init__(self, track: Track, style: dict, camera: Camera, seed: int):
        self.track, self.style, self.cam = track, style, camera
        self.tree = cKDTree(track.xy)
        rng = np.random.default_rng(seed + 7919)
        self.tex = rng.normal(0, 1, (256, 256)).astype(np.float32)                    # fine grain, 0.5 m cells
        sh = cv2.GaussianBlur(rng.normal(0, 1, (256, 256)).astype(np.float32), (0, 0), 6)
        self.shadow = (sh > np.quantile(sh, 0.8)).astype(np.float32)                   # blobs, 1 m cells
        self.shadow = cv2.GaussianBlur(self.shadow, (0, 0), 1.0)
        k = np.arange(1, 6)
        self.m_amp = rng.uniform(0.004, 0.02, 5) / k ** 0.3
        self.m_phase = rng.uniform(0, 2 * np.pi, 5)
        self.m_k = k * rng.integers(1, 4, 5)

    def render(self, x, y, theta, lateral_offset=0.0, brightness=1.0):
        st, cam, tr = self.style, self.cam, self.track
        fwd = np.array([np.cos(theta), np.sin(theta)])
        right = np.array([np.sin(theta), -np.cos(theta)])
        c = np.array([x, y]) + fwd * cam.forward - right * lateral_offset
        wx = c[0] + cam.gX * fwd[0] + cam.gY * right[0]
        wy = c[1] + cam.gX * fwd[1] + cam.gY * right[1]
        _, idx = self.tree.query(np.stack([wx, wy], 1))
        cxy, tg = tr.xy[idx], tr.tangent[idx]
        lat = tg[:, 0] * (wy - cxy[:, 1]) - tg[:, 1] * (wx - cxy[:, 0])
        a = np.abs(lat)
        hw = tr.width / 2
        n = self.tex[(np.floor(wx * 2).astype(np.int64)) % 256, (np.floor(wy * 2).astype(np.int64)) % 256]
        col = np.empty((len(wx), 3), np.float32)
        col[:] = st["ground"]
        col += (n * st["ground_var"])[:, None]
        road = a < hw
        nr = self.tex[(np.floor(wx[road] * 6).astype(np.int64)) % 256, (np.floor(wy[road] * 6).astype(np.int64)) % 256]
        col[road] = np.asarray(st["road"], np.float32) + (nr * st["road_var"])[:, None]
        col[(a > hw - 0.35) & (a < hw - 0.15)] = st["edge"]
        if st["center_line"]:
            s = idx * tr.ds
            col[(a < 0.09) & ((s % 9.0) < 4.5)] = st["center"]
        shade = self.shadow[(np.floor(wx).astype(np.int64)) % 256, (np.floor(wy).astype(np.int64)) % 256]
        col *= (1 - st["shadows"] * shade)[:, None]
        e = np.exp(-cam.g_dist / st["fog_len"])[:, None]
        col = col * e + np.asarray(st["fog"], np.float32) * (1 - e)

        img = np.empty((cam.H * cam.W, 3), np.float32)
        img[cam.g_idx] = col
        # sky: vertical gradient + mountain silhouette that rotates with the heading
        el = cam.s_elev
        g = np.clip(el / 0.35, 0, 1)[:, None]
        sky = np.asarray(st["sky_bot"], np.float32) * (1 - g) + np.asarray(st["sky_top"], np.float32) * g
        phi = theta - cam.s_az
        mh = 0.012 + (self.m_amp[None] * np.sin(self.m_k[None] * phi[:, None] + self.m_phase[None])).sum(1)
        mount = el < mh
        mc = np.asarray(st["mountain"], np.float32) * 0.6 + np.asarray(st["fog"], np.float32) * 0.4
        sky[mount] = mc
        img[cam.s_idx] = sky
        img = img.reshape(cam.H, cam.W, 3)
        img[cam.hood] = (30, 30, 34)
        img *= st["sun"] * brightness
        img = cv2.GaussianBlur(img, (3, 3), 0.6)   # mild anti-aliasing
        return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- environment
class DriveLiteEnv:
    name = "drivelite"

    def __init__(self, style: int = 0, img_size=(160, 320), side_cameras=False, side_offset=0.8,
                 randomize_lighting=False, max_steps=2000, dt=0.1, substeps=4, wheelbase=2.6,
                 max_steer_deg=25.0, steer_rate=2.5, start_speed_mph=10.0):
        self.style_id = style
        self.style = STYLES[style]
        self.cam = Camera(W=img_size[1], H=img_size[0])
        self.side_cameras, self.side_offset = side_cameras, side_offset
        self.randomize_lighting = randomize_lighting
        self.max_steps, self.dt, self.substeps = max_steps, dt, substeps
        self.L, self.dmax, self.steer_rate = wheelbase, np.radians(max_steer_deg), steer_rate
        self.start_speed = start_speed_mph / MPS_TO_MPH
        self.track = None

    # -- helpers
    def _nearest(self, x, y):
        _, i = self.tree.query([x, y])
        return int(i)

    def _lat(self, i, x, y):
        c, t = self.track.xy[i], self.track.tangent[i]
        return float(t[0] * (y - c[1]) - t[1] * (x - c[0]))

    def _obs(self):
        img = self.renderer.render(self.x, self.y, self.th, 0.0, self.bright)
        obs = {"image": img, "speed": self.v * MPS_TO_MPH}
        if self.side_cameras:
            obs["left"] = self.renderer.render(self.x, self.y, self.th, self.side_offset, self.bright)
            obs["right"] = self.renderer.render(self.x, self.y, self.th, -self.side_offset, self.bright)
        return obs

    # -- API
    def reset(self, seed: int):
        self.seed = int(seed)
        self.track = make_track(self.seed)
        self.tree = cKDTree(self.track.xy)
        rng = np.random.default_rng(self.seed + 1)
        self.bright = rng.uniform(0.75, 1.25) if self.randomize_lighting else 1.0
        self.renderer = Renderer(self.track, self.style, self.cam, self.seed)
        self.x, self.y = self.track.xy[0]
        t = self.track.tangent[0]
        self.th = float(np.arctan2(t[1], t[0]))
        self.v, self.delta = self.start_speed, 0.0
        self.idx, self.progress, self.steps = 0, 0.0, 0
        return self._obs()

    def step(self, steer: float, throttle: float):
        cmd = -float(np.clip(steer, -1, 1)) * self.dmax       # + right -> negative (CCW) wheel angle
        thr = float(np.clip(throttle, -1, 1))
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            self.delta += float(np.clip(cmd - self.delta, -self.steer_rate * h, self.steer_rate * h))
            acc = (3.5 * thr if thr >= 0 else 7.0 * thr) - 0.01 * self.v * self.v - 0.1
            self.v = float(np.clip(self.v + acc * h, 0.0, 30.0))
            self.x += self.v * np.cos(self.th) * h
            self.y += self.v * np.sin(self.th) * h
            self.th += self.v / self.L * np.tan(self.delta) * h
        n = len(self.track.xy)
        i = self._nearest(self.x, self.y)
        d = ((i - self.idx + n // 2) % n) - n // 2
        self.progress += d * self.track.ds
        self.idx = i
        self.steps += 1
        cte = self._lat(i, self.x, self.y)
        completion = self.progress / self.track.length
        off_road = abs(cte) > self.track.width / 2 - 0.9
        wrong_way = self.progress < -10.0
        success = completion >= 1.0
        timeout = self.steps >= self.max_steps
        done = off_road or wrong_way or success or timeout
        info = {"cte": cte, "off_road": bool(off_road or wrong_way), "crash": False, "success": bool(success),
                "route_completion": float(np.clip(completion, 0, 1)), "timeout": bool(timeout and not success),
                "speed": self.v * MPS_TO_MPH}
        return self._obs(), done, info

    def expert_steer(self) -> float:
        """Pure pursuit on the centreline, returned in policy units (+ = right)."""
        n = len(self.track.xy)
        i = self._nearest(self.x, self.y)
        Ld = float(np.clip(3.0 + 0.45 * self.v, 4.0, 14.0))
        tgt = self.track.xy[(i + int(Ld / self.track.ds)) % n]
        dx, dy = tgt[0] - self.x, tgt[1] - self.y
        xv = dx * np.cos(self.th) + dy * np.sin(self.th)
        yv = -dx * np.sin(self.th) + dy * np.cos(self.th)       # + = left
        kappa = 2 * yv / max(xv * xv + yv * yv, 1e-6)
        delta = np.arctan(self.L * kappa)
        return float(np.clip(-delta / self.dmax, -1, 1))

    def topdown(self, size=400):
        """Debug/paper figure: track with car position."""
        xy = self.track.xy
        lo, hi = xy.min(0) - 10, xy.max(0) + 10
        sc = (size - 20) / (hi - lo).max()
        P = ((xy - lo) * sc + 10).astype(np.int32)
        P[:, 1] = size - P[:, 1]
        img = np.full((size, size, 3), 255, np.uint8)
        cv2.polylines(img, [P], True, (90, 90, 90), max(1, int(self.track.width * sc)))
        cp = ((np.array([self.x, self.y]) - lo) * sc + 10).astype(int)
        cv2.circle(img, (int(cp[0]), int(size - cp[1])), 4, (220, 40, 40), -1)
        return img

    def close(self):
        pass
