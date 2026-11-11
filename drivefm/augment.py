"""Image augmentation for behavioural cloning.

All parameters are sampled ONCE per training sample and applied identically to every
context frame, so temporal models see a consistent clip. Returns the steering correction
implied by geometric augmentations (pan) and whether the clip was mirrored.
"""
from __future__ import annotations

import cv2
import numpy as np


def _brightness(img: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _shadow_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    x_top, x_bot = rng.uniform(0, w, size=2)
    ys, xs = np.mgrid[0:h, 0:w]
    boundary = x_top + (x_bot - x_top) * ys / max(h - 1, 1)
    left = xs < boundary
    return left if rng.random() < 0.5 else ~left


def augment_clip(imgs: list[np.ndarray], acfg: dict, rng: np.random.Generator):
    """imgs: list of HxWx3 uint8 RGB frames (same camera, oldest -> newest)."""
    h, w = imgs[0].shape[:2]
    pan_corr = 0.0

    scale = rng.uniform(*acfg["zoom_range"]) if rng.random() < acfg["p_zoom"] else 1.0
    dx = dy = 0.0
    if rng.random() < acfg["p_pan"]:
        dx = rng.uniform(-acfg["pan_px"], acfg["pan_px"])
        dy = rng.uniform(-acfg["pan_px"] / 4, acfg["pan_px"] / 4)
        # Road shifted right on screen => car is left of the lane centre => steer right (+).
        pan_corr = dx * acfg["pan_steer_per_px"]
    M = None
    if scale != 1.0 or dx or dy:
        M = np.float32([[scale, 0, (1 - scale) * w / 2 + dx], [0, scale, (1 - scale) * h / 2 + dy]])

    bright = rng.uniform(*acfg["brightness_range"]) if rng.random() < acfg["p_brightness"] else None
    shadow = None
    if rng.random() < acfg["p_shadow"]:
        shadow = (_shadow_mask(h, w, rng), rng.uniform(0.4, 0.8))
    blur_k = int(rng.choice([3, 5])) if rng.random() < acfg["p_blur"] else 0
    flip = bool(rng.random() < acfg["p_flip"])

    out = []
    for img in imgs:
        if M is not None:
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if bright is not None:
            img = _brightness(img, bright)
        if shadow is not None:
            mask, f = shadow
            img = img.copy()
            img[mask] = (img[mask].astype(np.float32) * f).astype(np.uint8)
        if blur_k:
            img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
        if flip:
            img = np.ascontiguousarray(img[:, ::-1])
        out.append(img)
    return out, pan_corr, flip
