"""Udacity simulator data: log parsing, leakage-free splits, balancing, sequence datasets.

Improvements over the usual random 80/20 split:
  * frames are grouped into recording segments using the timestamps in the image names;
  * the validation set is made of contiguous time blocks, and any training sample whose
    context/future window touches a validation block is purged (consecutive frames are
    near duplicates, so a random split leaks and makes validation loss meaninglessly low);
  * samples carry K past frames and H future steering values (action chunks).
"""
from __future__ import annotations

import os
import re
import warnings
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augment import augment_clip

COLS = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
CAMS = ("center", "left", "right")
TS_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{3})")
SPEED_NORM = 30.0  # simulator max speed (mph)


# --------------------------------------------------------------------------- parsing
def _parse_ts(path: str):
    m = TS_RE.search(os.path.basename(path))
    if not m:
        return np.nan
    y, mo, d, h, mi, s, ms = map(int, m.groups())
    return datetime(y, mo, d, h, mi, s, ms * 1000).timestamp()


def _resolve(p: str, root: str) -> str:
    p = str(p).strip()
    base = re.split(r"[\\/]", p)[-1]
    for cand in (os.path.join(root, "IMG", base), os.path.join(root, p), p):
        if os.path.exists(cand):
            return cand
    return os.path.join(root, "IMG", base)


def load_logs(paths: list[str], max_gap_s: float = 0.5) -> pd.DataFrame:
    frames = []
    for li, p in enumerate(paths):
        df = pd.read_csv(p, header=None, skipinitialspace=True)
        if str(df.iloc[0, 0]).strip().lower() == "center":
            df = df.iloc[1:].reset_index(drop=True)
        df = df.iloc[:, :7].copy()
        df.columns = COLS
        for c in ("steering", "throttle", "brake", "speed"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)
        root = os.path.dirname(os.path.abspath(p))
        for cam in CAMS:
            df[cam] = [_resolve(x, root) for x in df[cam].values]
        t = np.array([_parse_ts(x) for x in df["center"].values], dtype=np.float64)
        if np.isnan(t).any():
            warnings.warn(f"{p}: could not parse timestamps from image names; assuming 15 Hz")
            t = np.arange(len(df)) / 15.0
        df["t"] = t
        df["log_id"] = li
        df["log_name"] = os.path.basename(os.path.dirname(os.path.abspath(p)))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    missing = [x for x in df["center"].values[:: max(1, len(df) // 20)] if not os.path.exists(x)]
    if missing:
        raise FileNotFoundError(f"Image files not found, e.g. {missing[0]}. Keep IMG/ next to driving_log.csv.")
    dt = np.diff(df["t"].values, prepend=df["t"].values[0])
    new_seg = (np.diff(df["log_id"].values, prepend=df["log_id"].values[0]) != 0) | (dt < 0) | (dt > max_gap_s)
    new_seg[0] = False
    df["seg"] = np.cumsum(new_seg)
    return df


# --------------------------------------------------------------------------- indexing / splits
def valid_anchors(df: pd.DataFrame, K: int, stride: int, H: int) -> np.ndarray:
    n = len(df)
    seg = df["seg"].values
    idx = np.arange(n)
    first, last = idx - (K - 1) * stride, idx + H - 1
    ok = (first >= 0) & (last < n)
    f, l = np.clip(first, 0, n - 1), np.clip(last, 0, n - 1)
    ok &= (seg[f] == seg) & (seg[l] == seg)
    return idx[ok]


def block_split(df: pd.DataFrame, val_fraction: float, block_s: float, seed: int) -> np.ndarray:
    """Returns a boolean array: True where the frame belongs to a validation time block."""
    t0 = df.groupby("seg")["t"].transform("min").values
    block = np.floor((df["t"].values - t0) / block_s).astype(np.int64)
    keys = df["seg"].values.astype(np.int64) * 1_000_000 + block
    uniq = np.unique(keys)
    rng = np.random.default_rng(seed)
    n_val = int(round(val_fraction * len(uniq)))
    if len(uniq) > 1:
        n_val = min(max(n_val, 1), len(uniq) - 1)
    val_keys = set(rng.permutation(uniq)[:n_val].tolist())
    return np.array([k in val_keys for k in keys])


def purge(anchors: np.ndarray, is_val: np.ndarray, K: int, stride: int, H: int, want_val: bool) -> np.ndarray:
    cs = np.concatenate([[0], np.cumsum(is_val.astype(np.int64))])
    a, b = anchors - (K - 1) * stride, anchors + H - 1
    n_val = cs[b + 1] - cs[a]
    length = b - a + 1
    keep = (n_val == length) if want_val else (n_val == 0)
    return anchors[keep]


def balance(anchors: np.ndarray, steer: np.ndarray, bins: int, max_per_bin: int, seed: int) -> np.ndarray:
    if max_per_bin <= 0 or len(anchors) == 0:
        return anchors
    s = steer[anchors]
    edges = np.histogram_bin_edges(s, bins=bins)
    which = np.clip(np.digitize(s, edges[1:-1]), 0, bins - 1)
    rng = np.random.default_rng(seed)
    keep = []
    for b in range(bins):
        members = anchors[which == b]
        if len(members) > max_per_bin:
            members = rng.choice(members, max_per_bin, replace=False)
        keep.append(members)
    return np.sort(np.concatenate(keep))


# --------------------------------------------------------------------------- preprocessing
def preprocess(img_rgb: np.ndarray, cfg: dict) -> np.ndarray:
    """RGB uint8 HxWx3 (raw 320x160 sim frame) -> float32 3xhxw in [0,1] (YUV or RGB)."""
    d, m = cfg["data"], cfg["model"]
    h = img_rgb.shape[0]
    img = img_rgb[d["crop_top"]: h - d["crop_bottom"]]
    H, W = m["img_size"]
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    if m["color"] == "yuv":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return img.transpose(2, 0, 1).astype(np.float32) / 255.0


def load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- dataset
class DriveDataset(Dataset):
    def __init__(self, df: pd.DataFrame, anchors: np.ndarray, cfg: dict, train: bool, seed: int = 0):
        self.cfg, self.train = cfg, train
        d = cfg["data"]
        self.K, self.stride, self.H = d["context_frames"], d["frame_stride"], d["chunk"]
        self.paths = {c: df[c].values for c in CAMS}
        self.steer = df["steering"].values.astype(np.float32)
        self.speed = df["speed"].values.astype(np.float32)
        self.anchors = np.asarray(anchors, dtype=np.int64)
        self.decay = np.exp(-np.arange(self.H) / max(d["side_decay_tau"], 1e-6)).astype(np.float32)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, i):
        d, a = self.cfg["data"], int(self.anchors[i])
        ctx = a - self.stride * np.arange(self.K - 1, -1, -1)
        cam, corr = "center", 0.0
        if self.train and d["use_side_cameras"]:
            r = self.rng.random()
            if r < 1 / 3:
                cam, corr = "left", d["side_correction"]
            elif r < 2 / 3:
                cam, corr = "right", -d["side_correction"]
        imgs = [load_rgb(self.paths[cam][j]) for j in ctx]
        chunk = self.steer[a: a + self.H].copy()
        flip = False
        if self.train and self.cfg["augment"]["enabled"]:
            imgs, pan_corr, flip = augment_clip(imgs, self.cfg["augment"], self.rng)
            corr += pan_corr
        chunk = chunk + corr * self.decay
        if flip:
            chunk = -chunk
        chunk = np.clip(chunk, -1.0, 1.0).astype(np.float32)
        x = np.stack([preprocess(im, self.cfg) for im in imgs])
        return {
            "images": torch.from_numpy(x),
            "action": torch.from_numpy(chunk),
            "speed": torch.tensor(self.speed[a] / SPEED_NORM, dtype=torch.float32),
            "index": torch.tensor(a, dtype=torch.long),
        }


def worker_init_fn(_):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def build_datasets(cfg: dict, seed: int):
    d = cfg["data"]
    K, s, H = d["context_frames"], d["frame_stride"], d["chunk"]
    df = load_logs(d["train_logs"], d["max_gap_s"])
    anchors = valid_anchors(df, K, s, H)
    if d["split_mode"] == "random":   # leaky per-frame split (kept only to quantify the leakage)
        perm = np.random.default_rng(seed).permutation(anchors)
        n_val = int(round(d["val_fraction"] * len(perm)))
        tr, va = np.sort(perm[n_val:]), np.sort(perm[:n_val])
    else:
        is_val = block_split(df, d["val_fraction"], d["val_block_seconds"], seed)
        tr = purge(anchors, is_val, K, s, H, want_val=False)
        va = purge(anchors, is_val, K, s, H, want_val=True)
    tr_bal = balance(tr, df["steering"].values, d["balance_bins"], d["balance_max_per_bin"], seed)
    stats = {
        "frames": len(df), "segments": int(df["seg"].nunique()),
        "train_anchors_raw": len(tr), "train_anchors_balanced": len(tr_bal), "val_anchors": len(va),
    }
    if len(tr_bal) == 0 or len(va) == 0:
        raise RuntimeError(f"Empty split: {stats}. Record more data or lower val_block_seconds.")
    train_ds = DriveDataset(df, tr_bal, cfg, train=True, seed=seed)
    val_ds = DriveDataset(df, va, cfg, train=False)
    tests = {}
    for p in d["test_logs"]:
        tdf = load_logs([p], d["max_gap_s"])
        tests[os.path.basename(os.path.dirname(os.path.abspath(p)))] = DriveDataset(
            tdf, valid_anchors(tdf, K, s, H), cfg, train=False)
    # robust action scale for normalising targets of the generative heads
    st = df["steering"].values[tr_bal]
    scale = float(max(np.std(np.concatenate([st, -st])), 0.05))
    return train_ds, val_ds, tests, stats, scale


def build_eval_dataset(cfg: dict, log_path: str) -> DriveDataset:
    d = cfg["data"]
    df = load_logs([log_path], d["max_gap_s"])
    return DriveDataset(df, valid_anchors(df, d["context_frames"], d["frame_stride"], d["chunk"]), cfg, train=False)
