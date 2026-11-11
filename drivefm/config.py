"""Configuration: built-in defaults, YAML files, and `--set a.b=value` CLI overrides."""
from __future__ import annotations

import copy
from typing import Any, Iterable

import yaml

DEFAULTS_YAML = r"""
name: pilotnet_mse
seed: 0
out_dir: null              # default: runs/<name>_s<seed>

data:
  train_logs: [data/track1/driving_log.csv]   # one or more Udacity driving_log.csv files
  test_logs: []                               # held-out logs (e.g. Track 2) -- never used for training
  val_fraction: 0.2
  split_mode: block           # block (leakage-free, default) | random (per-frame split, as in the 2020 report)
  val_block_seconds: 20.0     # split by contiguous time blocks + purge (no near-duplicate leakage)
  max_gap_s: 0.5              # time gap that starts a new recording segment
  use_side_cameras: true
  side_correction: 0.2
  side_decay_tau: 6.0         # recovery correction decays over the action chunk (frames)
  balance_bins: 25
  balance_max_per_bin: 400    # 0 disables (original report used 25 bins / 400 cap)
  context_frames: 1           # K past frames given to the policy
  frame_stride: 1
  chunk: 1                    # H future steering values predicted (action chunking)
  crop_top: 60
  crop_bottom: 25
  num_workers: 4

augment:
  enabled: true
  p_flip: 0.5
  p_brightness: 0.5
  brightness_range: [0.3, 1.3]
  p_shadow: 0.3
  p_pan: 0.5
  pan_px: 40.0
  pan_steer_per_px: 0.004
  p_zoom: 0.3
  zoom_range: [1.0, 1.3]
  p_blur: 0.1

model:
  backbone: pilotnet          # pilotnet | timm:<model_name>  e.g. timm:vit_small_patch14_dinov2.lvd142m
  pretrained: true
  freeze_backbone: false
  unfreeze_last_n: 0          # with freeze_backbone, re-enable the last N blocks/stages
  img_size: [66, 200]         # H, W after crop + resize
  color: yuv                  # yuv (PilotNet) | rgb (pretrained backbones)
  pool: flatten               # flatten (PilotNet) | proj_flatten (spatial bottleneck) | attn (learned queries)
  pool_dim: 32                # channels per token kept by proj_flatten
  pool_queries: 4
  d_model: 100
  use_speed: false            # speed input can cause causal confusion / inertia (Codevilla et al. 2019)
  temporal_layers: 0
  heads: 4
  dropout: 0.0
  head: mse                   # mse | mdn | flow
  head_hidden: [50, 10]
  head_act: elu
  mdn_components: 5
  mdn_decode: mode            # mode | mean
  flow_hidden: 256
  flow_layers: 4
  flow_steps: 10
  flow_solver: euler          # euler | heun
  flow_t_sampling: logit_normal
  flow_decode: mode_mean      # mode_mean (mean-shift around densest mode) | kde_mode | mean | median
  infer_samples: 16           # flow samples / MC-dropout passes at evaluation time

train:
  epochs: 25
  batch_size: 64
  lr: 3.0e-4
  backbone_lr_mult: 0.1
  weight_decay: 1.0e-4
  warmup_epochs: 1
  ema_decay: 0.999
  grad_clip: 1.0
  amp: true
  steps_per_epoch: 0          # 0 = one pass over the (balanced) training set
  val_every: 1
  patience: 8
  select_metric: mae_t0
  init_ckpt: null             # warm start (e.g. DAgger iterations)
  device: auto
"""

DEFAULTS: dict = yaml.safe_load(DEFAULTS_YAML)


def _coerce(default: Any, v: Any) -> Any:
    """YAML reads '5e-4' as a string and '1' as an int; follow the type of the default."""
    if isinstance(default, float) and isinstance(v, (str, int)) and not isinstance(v, bool):
        return float(v)
    return v


def deep_update(base: dict, upd: dict, path: str = "") -> dict:
    for k, v in upd.items():
        key = f"{path}{k}"
        if k not in base:
            raise KeyError(f"Unknown config key '{key}'")
        if isinstance(base[k], dict) and isinstance(v, dict):
            deep_update(base[k], v, key + ".")
        else:
            base[k] = _coerce(base[k], v)
    return base


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node:
            raise KeyError(f"Unknown config key '{dotted}'")
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"Unknown config key '{dotted}'")
    node[parts[-1]] = _coerce(node[parts[-1]], value)


def load_config(path=None, overrides: Iterable[str] = ()) -> dict:
    """path: None, one YAML file, or a list of YAML files merged left to right (model + simulator preset)."""
    cfg = copy.deepcopy(DEFAULTS)
    paths = [path] if isinstance(path, str) else list(path or [])
    for p in paths:
        with open(p) as f:
            deep_update(cfg, yaml.safe_load(f) or {})
    for ov in overrides or ():
        if "=" not in ov:
            raise ValueError(f"Override must look like key=value, got '{ov}'")
        k, v = ov.split("=", 1)
        set_by_path(cfg, k.strip(), yaml.safe_load(v))
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    m, d = cfg["model"], cfg["data"]
    assert m["head"] in {"mse", "mdn", "flow"}, m["head"]
    assert m["pool"] in {"flatten", "proj_flatten", "attn"}, m["pool"]
    assert m["color"] in {"yuv", "rgb"}, m["color"]
    assert d["split_mode"] in {"block", "random"}, d["split_mode"]
    assert d["context_frames"] >= 1 and d["chunk"] >= 1
    if d["context_frames"] > 1 and m["temporal_layers"] == 0 and m["pool"] == "flatten":
        pass  # frames are concatenated + projected, allowed
    if m["backbone"] != "pilotnet" and not m["backbone"].startswith("timm:"):
        raise ValueError("model.backbone must be 'pilotnet' or 'timm:<name>'")
