from __future__ import annotations

import copy
import csv
import math
import os
import random

import numpy as np
import torch
import yaml


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EMA:
    """Exponential moving average of weights (with warm-up), used for evaluation & export."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.steps = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k]
            if v.dtype.is_floating_point:
                v.lerp_(src.detach(), 1 - d)
            else:
                v.copy_(src)


def cosine_with_warmup(total_steps: int, warmup_steps: int):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return f


def param_groups(model, lr, backbone_lr_mult, wd):
    groups = {(True, True): [], (True, False): [], (False, True): [], (False, False): []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bb = n.startswith("backbone.")
        no_decay = p.ndim <= 1 or n.endswith(("_pos", "queries", "cls"))
        groups[(is_bb, not no_decay)].append(p)
    out = []
    for (is_bb, decay), ps in groups.items():
        if ps:
            out.append({"params": ps, "lr": lr * (backbone_lr_mult if is_bb else 1.0),
                        "weight_decay": wd if decay else 0.0})
    return out


def save_checkpoint(path, model, cfg, extra=None):
    torch.save({"cfg": cfg, "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_policy(path, device="cpu"):
    from .models import Policy

    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(ck["cfg"])
    cfg["model"]["pretrained"] = False  # weights come from the checkpoint
    model = Policy(cfg)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    bad = [k for k in missing if not k.startswith("ood_")] + list(unexpected)
    if bad:
        raise RuntimeError(f"checkpoint/architecture mismatch: {bad[:5]}")
    model.extra = ck.get("extra", {})
    return model.to(device).eval(), ck["cfg"]


def load_policies(paths, device="cpu"):
    from .models import EnsemblePolicy

    pols = [load_policy(p, device) for p in paths]
    if len(pols) == 1:
        return pols[0]
    ens = EnsemblePolicy([p for p, _ in pols]).to(device).eval()
    ens.extra = pols[0][0].extra
    return ens, pols[0][1]


class CSVLogger:
    def __init__(self, path):
        self.path, self.fields = path, None

    def log(self, row: dict):
        new = not os.path.exists(self.path)
        if self.fields is None:
            self.fields = list(row.keys())
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def dump_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
