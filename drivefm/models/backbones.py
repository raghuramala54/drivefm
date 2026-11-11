"""Visual backbones. Every backbone maps (B,3,H,W) in [0,1] to tokens (B,N,C)."""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PilotNetBackbone(nn.Module):
    """Conv stack of Bojarski et al. 2016 (End to End Learning for Self-Driving Cars)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2), nn.ELU(),
            nn.Conv2d(24, 36, 5, 2), nn.ELU(),
            nn.Conv2d(36, 48, 5, 2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU(),
        )
        self.out_dim = 64
        self.register_buffer("mean", torch.full((1, 3, 1, 1), 0.5))
        self.register_buffer("std", torch.full((1, 3, 1, 1), 0.5))

    def forward(self, x):
        f = self.net((x - self.mean) / self.std)
        return f.flatten(2).transpose(1, 2)

    def trainable_groups(self):
        return [self.net]


class TimmBackbone(nn.Module):
    """Any timm model (DINOv2 / CLIP / SigLIP ViTs, ResNets, ConvNeXts, EfficientNets...)."""

    def __init__(self, name: str, pretrained: bool, img_size):
        super().__init__()
        import timm

        kw = dict(pretrained=pretrained, num_classes=0)
        try:  # ViT-style models accept arbitrary input sizes this way
            self.m = timm.create_model(name, img_size=tuple(img_size), dynamic_img_size=True, **kw)
        except TypeError:
            self.m = timm.create_model(name, **kw)
        pc = getattr(self.m, "pretrained_cfg", {}) or {}
        mean, std = pc.get("mean", IMAGENET_MEAN), pc.get("std", IMAGENET_STD)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.out_dim = self.m.num_features

    def forward(self, x):
        f = self.m.forward_features((x - self.mean) / self.std)
        if f.ndim == 4:
            if f.shape[1] == self.out_dim:      # NCHW
                f = f.flatten(2).transpose(1, 2)
            else:                               # NHWC (e.g. Swin)
                f = f.flatten(1, 2)
        return f

    def trainable_groups(self):
        m = self.m
        if hasattr(m, "blocks"):
            return list(m.blocks)
        if hasattr(m, "stages"):
            return list(m.stages)
        layers = [getattr(m, f"layer{i}") for i in range(1, 5) if hasattr(m, f"layer{i}")]
        if layers:
            return layers
        warnings.warn("Could not find blocks/stages for partial unfreezing; using whole model")
        return [m]


def build_backbone(mcfg: dict) -> nn.Module:
    name = mcfg["backbone"]
    if name == "pilotnet":
        bb = PilotNetBackbone()
    else:
        bb = TimmBackbone(name.split(":", 1)[1], mcfg["pretrained"], mcfg["img_size"])
    if mcfg["freeze_backbone"]:
        for p in bb.parameters():
            p.requires_grad_(False)
        n = int(mcfg["unfreeze_last_n"])
        if n > 0:
            groups = bb.trainable_groups()[-n:]
            # final norms after the unfrozen blocks are always trained too
            groups += [getattr(getattr(bb, "m", bb), a) for a in ("norm", "fc_norm", "head_norm")
                       if isinstance(getattr(getattr(bb, "m", bb), a, None), nn.Module)]
            for g in groups:
                for p in g.parameters():
                    p.requires_grad_(True)
    return bb
