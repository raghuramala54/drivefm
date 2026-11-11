"""Policy = backbone -> per-frame pooling -> temporal fusion -> action head."""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbones import build_backbone
from .heads import build_head


class Policy(nn.Module):
    def __init__(self, cfg: dict, action_scale: float = 1.0):
        super().__init__()
        m, d_cfg = cfg["model"], cfg["data"]
        self.cfg = cfg
        self.K, self.H = d_cfg["context_frames"], d_cfg["chunk"]
        self.backbone = build_backbone(m)
        self.backbone_frozen = not any(p.requires_grad for p in self.backbone.parameters())
        H_img, W_img = m["img_size"]
        with torch.no_grad():
            was = self.backbone.training
            self.backbone.eval()
            tok = self.backbone(torch.zeros(1, 3, H_img, W_img))
            self.backbone.train(was)
        n_tok, C = tok.shape[1], tok.shape[2]
        d = m["d_model"]
        self.pool = m["pool"]
        if self.pool == "flatten":
            self.frame_proj = nn.Sequential(nn.Flatten(), nn.Dropout(m["dropout"]), nn.Linear(n_tok * C, d), nn.ELU())
        elif self.pool == "proj_flatten":
            # per-token channel bottleneck, then flatten: keeps WHERE features are (road position -> steering)
            pd_ = m["pool_dim"]
            self.frame_proj = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, pd_), nn.GELU(), nn.Flatten(),
                                            nn.Dropout(m["dropout"]), nn.Linear(n_tok * pd_, d), nn.GELU())
        else:
            nq = m["pool_queries"]
            self.tok_proj = nn.Linear(C, d)
            self.tok_pos = nn.Parameter(torch.zeros(1, n_tok, d))
            self.queries = nn.Parameter(torch.randn(1, nq, d) * 0.02)
            self.pool_attn = nn.MultiheadAttention(d, m["heads"], dropout=m["dropout"], batch_first=True)
            self.pool_norm = nn.LayerNorm(d)
            self.pool_out = nn.Sequential(nn.Linear(nq * d, d), nn.GELU())
            nn.init.trunc_normal_(self.tok_pos, std=0.02)

        self.use_speed = m["use_speed"]
        if self.use_speed:
            self.speed_emb = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.temporal_layers = m["temporal_layers"]
        n_extra = 1 + int(self.use_speed)  # CLS (+ speed)
        if self.temporal_layers > 0:
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.time_pos = nn.Parameter(torch.zeros(1, self.K + n_extra, d))
            nn.init.trunc_normal_(self.time_pos, std=0.02)
            layer = nn.TransformerEncoderLayer(d, m["heads"], 4 * d, m["dropout"], activation="gelu",
                                               batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(layer, self.temporal_layers, enable_nested_tensor=False)
            self.ctx_norm = nn.LayerNorm(d)
        else:
            in_dim = self.K * d + (d if self.use_speed else 0)
            self.fuse = nn.Identity() if in_dim == d else nn.Sequential(nn.Linear(in_dim, d), nn.GELU())
        self.head = build_head(m, d, self.H)
        self.register_buffer("action_scale", torch.tensor(float(action_scale)))
        # feature-space novelty (epistemic proxy): Mahalanobis distance of the context embedding to the
        # training distribution (Lee et al., NeurIPS 2018). Fitted after training by fit_feature_density().
        self.register_buffer("ood_fitted", torch.tensor(0.0))
        self.register_buffer("ood_mean", torch.zeros(d))
        self.register_buffer("ood_prec", torch.eye(d))

    # ------------------------------------------------------------------ encoding
    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()  # keep frozen BN/dropout statistics fixed
        return self

    def _frame_embed(self, x):  # (N,3,H,W) -> (N,d)
        if self.backbone_frozen:
            with torch.no_grad():
                tok = self.backbone(x)
        else:
            tok = self.backbone(x)
        if self.pool in ("flatten", "proj_flatten"):
            return self.frame_proj(tok)
        t = self.tok_proj(tok) + self.tok_pos
        q = self.queries.expand(t.shape[0], -1, -1)
        o, _ = self.pool_attn(q, t, t, need_weights=False)
        return self.pool_out(self.pool_norm(o).flatten(1))

    def encode(self, images, speed=None):
        B, K = images.shape[:2]
        f = self._frame_embed(images.flatten(0, 1)).view(B, K, -1)
        if self.temporal_layers > 0:
            toks = [self.cls.expand(B, -1, -1)]
            if self.use_speed:
                toks.append(self.speed_emb(speed.view(B, 1, 1).to(f.dtype)))
            toks.append(f)
            z = torch.cat(toks, 1) + self.time_pos
            return self.ctx_norm(self.temporal(z)[:, 0])
        parts = [f.flatten(1)]
        if self.use_speed:
            parts.append(self.speed_emb(speed.view(B, 1).to(f.dtype)))
        return self.fuse(torch.cat(parts, -1))

    # ------------------------------------------------------------------ API
    def loss(self, batch):
        ctx = self.encode(batch["images"], batch.get("speed"))
        return self.head.loss(ctx, batch["action"] / self.action_scale)

    def novelty(self, ctx):
        diff = ctx.float() - self.ood_mean
        return ((diff @ self.ood_prec) * diff).sum(-1).clamp_min(0).sqrt() / (ctx.shape[-1] ** 0.5)

    @torch.no_grad()
    def predict(self, images, speed=None, n_samples=16, mc_dropout=False):
        ctx = self.encode(images, speed)
        if mc_dropout and n_samples > 1:   # MC-dropout for deterministic heads
            drop = [mod for mod in self.modules() if isinstance(mod, nn.Dropout)]
            for mod in drop:
                mod.train()
            outs = [self.head.predict(self.encode(images, speed), 1)["action"] for _ in range(n_samples)]
            for mod in drop:
                mod.eval()
            s = torch.stack(outs, 1)
            out = {"action": s.mean(1), "std": s.std(1)}
        else:
            out = self.head.predict(ctx, n_samples)
        sc = self.action_scale
        res = {"action": out["action"] * sc, "std": out["std"] * sc}
        if "samples" in out:
            res["samples"] = out["samples"] * sc
        if self.ood_fitted > 0:
            res["ood"] = self.novelty(ctx)
        return res


class EnsemblePolicy(nn.Module):
    """Deep ensemble: epistemic = disagreement between members, aleatoric = mean member std."""

    def __init__(self, members: list[Policy]):
        super().__init__()
        self.members = nn.ModuleList(members)
        c0 = members[0].cfg
        for mbr in members[1:]:
            assert mbr.cfg["model"]["img_size"] == c0["model"]["img_size"]
            assert mbr.cfg["model"]["color"] == c0["model"]["color"]
            assert mbr.K == members[0].K and mbr.H == members[0].H
        self.cfg, self.K, self.H = c0, members[0].K, members[0].H

    @torch.no_grad()
    def predict(self, images, speed=None, n_samples=16, mc_dropout=False):
        outs = [mbr.predict(images, speed, n_samples, mc_dropout) for mbr in self.members]
        a = torch.stack([o["action"] for o in outs], 1)
        s = torch.stack([o["std"] for o in outs], 1)
        epi = a.std(1) if len(outs) > 1 else torch.zeros_like(a[:, 0])
        alea = s.pow(2).mean(1).sqrt()
        res = {"action": a.mean(1), "std": (epi.pow(2) + alea.pow(2)).sqrt(),
               "std_epistemic": epi, "std_aleatoric": alea}
        if all("ood" in o for o in outs):
            res["ood"] = torch.stack([o["ood"] for o in outs], 1).mean(1)
        return res


@torch.no_grad()
def fit_feature_density(policy: Policy, loader, device, max_samples: int = 20000):
    """Fit mean + Ledoit-Wolf-shrunk precision of context embeddings on (un-augmented) training data."""
    from sklearn.covariance import LedoitWolf

    policy.eval()
    feats, n = [], 0
    for b in loader:
        feats.append(policy.encode(b["images"].to(device), b["speed"].to(device)).float().cpu())
        n += len(feats[-1])
        if n >= max_samples:
            break
    X = torch.cat(feats).numpy()
    lw = LedoitWolf().fit(X)
    policy.ood_mean.copy_(torch.as_tensor(lw.location_, dtype=torch.float32))
    policy.ood_prec.copy_(torch.as_tensor(lw.precision_, dtype=torch.float32))
    policy.ood_fitted.fill_(1.0)
    return len(X)


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train
