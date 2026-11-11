"""Action heads operating on normalised action chunks a in R^H.

  * MSEHead  -- deterministic regression (the PilotNet / original-project objective).
  * MDNHead  -- Gaussian mixture density network; decodes the most probable mode, so it
                does not average two valid manoeuvres into an invalid one.
  * FlowHead -- conditional flow matching (rectified flow) generative policy, the 2024-26
                successor of diffusion policies; sample spread gives aleatoric uncertainty.

Every head exposes  loss(ctx, a) -> scalar  and  predict(ctx, n) -> dict(action, std[, samples]).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTS = {"elu": nn.ELU, "gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


def mlp(dims, act="gelu", dropout=0.0, last_act=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers.append(ACTS[act]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- MSE
class MSEHead(nn.Module):
    def __init__(self, d, H, hidden, act="elu", dropout=0.0):
        super().__init__()
        self.net = mlp([d, *hidden, H], act, dropout)

    def loss(self, ctx, a):
        return F.mse_loss(self.net(ctx), a)

    def predict(self, ctx, n_samples=1):
        mu = self.net(ctx)
        return {"action": mu, "std": torch.zeros_like(mu)}


# --------------------------------------------------------------------------- MDN
class MDNHead(nn.Module):
    def __init__(self, d, H, hidden, M=5, act="gelu", dropout=0.0, decode="mode"):
        super().__init__()
        self.H, self.M, self.decode = H, M, decode
        self.trunk = mlp([d, *hidden], act, dropout, last_act=True) if hidden else nn.Identity()
        h = hidden[-1] if hidden else d
        self.pi = nn.Linear(h, M)
        self.mu = nn.Linear(h, M * H)
        self.log_sigma = nn.Linear(h, M * H)
        with torch.no_grad():  # spread initial component means to avoid mode collapse
            self.mu.bias.copy_(torch.linspace(-1.5, 1.5, M).repeat_interleave(H))
            self.log_sigma.bias.fill_(-0.5)

    def params(self, ctx):
        t = self.trunk(ctx)
        B = ctx.shape[0]
        logit = self.pi(t)
        mu = self.mu(t).view(B, self.M, self.H)
        ls = self.log_sigma(t).view(B, self.M, self.H).clamp(-7.0, 3.0)
        return logit, mu, ls

    def loss(self, ctx, a):
        logit, mu, ls = self.params(ctx)
        z = (a[:, None, :] - mu) * torch.exp(-ls)
        logp = (-0.5 * z.pow(2) - ls - 0.5 * math.log(2 * math.pi)).sum(-1)
        nll = -torch.logsumexp(F.log_softmax(logit, -1) + logp, dim=-1)
        return nll.mean() / self.H

    def predict(self, ctx, n_samples=1):
        logit, mu, ls = self.params(ctx)
        w = F.softmax(logit, -1)[..., None]
        mean = (w * mu).sum(1)
        var = (w * (torch.exp(2 * ls) + mu.pow(2))).sum(1) - mean.pow(2)
        if self.decode == "mode":
            k = w[..., 0].argmax(-1)
            action = mu[torch.arange(mu.shape[0], device=mu.device), k]
        else:
            action = mean
        return {"action": action, "std": var.clamp_min(0).sqrt(), "weights": w[..., 0], "means": mu}


# --------------------------------------------------------------------------- Flow matching
class SinusoidalEmb(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        ang = 1000.0 * t[:, None] * freqs[None]
        return torch.cat([ang.sin(), ang.cos()], -1)


class FiLMResBlock(nn.Module):
    def __init__(self, h, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(h, elementwise_affine=False)
        self.film = nn.Linear(h, 3 * h)
        self.ff = nn.Sequential(nn.Linear(h, 4 * h), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * h, h))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, cond):
        scale, shift, gate = self.film(cond).chunk(3, -1)
        y = self.norm(x) * (1 + scale) + shift
        return x + gate.tanh().add(1) * self.ff(y)  # adaLN-style gated residual (gate=1 at init)


class FlowHead(nn.Module):
    def __init__(self, d, H, hidden=512, layers=4, steps=10, solver="euler",
                 t_sampling="logit_normal", decode="mode_mean", dropout=0.0):
        super().__init__()
        self.H, self.steps, self.solver = H, steps, solver
        self.t_sampling, self.decode = t_sampling, decode
        self.x_in = nn.Linear(H, hidden)
        self.t_emb = nn.Sequential(SinusoidalEmb(128), nn.Linear(128, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.c_in = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList([FiLMResBlock(hidden, dropout) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, H)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def velocity(self, x, t, ctx):
        h = self.x_in(x)
        cond = self.t_emb(t) + self.c_in(ctx)
        for b in self.blocks:
            h = b(h, cond)
        return self.out(self.out_norm(h))

    def _sample_t(self, B, device):
        if self.t_sampling == "logit_normal":   # emphasises intermediate t (Esser et al. 2024)
            return torch.sigmoid(torch.randn(B, device=device))
        return torch.rand(B, device=device)

    def loss(self, ctx, a):
        x0 = torch.randn_like(a)
        t = self._sample_t(a.shape[0], a.device)
        xt = (1 - t[:, None]) * x0 + t[:, None] * a
        return F.mse_loss(self.velocity(xt, t, ctx), a - x0)

    @torch.no_grad()
    def sample(self, ctx, n):
        B = ctx.shape[0]
        c = ctx.repeat_interleave(n, 0)
        x = torch.randn(B * n, self.H, device=ctx.device, dtype=ctx.dtype)
        dt = 1.0 / self.steps
        for i in range(self.steps):
            t = torch.full((B * n,), i * dt, device=ctx.device, dtype=ctx.dtype)
            v = self.velocity(x, t, c)
            if self.solver == "heun" and i < self.steps - 1:
                x_e = x + dt * v
                v2 = self.velocity(x_e, t + dt, c)
                x = x + 0.5 * dt * (v + v2)
            else:
                x = x + dt * v
        return x.view(B, n, self.H)

    @staticmethod
    def kde_mode(s, average=True):
        """Mode-seeking decoding of samples s (B,n,H).

        Finds the sample with the highest kernel density using a *local* bandwidth (distance to the
        n/4-th nearest neighbour, so two well-separated modes never share a kernel). If `average`,
        returns the kernel-weighted mean of the samples around that mode (one mean-shift step):
        it keeps the multi-modal choice but removes per-sample noise / jitter.
        """
        B, n, _ = s.shape
        d2 = torch.cdist(s, s).pow(2)
        k = max(1, n // 4)
        h2 = d2.sort(-1).values[..., k].median(-1).values.clamp_min(1e-8)[:, None, None]
        dens = torch.exp(-d2 / h2).sum(-1)
        idx = dens.argmax(-1)
        ar = torch.arange(B, device=s.device)
        if not average:
            return s[ar, idx]
        w = torch.exp(-d2[ar, idx] / (4 * h2[:, 0]))          # (B,n)
        return (w[..., None] * s).sum(1) / w.sum(1, keepdim=True)

    def predict(self, ctx, n_samples=16):
        n = max(int(n_samples), 2)
        s = self.sample(ctx, n)
        if self.decode == "mode_mean":
            action = self.kde_mode(s, average=True)
        elif self.decode == "kde_mode":
            action = self.kde_mode(s, average=False)
        elif self.decode == "median":
            action = s.median(1).values
        else:
            action = s.mean(1)
        return {"action": action, "std": s.std(1), "samples": s}


def build_head(m: dict, d: int, H: int) -> nn.Module:
    if m["head"] == "mse":
        return MSEHead(d, H, m["head_hidden"], m["head_act"], m["dropout"])
    if m["head"] == "mdn":
        return MDNHead(d, H, m["head_hidden"], m["mdn_components"], m["head_act"], m["dropout"], m["mdn_decode"])
    return FlowHead(d, H, m["flow_hidden"], m["flow_layers"], m["flow_steps"], m["flow_solver"],
                    m["flow_t_sampling"], m["flow_decode"], m["dropout"])
