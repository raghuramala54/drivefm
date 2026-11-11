"""End-to-end smoke tests on a tiny synthetic Udacity-format dataset (CPU, ~1-2 min)."""
import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import evaluate as eval_script  # noqa: E402
import make_synthetic_data  # noqa: E402
import train as train_script  # noqa: E402

from drivefm.agent import DrivingAgent  # noqa: E402
from drivefm.config import load_config  # noqa: E402
from drivefm.data import build_datasets, load_logs  # noqa: E402
from drivefm.models import Policy  # noqa: E402
from drivefm.utils import load_policies  # noqa: E402


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    tr, te = str(root / "track1"), str(root / "track2")
    make_synthetic_data.main(["--out", tr, "--frames", "160", "--sessions", "2"])
    make_synthetic_data.main(["--out", te, "--frames", "60", "--sessions", "1", "--style", "2", "--seed", "1"])
    return os.path.join(tr, "driving_log.csv"), os.path.join(te, "driving_log.csv")


def common(data, extra=()):
    tr, te = data
    return [f"data.train_logs=[{tr}]", f"data.test_logs=[{te}]", "data.num_workers=0", "data.val_block_seconds=2",
            "data.balance_max_per_bin=0", "train.epochs=2", "train.batch_size=8", "train.steps_per_epoch=3",
            "model.infer_samples=4", *extra]


def test_log_parsing_and_split(data):
    df = load_logs([data[0]])
    assert df["seg"].nunique() == 2                      # 60 s gap splits sessions
    cfg = load_config(None,
                      common(data, ["data.context_frames=3", "data.chunk=4", "data.frame_stride=2"]))
    tr, va, tests, stats, scale = build_datasets(cfg, 0)
    assert set(tr.anchors).isdisjoint(set(va.anchors))
    b = tr[0]
    assert b["images"].shape == (3, 3, 66, 200) and b["action"].shape == (4,)


@pytest.mark.parametrize("name,extra", [
    ("pilotnet_mse", []),
    ("pilotnet_mdn_chunk", ["model.head=mdn", "data.chunk=4", "model.head_hidden=[32]"]),
    ("pilotnet_flow_temporal", ["model.head=flow", "data.chunk=4", "data.context_frames=3", "model.pool=attn",
                                "model.d_model=64", "model.temporal_layers=1", "model.flow_hidden=64",
                                "model.flow_layers=2", "model.use_speed=true"]),
    ("resnet_frozen_flow", ["model.backbone=timm:resnet18", "model.pretrained=false", "model.freeze_backbone=true",
                            "model.unfreeze_last_n=1", "model.img_size=[64,192]", "model.color=rgb",
                            "model.pool=proj_flatten", "model.d_model=64", "model.head=flow", "model.flow_hidden=64",
                            "model.flow_layers=2", "data.chunk=4"]),
    ("vit_mse_random_split", ["model.backbone=timm:vit_tiny_patch16_224", "model.pretrained=false",
                              "model.img_size=[64,192]", "model.color=rgb", "model.pool=attn", "model.d_model=64",
                              "model.head_hidden=[64]", "data.split_mode=random"]),
])
def test_train_eval(data, tmp_path, name, extra):
    run = str(tmp_path / name)
    train_script.main(["--out", run, "--set", *common(data, extra), f"name={name}"])
    m = json.load(open(os.path.join(run, "metrics.json")))
    assert np.isfinite(m["val"]["mae_t0"]) and "track2" in m["test"]
    eval_script.main(["--ckpt", os.path.join(run, "best.pt"), "--device", "cpu", "--samples", "4"])
    # deep ensemble of the same checkpoint twice + the closed-loop agent
    pol, cfg = load_policies([os.path.join(run, "best.pt")] * 2, "cpu")
    agent = DrivingAgent(pol, cfg, "cpu", n_samples=4)
    img = np.random.randint(0, 255, (160, 320, 3), np.uint8)
    for _ in range(5):
        out = agent.act(img, speed=15.0)
    assert -1 <= out["steer"] <= 1 and -1 <= out["throttle"] <= 1


def test_heads_learn_bimodal_target():
    """Flow & MDN heads should represent a 50/50 left/right target that MSE averages to ~0."""
    torch.manual_seed(0)
    from drivefm.models.heads import FlowHead, MDNHead, MSEHead
    ctx = torch.zeros(512, 8)
    a = torch.where(torch.rand(512, 1) < 0.5, -1.0, 1.0)
    heads = {"mse": MSEHead(8, 1, [32], "gelu"), "mdn": MDNHead(8, 1, [32], 2),
             "flow": FlowHead(8, 1, 64, 2, steps=20)}
    for h in heads.values():
        opt = torch.optim.Adam(h.parameters(), 3e-3)
        for _ in range(400):
            opt.zero_grad()
            h.loss(ctx, a).backward()
            opt.step()
    with torch.no_grad():
        assert heads["mse"].predict(ctx[:1])["action"].abs().item() < 0.3
        assert heads["mdn"].predict(ctx[:1])["action"].abs().item() > 0.7
        s = heads["flow"].sample(ctx[:1], 256)[0, :, 0]
        assert (s.abs() > 0.5).float().mean() > 0.7      # samples land on the two modes
        assert heads["flow"].predict(ctx[:1], 32)["action"].abs().item() > 0.5


def test_policy_shapes():
    cfg = load_config(None, ["model.head=flow", "data.chunk=8", "data.context_frames=2", "model.pool=attn",
                             "model.d_model=32", "model.temporal_layers=1", "model.flow_hidden=32"])
    p = Policy(cfg, 0.2)
    x = torch.rand(3, 2, 3, 66, 200)
    out = p.predict(x, torch.zeros(3), 5)
    assert out["action"].shape == (3, 8) and out["samples"].shape == (3, 5, 8)
