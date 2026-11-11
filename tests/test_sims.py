"""Simulator tests: DriveLite physics/expert/sign conventions, and the full
collect -> train -> closed-loop eval -> DAgger loop. MetaDrive test runs only with DRIVEFM_TEST_METADRIVE=1."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import closed_loop_eval  # noqa: E402
import collect  # noqa: E402
import dagger  # noqa: E402
import train as train_script  # noqa: E402

from drivefm.envs import make_env, parse_seeds  # noqa: E402
from drivefm.envs.drivelite import STYLES, make_track  # noqa: E402


def test_parse_seeds():
    assert parse_seeds("0-3,10") == [0, 1, 2, 3, 10]


def test_tracks_valid_and_deterministic():
    a, b = make_track(7), make_track(7)
    assert np.allclose(a.xy, b.xy)
    assert np.abs(a.kappa).max() <= 1 / 15 + 1e-9 and 300 < a.length < 1500


def test_steering_sign_convention():
    env = make_env("drivelite", style=0)
    env.reset(3)
    for _ in range(15):
        _, _, info = env.step(1.0, 0.2)      # + = right
    assert info["cte"] < -0.3                 # cte is + to the LEFT of the centreline


@pytest.mark.parametrize("style", sorted(STYLES))
def test_expert_drives_and_renders(style):
    env = make_env("drivelite", style=style, side_cameras=True, max_steps=150)
    obs = env.reset(11)
    assert obs["image"].shape == (160, 320, 3) and obs["image"].dtype == np.uint8 and "left" in obs
    ctes = []
    while True:
        obs, done, info = env.step(env.expert_steer(), 0.3)
        ctes.append(abs(info["cte"]))
        if done:
            break
    assert not info["off_road"] and max(ctes) < 0.5


def test_full_loop(tmp_path):
    d = str(tmp_path)
    collect.main(["--env", "drivelite", "--out", f"{d}/dl_train", "--seeds", "0-1", "--max_steps", "120"])
    collect.main(["--env", "drivelite", "--out", f"{d}/dl_test_snow", "--seeds", "1000", "--style", "2",
                  "--max_steps", "60", "--noise", "0"])
    meta = json.load(open(f"{d}/dl_train/meta.json"))
    assert meta["frames_total"] > 150
    run = f"{d}/run"
    train_script.main(["--config", os.path.join(ROOT, "configs/pilotnet_mse.yaml"),
                       os.path.join(ROOT, "configs/sims/drivelite.yaml"), "--out", run, "--set",
                       f"data.train_logs=[{d}/dl_train/driving_log.csv]", f"data.test_logs=[{d}/dl_test_snow/driving_log.csv]",
                       "data.num_workers=0", "data.val_block_seconds=3", "train.epochs=1", "train.steps_per_epoch=3",
                       "model.head=flow", "model.pool=proj_flatten", "model.flow_hidden=32", "model.flow_layers=1",
                       "data.chunk=4", "model.infer_samples=4"])
    res = closed_loop_eval.main(["--env", "drivelite", "--seeds", "1000", "--max_steps", "25", "--ckpt",
                                 f"{run}/best.pt", "--out", f"{d}/cl", "--device", "cpu", "--video", "1"])
    s = res["summary"]
    assert s["episodes"] == 1 and 0 <= s["route_completion"] <= 1 and np.isfinite(s["expert_disagreement"])
    assert os.path.exists(os.path.join(d, "cl", res["name"], "seed1000.mp4"))
    dagger.main(["--env", "drivelite", "--init_ckpt", f"{run}/best.pt", "--base_logs", f"{d}/dl_train/driving_log.csv",
                 "--out", f"{d}/dagger", "--iters", "1", "--seeds_per_iter", "1", "--seed_start", "500",
                 "--max_steps", "40", "--epochs", "1", "--query", "uncertainty", "--risk_tau", "0.0",
                 "--device", "cpu", "--train_set", "data.num_workers=0", "train.steps_per_epoch=2"])
    hist = json.load(open(f"{d}/dagger/dagger_history.json"))
    assert hist[0]["rollout"]["labels"] > 0


@pytest.mark.skipif(os.environ.get("DRIVEFM_TEST_METADRIVE") != "1", reason="set DRIVEFM_TEST_METADRIVE=1")
def test_metadrive_wrapper():
    env = make_env("metadrive", max_steps=20)
    obs = env.reset(0)
    assert obs["image"].shape == (160, 320, 3)
    for _ in range(5):
        obs, done, info = env.step(env.expert_steer(), 0.3)
    assert {"cte", "off_road", "crash", "success", "route_completion"} <= set(info)
    env.close()
