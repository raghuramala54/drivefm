#!/usr/bin/env python
"""Check (and benchmark) this machine for the DriveFM study.

  python scripts/check_setup.py              # GPU, DINOv2 download + speed, DriveLite speed, MetaDrive if installed
  python scripts/check_setup.py --no_dinov2  # skip the ~90 MB DINOv2 download

Prints PASS/WARN/FAIL lines, recommended WORKERS / JOBS for scripts/run_focused.sh, and rough time
estimates for the laptop profile. Also writes results to logs/machine_check.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

R = {}


def say(status, msg):
    print(f"[{status:4}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_dinov2", action="store_true")
    ap.add_argument("--no_metadrive", action="store_true")
    ap.add_argument("--random_weights", action="store_true", help="debug: skip the DINOv2 download, time random weights")
    args = ap.parse_args()

    # ---------------------------------------------------------------- system
    print("\n== system")
    py = sys.version_info
    R["python"] = platform.python_version()
    say("PASS" if py >= (3, 9) else "FAIL", f"Python {R['python']} ({platform.system()} {platform.release()})")
    if py >= (3, 12):
        say("WARN", "MetaDrive needs Python <= 3.11; DriveLite works fine. Use Python 3.11 if you want the MetaDrive stage.")
    cores = os.cpu_count() or 1
    R["cpu_cores"] = cores
    say("PASS", f"CPU threads: {cores}")
    try:
        import psutil
        ram = psutil.virtual_memory().total / 2**30
        R["ram_gb"] = round(ram, 1)
        say("PASS" if ram >= 15 else "WARN", f"RAM: {ram:.1f} GB" + ("" if ram >= 15 else " (MetaDrive stage: use MD_JOBS=1)"))
    except ImportError:
        ram = None
    free = shutil.disk_usage(".").free / 2**30
    R["disk_free_gb"] = round(free, 1)
    say("PASS" if free >= 10 else "WARN", f"free disk here: {free:.0f} GB (laptop profile needs ~10 GB)")

    # ---------------------------------------------------------------- torch / GPU
    print("\n== PyTorch / GPU")
    import torch
    R["torch"] = torch.__version__
    say("PASS", f"torch {torch.__version__} (CUDA build {torch.version.cuda})")
    cuda = torch.cuda.is_available()
    vram = 0.0
    if not cuda:
        say("FAIL", "CUDA not available -> training runs on CPU (very slow). Reinstall torch from the CUDA index "
                    "(see setup scripts) and update the NVIDIA driver.")
    else:
        p = torch.cuda.get_device_properties(0)
        vram = p.total_memory / 2**30
        R.update(gpu=p.name, vram_gb=round(vram, 1), capability=f"{p.major}.{p.minor}",
                 bf16=bool(torch.cuda.is_bf16_supported()))
        say("PASS", f"GPU {p.name}, {vram:.1f} GB VRAM, compute {p.major}.{p.minor}, bf16={R['bf16']}")
        x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        for _ in range(3):
            x @ x
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(20):
            x @ x
        torch.cuda.synchronize()
        tf = 20 * 2 * 4096 ** 3 / (time.time() - t) / 1e12
        R["bf16_tflops"] = round(tf, 1)
        say("PASS", f"bf16 matmul: {tf:.1f} TFLOPS")
    dev = "cuda" if cuda else "cpu"

    # ---------------------------------------------------------------- DINOv2
    sec_per_epoch_dino = None
    if not args.no_dinov2:
        print("\n== DINOv2 backbone (downloads ~90 MB once)")
        try:
            from drivefm.config import load_config
            from drivefm.models import Policy
            cfg = load_config(["configs/focused/dinov2_flow.yaml", "configs/sims/drivelite.yaml"],
                              ["model.pretrained=false"] if args.random_weights else [])
            t = time.time()
            pol = Policy(cfg, 0.2).to(dev).train()
            say("PASS", f"DINOv2 weights loaded ({time.time() - t:.0f}s)")
            B, K = 64, cfg["data"]["context_frames"]
            H, W = cfg["model"]["img_size"]
            batch = {"images": torch.rand(B, K, 3, H, W, device=dev), "action": torch.randn(B, cfg["data"]["chunk"], device=dev) * 0.2,
                     "speed": torch.zeros(B, device=dev)}
            opt = torch.optim.AdamW([p for p in pol.parameters() if p.requires_grad], 1e-4)
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if cuda else torch.autocast("cpu", enabled=False)
            for i in range(8):
                if i == 3:
                    if cuda:
                        torch.cuda.synchronize()
                    t = time.time()
                with amp:
                    loss = pol.loss(batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if cuda:
                torch.cuda.synchronize()
            step = (time.time() - t) / 5
            sec_per_epoch_dino = step * 42000 / B
            mem = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0
            R.update(dinov2_step_s=round(step, 3), dinov2_peak_vram_gb=round(mem, 2))
            say("PASS" if mem < 0.8 * max(vram, 1e-9) or not cuda else "WARN",
                f"train step, batch {B} x {K} frames: {step * 1000:.0f} ms, peak VRAM {mem:.2f} GB "
                f"-> ~{sec_per_epoch_dino / 60:.1f} min/epoch of GPU compute on 42k samples")
        except Exception as e:  # noqa: BLE001
            say("FAIL", f"DINOv2 check failed: {e!r} (internet access to huggingface.co is needed once)")

    # ---------------------------------------------------------------- DriveLite
    print("\n== DriveLite simulator")
    from drivefm.envs import make_env
    env = make_env("drivelite", style=0)
    env.reset(0)
    t = time.time()
    for _ in range(200):
        env.step(env.expert_steer(), 0.3)
    fps = 200 / (time.time() - t)
    R["drivelite_fps"] = round(fps, 1)
    say("PASS", f"render + physics: {fps:.0f} steps/s per process")

    # ---------------------------------------------------------------- MetaDrive
    if not args.no_metadrive:
        print("\n== MetaDrive (optional)")
        try:
            import metadrive  # noqa: F401
            env = make_env("metadrive", max_steps=200)
            env.reset(0)
            t = time.time()
            for _ in range(50):
                env.step(env.expert_steer(), 0.3)
            mfps = 50 / (time.time() - t)
            env.close()
            R["metadrive_fps"] = round(mfps, 1)
            say("PASS" if mfps > 10 else "WARN", f"MetaDrive with camera: {mfps:.0f} steps/s")
        except ImportError:
            say("INFO", "MetaDrive not installed (optional): pip install -r requirements-metadrive.txt (Python <= 3.11)")
        except Exception as e:  # noqa: BLE001
            say("WARN", f"MetaDrive installed but camera rendering failed: {e!r}")

    # ---------------------------------------------------------------- recommendations
    print("\n== recommendations for scripts/run_focused.sh")
    workers = max(2, min(10, cores // 2))
    jobs = max(1, min(6, cores // 4, int(vram // 1.0) if cuda else 2))
    R.update(rec_workers=workers, rec_jobs=jobs)
    print(f"  WORKERS={workers}  JOBS={jobs}")
    if sec_per_epoch_dino:
        # rough: 6 DINOv2 runs x ~30 epochs (+ data loading overhead ~1.5x); PilotNet runs are cheap
        train_h = 6 * 30 * sec_per_epoch_dino * 1.5 / 3600 + 0.5
        cl_h = 150 * 20 * 700 / max(min(fps, 40), 1) / jobs / 3600
        R.update(est_train_h=round(train_h, 1), est_closed_loop_h=round(cl_h, 1))
        print(f"  rough laptop-profile time: data ~0.3 h, training ~{train_h:.1f} h, closed loop ~{cl_h:.1f} h, "
              f"DAgger ~2-3 h")
    os.makedirs("logs", exist_ok=True)
    with open("logs/machine_check.json", "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2)
    print("\nsaved logs/machine_check.json")


if __name__ == "__main__":
    main()
