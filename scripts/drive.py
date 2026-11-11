#!/usr/bin/env python
"""Closed-loop driving server for the Udacity self-driving-car simulator (autonomous mode).

  pip install -r requirements-sim.txt
  python scripts/drive.py --ckpt runs/dinov2_flow_s0/best.pt --log closed_loop/dinov2_flow_track2_gate.csv
  # then start the simulator, pick a track, click "Autonomous Mode"

Press ENTER in this terminal every time you have to rescue the car (it left the road / you
reset it). Those keypresses are logged as interventions and drive the autonomy metric.
"""
from __future__ import annotations

import argparse
import base64
import csv
import os
import sys
import time
from io import BytesIO

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="checkpoint(s); several = deep ensemble")
    ap.add_argument("--log", required=True, help="telemetry CSV to write")
    ap.add_argument("--port", type=int, default=4567)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--mc_dropout", action="store_true")
    ap.add_argument("--max_speed", type=float, default=20.0)
    ap.add_argument("--min_speed", type=float, default=8.0)
    ap.add_argument("--sigma_lo", type=float, default=None, help="default: median validation sigma (from ckpt)")
    ap.add_argument("--sigma_hi", type=float, default=None, help="default: 95th-pct validation sigma (from ckpt)")
    ap.add_argument("--no_gate", action="store_true", help="constant target speed (ablation)")
    ap.add_argument("--unc_source", default="max", choices=["sigma", "ood", "max"])
    ap.add_argument("--no_temporal_ensemble", action="store_true")
    ap.add_argument("--te_decay", type=float, default=0.5)
    args = ap.parse_args()

    import eventlet
    import eventlet.tpool
    import eventlet.wsgi
    import socketio
    from flask import Flask
    from PIL import Image

    from drivefm.agent import DrivingAgent
    from drivefm.utils import get_device, load_policies

    device = get_device(args.device)
    policy, cfg = load_policies(args.ckpt, device)
    agent = DrivingAgent(policy, cfg, device, n_samples=args.samples or cfg["model"]["infer_samples"],
                         mc_dropout=args.mc_dropout, max_speed=args.max_speed, min_speed=args.min_speed,
                         sigma_lo=args.sigma_lo, sigma_hi=args.sigma_hi, gate=not args.no_gate, unc_source=args.unc_source,
                         temporal_ensemble=not args.no_temporal_ensemble, te_decay=args.te_decay)
    print(f"[gate] {'off' if args.no_gate else args.unc_source} sigma {agent.sigma_lo:.4f}-{agent.sigma_hi:.4f} "
          f"ood {agent.ood_lo}-{agent.ood_hi}")
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    fh = open(args.log, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(["wall_time", "step", "mode", "speed", "steer", "throttle", "risk", "target_speed",
                     "latency_ms", "intervention"])
    state = {"intervention": 0}

    sio = socketio.Server()
    app = Flask(__name__)

    def send(steer, throttle):
        sio.emit("steer", data={"steering_angle": str(steer), "throttle": str(throttle)}, skip_sid=True)

    @sio.on("telemetry")
    def telemetry(sid, data):
        if not data:  # manual mode
            writer.writerow([time.time(), agent.step, "manual", "", "", "", "", "", "", 0])
            sio.emit("manual", data={}, skip_sid=True)
            return
        speed = float(data["speed"])
        img = np.asarray(Image.open(BytesIO(base64.b64decode(data["image"]))).convert("RGB"))
        t0 = time.time()
        out = agent.act(img, speed)
        lat = (time.time() - t0) * 1000
        flag, state["intervention"] = state["intervention"], 0
        writer.writerow([t0, agent.step, "auto", f"{speed:.3f}", f"{out['steer']:.5f}", f"{out['throttle']:.4f}",
                         f"{out['risk']:.4f}", f"{out['target_speed']:.2f}", f"{lat:.1f}", flag])
        fh.flush()
        send(out["steer"], out["throttle"])

    @sio.on("connect")
    def connect(sid, environ):
        print("[sim] connected", sid)
        agent.reset()
        send(0.0, 0.0)

    def stdin_watcher():
        while True:
            line = eventlet.tpool.execute(sys.stdin.readline)
            if line == "":
                return
            state["intervention"] = 1
            print("[intervention marked]")

    eventlet.spawn(stdin_watcher)
    print(f"[drive] listening on :{args.port}  (ENTER = mark intervention, Ctrl-C = stop)")
    wsgi_app = socketio.WSGIApp(sio, app)
    try:
        eventlet.wsgi.server(eventlet.listen(("", args.port)), wsgi_app)
    finally:
        fh.close()


if __name__ == "__main__":
    main()
