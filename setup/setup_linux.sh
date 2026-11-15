#!/usr/bin/env bash
# DriveFM one-shot setup for Linux or WSL2 (Ubuntu). Run from the drivefm folder:
#     bash setup/setup_linux.sh            # add --no-metadrive to skip the optional simulator
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
MD=1; [ "${1:-}" = "--no-metadrive" ] && MD=0

PY=$(command -v python3.11 || true)
if [ -z "$PY" ]; then
  echo "Python 3.11 not found. Ubuntu: sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.11 python3.11-venv"
  echo "(or: pip install uv && uv python install 3.11, then re-run with PY=\$(uv python find 3.11))"
  exit 1
fi
if ! command -v nvidia-smi >/dev/null; then echo "nvidia-smi not found: install the NVIDIA driver (WSL: the Windows driver is enough)"; exit 1; fi
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); MAJOR=${DRV%%.*}
if [ "$MAJOR" -ge 580 ]; then CU=cu130; elif [ "$MAJOR" -ge 560 ]; then CU=cu126; else echo "driver $DRV too old - update it"; exit 1; fi
echo "driver $DRV -> PyTorch wheel $CU"

[ -x .venv/bin/python ] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/$CU
python -m pip install -r requirements.txt
[ $MD = 1 ] && { python -m pip install -r requirements-metadrive.txt || echo "MetaDrive install failed (optional)"; }
python scripts/check_setup.py
python -m pytest -q
echo -e "\nSetup complete. Next:\n  source .venv/bin/activate\n  PROFILE=quick bash scripts/run_focused.sh\n  PROFILE=laptop bash scripts/run_focused.sh"
echo "(headless Linux without a display: run MetaDrive stages under  xvfb-run -a ...)"
