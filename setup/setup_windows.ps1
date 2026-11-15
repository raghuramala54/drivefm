# DriveFM one-shot setup for Windows (tested target: RTX 4050 laptop, Windows 11).
#
# Run from the drivefm folder in PowerShell:
#     powershell -ExecutionPolicy Bypass -File setup\setup_windows.ps1
#
# What it does: checks Python 3.11 + NVIDIA driver, creates .venv, installs PyTorch with CUDA,
# the project requirements and (optionally) MetaDrive, then runs the hardware check and the tests.
# Re-running is safe.

param([switch]$NoMetaDrive, [switch]$SkipTests)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUTF8 = "1"

function Run([string]$what, [scriptblock]$cmd) {
    Write-Host "`n>>> $what" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $what (exit $LASTEXITCODE)" }
}

# 1. Python 3.11 (MetaDrive supports <= 3.11; everything else works on it too)
$hasPy = $false
try { $null = & py -3.11 --version 2>$null; $hasPy = ($LASTEXITCODE -eq 0) } catch { $hasPy = $false }
if (-not $hasPy) {
    Write-Host "Python 3.11 is not installed. Install it, then re-run this script:" -ForegroundColor Yellow
    Write-Host "    winget install -e --id Python.Python.3.11"
    exit 1
}

# 2. NVIDIA driver -> CUDA wheel (CUDA 13.0 needs driver >= 580, CUDA 12.6 needs >= 560)
$drv = $null
try { $drv = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1) } catch {}
if (-not $drv) {
    Write-Host "nvidia-smi not found. Install the latest NVIDIA GeForce driver (GeForce Experience / NVIDIA App), then re-run." -ForegroundColor Yellow
    exit 1
}
$parts = $drv.Split(",")
$gpu = $parts[0].Trim(); $ver = $parts[1].Trim()
$major = [int]($ver.Split(".")[0])
if ($major -ge 580) { $cu = "cu130" } elseif ($major -ge 560) { $cu = "cu126" } else {
    Write-Host "Driver $ver is too old for current PyTorch. Update the NVIDIA driver, then re-run." -ForegroundColor Yellow
    exit 1
}
Write-Host "GPU: $gpu  driver $ver  -> PyTorch CUDA wheel: $cu"

# 3. virtual environment
if (-not (Test-Path ".venv\Scripts\python.exe")) { Run "create .venv (Python 3.11)" { py -3.11 -m venv .venv } }
$py = ".\.venv\Scripts\python.exe"
Run "upgrade pip" { & $py -m pip install --upgrade pip wheel }
Run "install PyTorch ($cu)" { & $py -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$cu" }
Run "install requirements" { & $py -m pip install -r requirements.txt }
if (-not $NoMetaDrive) {
    & $py -m pip install -r requirements-metadrive.txt
    if ($LASTEXITCODE -ne 0) { Write-Host "MetaDrive install failed (optional) - DriveLite still works." -ForegroundColor Yellow }
}

# 4. check + tests
Run "hardware check" { & $py scripts\check_setup.py }
if (-not $SkipTests) { Run "test suite (~2 min)" { & $py -m pytest -q } }

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Next (in Git Bash, from this folder):"
Write-Host "    source .venv/Scripts/activate"
Write-Host "    PROFILE=quick bash scripts/run_focused.sh      # ~30-60 min end-to-end check"
Write-Host "    PROFILE=laptop bash scripts/run_focused.sh     # the paper run"
