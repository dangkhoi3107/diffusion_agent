#!/usr/bin/env bash
# =====================================================================
#  Camouflage latent blending - one entry point (local + Kaggle)
#
#  bash run.sh <command> [--set key=value ...] [--grid key=v1,v2 ...]
#    ui           Gradio app (canvas, stage previews, batch)   -> share link
#    run          first subject x first background
#    batch        every subject x every background (+ metrics.csv, grid.png, zip)
#    sweep        batch over a parameter grid (configs/*.yaml `sweep:` or --grid)
#    preview      hints + mask + layout only, no diffusion (fast tuning)
#    show-config  print the resolved config
#    import-legacy --legacy-json ui_config_xxx.json   (old notebook config -> YAML)
#
#  Quick settings via environment variables (all optional):
#    CONFIG=configs/default.yaml     SUBJECTS=<img|folder>   BACKGROUNDS=<img|folder>
#    OUT_DIR=/kaggle/working/outputs STEPS=57  SEED=1234  SHARE=true
#    RUN_NAME=my_run SKIP_EXISTING=1 (resume an interrupted batch/sweep)
#    INSTALL_DEPS=auto|1|0           PYTHON=python3
#
#  Examples:
#    bash run.sh batch
#    SUBJECTS=/kaggle/input/x/cat.jpg bash run.sh run --set blend.alpha_end=0.6
#    bash run.sh sweep --grid blend.alpha_end=0.4,0.55,0.7 --grid mask.gamma=0.4,0.6
# =====================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMAND="${1:-batch}"
[[ $# -gt 0 ]] && shift

PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
CONFIG="${CONFIG:-$ROOT/configs/default.yaml}"
INSTALL_DEPS="${INSTALL_DEPS:-auto}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1

deps_ok() {
  "$PYTHON" - <<'PY' >/dev/null 2>&1
import importlib.metadata as md
from packaging.version import Version
need = {"diffusers": "0.30.0", "transformers": "4.44.0", "accelerate": "0.33.0", "gradio": "4.44.0", "PyYAML": "6.0"}
assert all(Version(md.version(pkg)) >= Version(v) for pkg, v in need.items())
import cv2, torch  # provided by the Kaggle image
PY
}

if [[ "$INSTALL_DEPS" == "1" ]] || { [[ "$INSTALL_DEPS" == "auto" ]] && ! deps_ok; }; then
  echo "[run.sh] Installing requirements ..."
  "$PYTHON" -m pip install -q -r "$ROOT/requirements.txt"
fi

ARGS=()
[[ "$COMMAND" != "import-legacy" ]] && ARGS+=(--config "$CONFIG")
if [[ -n "${SUBJECTS:-}" ]]; then ARGS+=(--subjects "$SUBJECTS"); fi
if [[ -n "${BACKGROUNDS:-}" ]]; then ARGS+=(--backgrounds "$BACKGROUNDS"); fi
if [[ -n "${OUT_DIR:-}" ]]; then ARGS+=(--out "$OUT_DIR"); fi
if [[ -n "${RUN_NAME:-}" ]]; then ARGS+=(--run-name "$RUN_NAME"); fi
if [[ -n "${STEPS:-}" ]]; then ARGS+=(--set "sampling.steps=$STEPS"); fi
if [[ -n "${SEED:-}" ]]; then ARGS+=(--set "sampling.seed=$SEED"); fi
if [[ -n "${SHARE:-}" ]]; then ARGS+=(--set "ui.share=$SHARE"); fi
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then ARGS+=(--set "output.skip_existing=true"); fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[run.sh] GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
  echo "[run.sh] No NVIDIA GPU detected - diffusion will be very slow on CPU."
fi
echo "[run.sh] $PYTHON -m camo $COMMAND ${ARGS[*]} $*"
exec "$PYTHON" -m camo "$COMMAND" "${ARGS[@]}" "$@"
