#!/usr/bin/env bash
# Launch the standalone post-processing service.
#
# Required model setup (large ONNX files live OUTSIDE this repo):
#   FACE_SERVICE_MODELS_DIR   dir containing buffalo_l/ + the swap model
#   FACE_SERVICE_MODEL_PATH   path to inswapper_128.onnx or reswapper_256.onnx
#   FACE_ENHANCE_MODEL_PATH   (optional) GFPGAN/CodeFormer/GPEN ONNX
#
# Service settings:
#   PP_PORT (default 8005), PP_HOST (default 0.0.0.0)
set -euo pipefail
cd "$(dirname "$0")"

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec python -m pp_service.server
