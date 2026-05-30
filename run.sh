#!/usr/bin/env bash
# Launch the standalone post-processing service.
#
# Configuration lives in config.yaml (edit that, then run this).
# On start the server scans models/ and workflows/, pings each ComfyUI url,
# and prints what it found and which methods are READY.
#
# Models auto-resolve from ./models (reswapper_256.onnx, GFPGANv1.4.onnx);
# no environment variables are required. Env vars (PP_*, COMFY_*, FACE_*) still
# override config.yaml if you want to change something for one run.
set -euo pipefail
cd "$(dirname "$0")"

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec python -m pp_service.server
