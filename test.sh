#!/usr/bin/env bash
# Quick manual test against a running service.
#
#   ./test.sh                              -> health (what's READY)
#   ./test.sh SCENE.png REF.png            -> post-process (default method)
#   ./test.sh SCENE.png REF.png comfyui    -> force a method (internal|comfyui|multiswap)
#   ./test.sh SCENE.png A.png B.png        -> multiple reference faces
#
# Env: PP_URL (default http://127.0.0.1:8005), OUT (default ./pp_out.png)
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python"
URL="${PP_URL:-http://127.0.0.1:8005}"
OUT="${OUT:-./pp_out.png}"

if [ "$#" -eq 0 ]; then
  exec "$PY" test_postprocess.py --health --url "$URL"
fi

SCENE="$1"; shift
# Trailing arg is a method name if it matches one; otherwise all args are refs.
METHOD=""
ARGS=("$@")
last="${ARGS[${#ARGS[@]}-1]}"
case "$last" in
  internal|comfyui|multiswap)
    METHOD="$last"
    unset 'ARGS[${#ARGS[@]}-1]'
    ;;
esac

CMD=("$PY" test_postprocess.py "$SCENE" "${ARGS[@]}" -o "$OUT" --url "$URL")
[ -n "$METHOD" ] && CMD+=(--method "$METHOD")
echo "+ ${CMD[*]}"
exec "${CMD[@]}"
