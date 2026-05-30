#!/usr/bin/env bash
# Restart the PP service on a FIXED port (default 8005), killing the previous
# instance first via the PID file. Logs to tmp/server.log so you can watch the
# JSON flow live:  tail -f tmp/server.log
set -uo pipefail
cd "$(dirname "$0")"

PORT="${PP_PORT:-8005}"
PIDFILE="tmp/server.pid"
LOG="tmp/server.log"
mkdir -p tmp

# stop previous instance (explicit PID — avoids pgrep/pkill which exit 144 here)
if [ -f "$PIDFILE" ]; then
  OLD="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -n "${OLD:-}" ] && kill -0 "$OLD" 2>/dev/null; then
    kill "$OLD" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$OLD" 2>/dev/null || break; sleep 0.3; done
    kill -9 "$OLD" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
fi

PY=".venv/bin/python"; [ -x "$PY" ] || PY="python"
PP_PORT="$PORT" nohup "$PY" -m pp_service.server > "$LOG" 2>&1 &
echo "$!" > "$PIDFILE"
echo "started pid $(cat "$PIDFILE") on port $PORT  (log: $LOG)"
