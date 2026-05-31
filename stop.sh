#!/usr/bin/env bash
# Stop the PP service started via run.sh / dev-restart.sh (uses tmp/server.pid),
# and clean up any stray pp_service.server processes.
set -uo pipefail
cd "$(dirname "$0")"

PIDFILE="tmp/server.pid"
if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    echo "stopped pid $PID (from $PIDFILE)"
  fi
  rm -f "$PIDFILE"
fi

# stray instances (e.g. started directly, not via the pidfile)
for p in $(pgrep -f "pp_service.server" 2>/dev/null || true); do
  kill "$p" 2>/dev/null && echo "stopped stray pid $p" || true
done
echo "done"
