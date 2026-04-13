#!/bin/bash
# scripts/start_tpu_job_detached.sh
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <launcher-script> [args...]"
  exit 1
fi

LAUNCHER="$1"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
JOB_NAME="$(basename "$LAUNCHER" .sh)"
LOG_PATH="$LOG_DIR/${JOB_NAME}_${STAMP}.log"
PID_PATH="${LOG_PATH}.pid"

mkdir -p "$LOG_DIR"

nohup bash "$LAUNCHER" "$@" >"$LOG_PATH" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_PATH"

echo "Started $LAUNCHER"
echo "PID: $PID"
echo "Log: $LOG_PATH"
echo "PID file: $PID_PATH"
