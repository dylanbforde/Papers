#!/bin/bash
# scripts/tpu_screen_ensure.sh
set -euo pipefail
export CLOUDSDK_CORE_DISABLE_PROMPTS=1
export TERM="${TERM:-xterm-256color}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-forde.dylan@gmail.com}"
PROJECT_ID="${PROJECT_ID:-sinkhorn-neurips}"
PRIMARY_ZONE="${PRIMARY_ZONE:-us-east5-a}"
PRIMARY_RESOURCE_ID="${PRIMARY_RESOURCE_ID:-sinkhorn-screen-queue}"
PRIMARY_NODE_ID="${PRIMARY_NODE_ID:-sinkhorn-screen-cluster}"
RACE_ZONES="${RACE_ZONES:-us-east5-a us-east5-b}"
RACE_POLL_INTERVAL_SECS="${RACE_POLL_INTERVAL_SECS:-60}"

LAUNCHER="$SCRIPT_DIR/tpu_sinkhorn_screen.sh"
DETACHED="$SCRIPT_DIR/start_tpu_job_detached.sh"
RACE_WRAPPER="$SCRIPT_DIR/tpu_multi_zone_race.sh"

read -r -a ZONES <<< "$RACE_ZONES"

set_gcloud_context() {
  gcloud config set account "$GCLOUD_ACCOUNT" >/dev/null 2>&1 || true
  gcloud config set project "$PROJECT_ID" >/dev/null 2>&1 || true
  gcloud config set compute/zone "$PRIMARY_ZONE" >/dev/null 2>&1 || true
}

queue_state() {
  local zone="$1"
  local resource_id="$2"
  gcloud compute tpus queued-resources describe "$resource_id" \
    --project="$PROJECT_ID" \
    --zone="$zone" \
    --format='value(state.state)' 2>/dev/null || true
}

delete_queue() {
  local zone="$1"
  local resource_id="$2"
  gcloud compute tpus queued-resources delete "$resource_id" \
    --project="$PROJECT_ID" \
    --zone="$zone" \
    --force \
    --quiet || true
}

print_status() {
  echo "=== Processes ==="
  pgrep -af 'scripts/tpu_sinkhorn_screen.sh|scripts/tpu_multi_zone_race.sh|scripts/start_tpu_job_detached.sh' || true
  echo "=== Queues ==="
  local zone
  for zone in "${ZONES[@]}"; do
    gcloud compute tpus queued-resources list --project="$PROJECT_ID" --zone="$zone" || true
  done
  echo "=== TPU VMs ==="
  for zone in "${ZONES[@]}"; do
    gcloud compute tpus tpu-vm list --project="$PROJECT_ID" --zone="$zone" || true
  done
}

restart_single_zone_screen() {
  echo "[$(date)] Starting detached single-zone screen launcher in $PRIMARY_ZONE ..."
  bash "$DETACHED" "$LAUNCHER"
}

start_multi_zone_race() {
  echo "[$(date)] Starting detached multi-zone race across: ${ZONES[*]}"
  RACE_ZONES="$RACE_ZONES" \
  RACE_POLL_INTERVAL_SECS="$RACE_POLL_INTERVAL_SECS" \
  bash "$DETACHED" "$RACE_WRAPPER" "$LAUNCHER"
}

resume_existing_queue_with_race() {
  echo "[$(date)] Starting detached multi-zone race while preserving existing $PRIMARY_ZONE queue ..."
  EXISTING_ZONE="$PRIMARY_ZONE" \
  EXISTING_RESOURCE_ID="$PRIMARY_RESOURCE_ID" \
  EXISTING_NODE_ID="$PRIMARY_NODE_ID" \
  RACE_ZONES="$RACE_ZONES" \
  RACE_POLL_INTERVAL_SECS="$RACE_POLL_INTERVAL_SECS" \
  bash "$DETACHED" "$RACE_WRAPPER" "$LAUNCHER"
}

set_gcloud_context

if ! gcloud auth print-access-token >/dev/null 2>&1; then
  echo "✗ gcloud credentials are not usable for account $GCLOUD_ACCOUNT."
  echo "Run: gcloud auth login && gcloud config set account $GCLOUD_ACCOUNT"
  exit 1
fi

if [ ! -f "$RACE_WRAPPER" ]; then
  echo "✗ Missing $RACE_WRAPPER. Pull the latest repo state first."
  exit 1
fi

RACE_SUPPORTS_EXISTING=0
if grep -Eq 'EXISTING_ZONE|EXISTING_RESOURCE_ID|EXISTING_NODE_ID' "$RACE_WRAPPER" 2>/dev/null; then
  RACE_SUPPORTS_EXISTING=1
fi

PRIMARY_STATE="$(queue_state "$PRIMARY_ZONE" "$PRIMARY_RESOURCE_ID")"

BROKEN_RACE_IDS=()
for zone in "${ZONES[@]}"; do
  if [ "$zone" = "$PRIMARY_ZONE" ]; then
    maybe_broken="${PRIMARY_RESOURCE_ID}-${zone}-"
  else
    maybe_broken="${PRIMARY_RESOURCE_ID}-${zone}-"
  fi
  maybe_broken_state="$(queue_state "$zone" "$maybe_broken")"
  if [ -n "$maybe_broken_state" ]; then
    BROKEN_RACE_IDS+=("${zone}:${maybe_broken}:${maybe_broken_state}")
  fi
done

if [ "${#BROKEN_RACE_IDS[@]}" -gt 0 ]; then
  echo "[$(date)] Found broken stray race queues from the earlier wrapper version."
  pgrep -af 'scripts/tpu_multi_zone_race.sh' || true
  pkill -f 'scripts/tpu_multi_zone_race.sh' || true
  for item in "${BROKEN_RACE_IDS[@]}"; do
    IFS=: read -r zone resource_id state <<< "$item"
    echo "[$(date)] Deleting stray queue $resource_id in $zone (state: $state)"
    delete_queue "$zone" "$resource_id"
  done
  PRIMARY_STATE="$(queue_state "$PRIMARY_ZONE" "$PRIMARY_RESOURCE_ID")"
fi

if pgrep -f 'bash scripts/tpu_multi_zone_race.sh' >/dev/null 2>&1; then
  echo "[$(date)] A multi-zone race wrapper is already running. Leaving it alone."
  print_status
  exit 0
fi

if pgrep -f 'bash scripts/tpu_sinkhorn_screen.sh' >/dev/null 2>&1; then
  echo "[$(date)] A single-zone screen launcher is already running. Leaving it alone."
  print_status
  exit 0
fi

if [ -n "$PRIMARY_STATE" ]; then
  echo "[$(date)] Found existing canonical queue $PRIMARY_RESOURCE_ID in $PRIMARY_ZONE (state: $PRIMARY_STATE)."
  if [ "$RACE_SUPPORTS_EXISTING" = "1" ]; then
    resume_existing_queue_with_race
  else
    restart_single_zone_screen
  fi
  exit 0
fi

start_multi_zone_race
