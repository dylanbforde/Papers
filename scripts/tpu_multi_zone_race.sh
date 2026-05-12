#!/bin/bash
# scripts/tpu_multi_zone_race.sh
set -euo pipefail
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_LAUNCHER="$SCRIPT_DIR/tpu_sinkhorn_screen.sh"
LAUNCHER="${1:-$DEFAULT_LAUNCHER}"
if [ "$#" -gt 0 ]; then
  shift
fi

if [[ "$LAUNCHER" != /* ]]; then
  LAUNCHER="$REPO_ROOT/${LAUNCHER#./}"
fi

if [ ! -f "$LAUNCHER" ]; then
  echo "✗ Launcher script not found: $LAUNCHER"
  exit 1
fi

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your GCP project id.}"
BUCKET_NAME="${BUCKET_NAME:?Set BUCKET_NAME to your GCS bucket name.}"
ACTIVE_ACCOUNT="${ACTIVE_ACCOUNT:-$(gcloud config get-value account 2>/dev/null || true)}"
RACE_ZONES="${RACE_ZONES:-us-east5-a us-east5-b}"
RACE_POLL_INTERVAL_SECS="${RACE_POLL_INTERVAL_SECS:-60}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v6e-8}"
RUNTIME_VERSION="${RUNTIME_VERSION:-v2-alpha-tpuv6e}"
PRESERVE_RACE_QUEUES_ON_EXIT="${PRESERVE_RACE_QUEUES_ON_EXIT:-0}"
EXISTING_ZONE="${EXISTING_ZONE:-}"
EXISTING_RESOURCE_ID="${EXISTING_RESOURCE_ID:-}"
EXISTING_NODE_ID="${EXISTING_NODE_ID:-}"

case "$(basename "$LAUNCHER")" in
  tpu_sinkhorn_screen.sh)
    DEFAULT_RESOURCE_BASE="sinkhorn-screen-queue"
    DEFAULT_NODE_BASE="sinkhorn-screen-cluster"
    ;;
  tpu_auto_run.sh)
    DEFAULT_RESOURCE_BASE="sinkhorn-queue"
    DEFAULT_NODE_BASE="sinkhorn-benchmark-cluster"
    ;;
  tpu_k_ablation.sh)
    DEFAULT_RESOURCE_BASE="sinkhorn-k-ablation-queue"
    DEFAULT_NODE_BASE="sinkhorn-k-ablation-cluster"
    ;;
  *)
    DEFAULT_RESOURCE_BASE="sinkhorn-race-queue"
    DEFAULT_NODE_BASE="sinkhorn-race-cluster"
    ;;
esac

RESOURCE_BASE="${RESOURCE_BASE:-$DEFAULT_RESOURCE_BASE}"
NODE_BASE="${NODE_BASE:-$DEFAULT_NODE_BASE}"

if [ -z "$ACTIVE_ACCOUNT" ]; then
  echo "✗ No active gcloud account found. Run 'gcloud auth login' or 'gcloud config set account ACCOUNT' before launching TPU jobs."
  exit 1
fi

read -r -a ZONES <<< "$RACE_ZONES"
if [ "${#ZONES[@]}" -eq 0 ]; then
  echo "✗ No race zones configured. Set RACE_ZONES to a space-separated list like 'us-east5-a us-east5-b'."
  exit 1
fi

sanitize_zone() {
  printf '%s' "$1" | tr -c '[:alnum:]-' '-'
}

resource_for_zone() {
  local zone="$1"
  if [ -n "$EXISTING_ZONE" ] && [ "$zone" = "$EXISTING_ZONE" ] && [ -n "$EXISTING_RESOURCE_ID" ]; then
    echo "$EXISTING_RESOURCE_ID"
    return
  fi
  echo "${RESOURCE_BASE}-$(sanitize_zone "$zone")"
}

node_for_zone() {
  local zone="$1"
  if [ -n "$EXISTING_ZONE" ] && [ "$zone" = "$EXISTING_ZONE" ] && [ -n "$EXISTING_NODE_ID" ]; then
    echo "$EXISTING_NODE_ID"
    return
  fi
  echo "${NODE_BASE}-$(sanitize_zone "$zone")"
}

describe_state() {
  local zone="$1"
  local resource_id="$2"
  gcloud compute tpus queued-resources describe "$resource_id" \
    --project="$PROJECT_ID" \
    --zone="$zone" \
    --format='value(state.state)' 2>/dev/null || true
}

delete_resource() {
  local zone="$1"
  local resource_id="$2"
  gcloud compute tpus queued-resources delete "$resource_id" \
    --project="$PROJECT_ID" \
    --zone="$zone" \
    --force \
    --quiet || true
}

cleanup() {
  local exit_code=$?
  if [ "$PRESERVE_RACE_QUEUES_ON_EXIT" = "1" ]; then
    echo "[$(date)] Preserving race queues on exit because PRESERVE_RACE_QUEUES_ON_EXIT=1."
    return "$exit_code"
  fi

  echo "[$(date)] Cleaning up multi-zone race queues..."
  local zone
  for zone in "${ZONES[@]}"; do
    local resource_id
    resource_id="$(resource_for_zone "$zone")"
    local state
    state="$(describe_state "$zone" "$resource_id")"
    if [ -n "$state" ]; then
      echo "[$(date)] Deleting $resource_id in $zone (state: $state)"
      delete_resource "$zone" "$resource_id"
    fi
  done
  return "$exit_code"
}
trap cleanup EXIT INT TERM

echo "Multi-zone TPU race"
echo "  Launcher: $LAUNCHER"
echo "  Zones: ${ZONES[*]}"
echo "  Project: $PROJECT_ID"
echo "  Resource base: $RESOURCE_BASE"
echo "  Node base: $NODE_BASE"
if [ -n "$EXISTING_ZONE" ]; then
  echo "  Existing queue: $EXISTING_ZONE -> ${EXISTING_RESOURCE_ID:-unset} / ${EXISTING_NODE_ID:-unset}"
fi

for zone in "${ZONES[@]}"; do
  resource_id="$(resource_for_zone "$zone")"
  node_id="$(node_for_zone "$zone")"
  state="$(describe_state "$zone" "$resource_id")"
  if [ -n "$state" ]; then
    echo "✓ Reusing existing queued resource $resource_id in $zone (state: $state)"
    continue
  fi

  echo "[$(date)] Creating queued resource $resource_id in $zone ..."
  gcloud compute tpus queued-resources create "$resource_id" \
    --project="$PROJECT_ID" \
    --zone="$zone" \
    --node-id="$node_id" \
    --accelerator-type="$ACCELERATOR_TYPE" \
    --runtime-version="$RUNTIME_VERSION" \
    --quiet
done

WINNER_ZONE=""
WINNER_RESOURCE_ID=""
WINNER_NODE_ID=""

while true; do
  active_count=0
  failed_count=0

  for zone in "${ZONES[@]}"; do
    resource_id="$(resource_for_zone "$zone")"
    state="$(describe_state "$zone" "$resource_id")"
    if [ -z "$state" ]; then
      state="MISSING"
    fi

    echo "[$(date)] $zone -> $resource_id : $state"

    if [ "$state" = "ACTIVE" ] && [ -z "$WINNER_ZONE" ]; then
      WINNER_ZONE="$zone"
      WINNER_RESOURCE_ID="$resource_id"
      WINNER_NODE_ID="$(node_for_zone "$zone")"
      active_count=$((active_count + 1))
      continue
    fi

    if [ "$state" = "ACTIVE" ]; then
      active_count=$((active_count + 1))
    elif [ "$state" = "FAILED" ] || [ "$state" = "SUSPENDED" ] || [ "$state" = "MISSING" ]; then
      failed_count=$((failed_count + 1))
    fi
  done

  if [ -n "$WINNER_ZONE" ]; then
    echo "✓ Winner: $WINNER_ZONE ($WINNER_RESOURCE_ID / $WINNER_NODE_ID)"
    break
  fi

  if [ "$failed_count" -eq "${#ZONES[@]}" ]; then
    echo "✗ All race zones failed or disappeared before activation."
    exit 1
  fi

  if [ "$active_count" -gt 1 ]; then
    echo "[$(date)] Multiple zones became ACTIVE simultaneously; selecting the first observed winner on the next poll."
  fi

  sleep "$RACE_POLL_INTERVAL_SECS"
done

for zone in "${ZONES[@]}"; do
  if [ "$zone" = "$WINNER_ZONE" ]; then
    continue
  fi

  resource_id="$(resource_for_zone "$zone")"
  state="$(describe_state "$zone" "$resource_id")"
  if [ -n "$state" ]; then
    echo "[$(date)] Deleting losing queued resource $resource_id in $zone (state: $state)"
    delete_resource "$zone" "$resource_id"
  fi
done

export PROJECT_ID
export BUCKET_NAME
export ZONE="$WINNER_ZONE"
export RESOURCE_ID="$WINNER_RESOURCE_ID"
export NODE_ID="$WINNER_NODE_ID"
export REQUIRE_EXISTING_RESOURCE=1
export REQUIRE_ACTIVE_RESOURCE=1

echo "[$(date)] Handing off to $LAUNCHER with winner zone $WINNER_ZONE ..."
exec bash "$LAUNCHER" "$@"
