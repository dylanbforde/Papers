#!/bin/bash
# scripts/tpu_queue_status.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your GCP project id.}"
ZONE="${ZONE:-us-east5-a}"
RESOURCE_ID="${1:-${RESOURCE_ID:-sinkhorn-screen-queue}}"

gcloud compute tpus queued-resources describe "$RESOURCE_ID" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --format="yaml(name,state.state,state.stateInitiatedTime)"
