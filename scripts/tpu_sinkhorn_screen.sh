#!/bin/bash
# scripts/tpu_sinkhorn_screen.sh
set -euo pipefail
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_ID="${PROJECT_ID:-sinkhorn-neurips}"
ZONE="${ZONE:-us-east5-a}"
NODE_ID="${NODE_ID:-sinkhorn-screen-cluster}"
RESOURCE_ID="${RESOURCE_ID:-sinkhorn-screen-queue}"
BUCKET_NAME="${BUCKET_NAME:-sinkhorn-neurips}"
MAX_HOURS_PER_RUN="${MAX_HOURS_PER_RUN:-0.20}"
METRIC_EVERY="${METRIC_EVERY:-25}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
AUTOTUNE_MAX_PER_DEVICE_BATCH_SIZE="${AUTOTUNE_MAX_PER_DEVICE_BATCH_SIZE:-8}"
GCS_DATA_PREFIX="${GCS_DATA_PREFIX:-gs://$BUCKET_NAME/pfam_shards}"
GCS_ARTIFACT_PREFIX="${GCS_ARTIFACT_PREFIX:-gs://$BUCKET_NAME/sinkhorn_screen}"
RUN_ID="${RUN_ID:-screen_$(date -u +%Y%m%dT%H%M%SZ)}"
SCREEN_CONFIGS="${SCREEN_CONFIGS:-k2_eps1:2:1.0:0:2.0:1024:none:none:1.0:0.0:0.0,k4_eps1:4:1.0:0:2.0:1024:none:none:1.0:0.0:0.0,muot_dustbin_mono:2:1.0:0:2.0:1024:none:dustbin_block:1.0:0.0:0.05,muot_dustbin_hybrid:2:1.0:0:2.0:1024:none:dustbin_block:0.5:0.5:0.05,narrow128_diag:2:1.0:0:2.0:128:none:none:1.0:0.0:0.0,narrow128_transport:2:1.0:0:2.0:128:transport_scout:none:1.0:0.0:0.0}"
ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"
SOURCE_ARCHIVE="${SOURCE_ARCHIVE:-/tmp/sinkhorn-aligner-${SOURCE_COMMIT}.tar.gz}"
ARTIFACT_DEST="${GCS_ARTIFACT_PREFIX%/}/$RUN_ID"
ARTIFACTS_SYNCED=0

if [ -z "$ACTIVE_ACCOUNT" ]; then
  echo "✗ No active gcloud account found. Run 'gcloud auth login' or 'gcloud config set account ACCOUNT' before launching TPU jobs."
  exit 1
fi

sync_artifacts() {
  if [ "$ARTIFACTS_SYNCED" -eq 1 ]; then
    return 0
  fi

  local status
  status="$(gcloud compute tpus queued-resources describe "$RESOURCE_ID" --project="$PROJECT_ID" --zone="$ZONE" --format='value(state.state)' 2>/dev/null || true)"
  if [ "$status" != "ACTIVE" ]; then
    echo "[$(date)] Skipping artifact sync (queue state: ${status:-unknown})."
    return 0
  fi

  echo "[$(date)] Syncing screen artifacts to $ARTIFACT_DEST ..."
  if gcloud compute tpus tpu-vm ssh "$NODE_ID" --project="$PROJECT_ID" --zone="$ZONE" --command="
    set -euo pipefail && \
    export PATH=\"\$PATH:/home/\$USER/.local/bin\" && \
    if [ ! -d sinkhorn-aligner ]; then
      echo 'No source snapshot found on leader; skipping artifact sync.' && exit 0
    fi && \
    cd sinkhorn-aligner && \
    mkdir -p logs screen_checkpoints && \
    printf 'run_id=%s\nresource_id=%s\nnode_id=%s\nsource_commit=%s\n' '$RUN_ID' '$RESOURCE_ID' '$NODE_ID' '$SOURCE_COMMIT' > screen_run_metadata.txt && \
    gsutil -m cp -r logs screen_checkpoints screen_run_metadata.txt '$ARTIFACT_DEST/'" --quiet --ssh-flag="-o StrictHostKeyChecking=no" --ssh-flag="-o UserKnownHostsFile=/dev/null"; then
    echo "[$(date)] Screen artifacts synced to $ARTIFACT_DEST"
    ARTIFACTS_SYNCED=1
  else
    echo "[$(date)] Warning: failed to sync screen artifacts to $ARTIFACT_DEST"
  fi
}

cleanup() {
  local exit_code=$?
  sync_artifacts || true
  echo "[$(date)] Initiating mandatory cleanup to stop billing..."
  gcloud compute tpus queued-resources delete "$RESOURCE_ID" --project="$PROJECT_ID" --zone="$ZONE" --force --quiet || true
  echo "[$(date)] Cleanup complete."
  return "$exit_code"
}
trap cleanup EXIT

echo "Step 1: Queuing TPU for Sinkhorn screen..."
EXISTING_STATE="$(gcloud compute tpus queued-resources describe "$RESOURCE_ID" --project="$PROJECT_ID" --zone="$ZONE" --format='value(state.state)' 2>/dev/null || true)"
if [ -n "$EXISTING_STATE" ]; then
  echo "✓ Reusing existing queued resource $RESOURCE_ID (state: $EXISTING_STATE)"
else
  gcloud compute tpus queued-resources create "$RESOURCE_ID" \
    --project="$PROJECT_ID" --zone="$ZONE" --node-id="$NODE_ID" \
    --accelerator-type=v6e-8 --runtime-version=tpu-ubuntu2204-base --quiet
fi

echo "Step 2: Waiting for TPU to become ACTIVE..."
while true; do
  STATUS="$(gcloud compute tpus queued-resources describe "$RESOURCE_ID" --project="$PROJECT_ID" --zone="$ZONE" --format='value(state.state)')"
  if [ -z "$STATUS" ]; then
    sleep 60
    continue
  fi

  echo "[$(date)] Status: $STATUS"
  if [ "$STATUS" = "ACTIVE" ]; then
    echo "✓ TPU is ACTIVE!"
    break
  fi
  if [ "$STATUS" = "FAILED" ]; then
    echo "✗ TPU Provisioning FAILED."
    exit 1
  fi
  sleep 60
done

echo "Step 3: Installing dependencies..."
gcloud compute tpus tpu-vm ssh "$NODE_ID" --project="$PROJECT_ID" --zone="$ZONE" --worker=all --command="
  set -euo pipefail && \
  export DEBIAN_FRONTEND=noninteractive && \
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip && \
  export PATH=\"\$PATH:/home/\$USER/.local/bin\" && \
  pip install -U -q jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
  pip install -U -q optax" --quiet --ssh-flag="-o StrictHostKeyChecking=no" --ssh-flag="-o UserKnownHostsFile=/dev/null"

echo "Step 4: Packaging and uploading source snapshot at commit $SOURCE_COMMIT..."
git -C "$REPO_ROOT" archive --format=tar.gz --output="$SOURCE_ARCHIVE" "$SOURCE_COMMIT"
gcloud compute tpus tpu-vm ssh "$NODE_ID" --project="$PROJECT_ID" --zone="$ZONE" --worker=all --command="
  set -euo pipefail && \
  rm -rf sinkhorn-aligner && \
  mkdir -p sinkhorn-aligner" --quiet --ssh-flag="-o StrictHostKeyChecking=no" --ssh-flag="-o UserKnownHostsFile=/dev/null"
gcloud compute tpus tpu-vm scp "$SOURCE_ARCHIVE" "$NODE_ID:~/sinkhorn-aligner.tar.gz" --project="$PROJECT_ID" --zone="$ZONE" --worker=all --quiet --scp-flag="-o StrictHostKeyChecking=no" --scp-flag="-o UserKnownHostsFile=/dev/null"
gcloud compute tpus tpu-vm ssh "$NODE_ID" --project="$PROJECT_ID" --zone="$ZONE" --worker=all --command="
  set -euo pipefail && \
  tar -xzf ~/sinkhorn-aligner.tar.gz -C sinkhorn-aligner && \
  rm -f ~/sinkhorn-aligner.tar.gz && \
  echo source_commit=$SOURCE_COMMIT" --quiet --ssh-flag="-o StrictHostKeyChecking=no" --ssh-flag="-o UserKnownHostsFile=/dev/null"

echo "Step 4b: Fetching pre-computed Pfam shards from $GCS_DATA_PREFIX..."
gcloud compute tpus tpu-vm ssh "$NODE_ID" --project="$PROJECT_ID" --zone="$ZONE" --worker=all --command="
  set -euo pipefail && \
  cd sinkhorn-aligner && \
  mkdir -p data/pfam_shards && \
  gsutil -m cp -r $GCS_DATA_PREFIX/* data/pfam_shards/" --quiet --ssh-flag="-o StrictHostKeyChecking=no" --ssh-flag="-o UserKnownHostsFile=/dev/null"

echo "Step 5: Starting short Sinkhorn configuration screen..."
gcloud compute tpus tpu-vm ssh "$NODE_ID" --project="$PROJECT_ID" --zone="$ZONE" --worker=all --command="
  set -euo pipefail && \
  export PATH=\"\$PATH:/home/\$USER/.local/bin\" && \
  cd sinkhorn-aligner && \
  mkdir -p logs screen_checkpoints && \
  IFS=',' read -r -a CONFIG_ENTRIES <<< \"$SCREEN_CONFIGS\" && \
  for ENTRY in \"\${CONFIG_ENTRIES[@]}\"; do \
    IFS=':' read -r RUN_TAG REFINE_STEPS EPSILON EPSILON_SCALING_STEPS EPSILON_SCALING_FACTOR BAND_WIDTH_CFG SUPPORT_RECONSTRUCTION_CFG GAP_MODE_CFG TRAIN_RECONSTRUCTION_WEIGHT TRAIN_SPARSE_CE_WEIGHT TRAIN_MONOTONICITY_WEIGHT <<< \"\$ENTRY\" && \
    BAND_WIDTH_CFG=\"\${BAND_WIDTH_CFG:-1024}\" && \
    SUPPORT_RECONSTRUCTION_CFG=\"\${SUPPORT_RECONSTRUCTION_CFG:-none}\" && \
    GAP_MODE_CFG=\"\${GAP_MODE_CFG:-none}\" && \
    TRAIN_RECONSTRUCTION_WEIGHT=\"\${TRAIN_RECONSTRUCTION_WEIGHT:-1.0}\" && \
    TRAIN_SPARSE_CE_WEIGHT=\"\${TRAIN_SPARSE_CE_WEIGHT:-0.0}\" && \
    TRAIN_MONOTONICITY_WEIGHT=\"\${TRAIN_MONOTONICITY_WEIGHT:-0.0}\" && \
    echo \"Running \$RUN_TAG for $MAX_HOURS_PER_RUN hours\" && \
    python3 -u src/benchmarks/benchmark_pfam_training.py \
      --method sinkhorn_ot \
      --max_hours $MAX_HOURS_PER_RUN \
      --data_dir data/pfam_shards \
      --ckpt_dir screen_checkpoints/\$RUN_TAG \
      --band_width \$BAND_WIDTH_CFG \
      --n_iters 15 \
      --sinkhorn_gradient_mode tail_refinement \
      --sinkhorn_refine_steps \$REFINE_STEPS \
      --sinkhorn_epsilon \$EPSILON \
      --sinkhorn_epsilon_scaling_steps \$EPSILON_SCALING_STEPS \
      --sinkhorn_epsilon_scaling_factor \$EPSILON_SCALING_FACTOR \
      --support_reconstruction \$SUPPORT_RECONSTRUCTION_CFG \
      --train_reconstruction_weight \$TRAIN_RECONSTRUCTION_WEIGHT \
      --train_sparse_ce_weight \$TRAIN_SPARSE_CE_WEIGHT \
      --train_monotonicity_weight \$TRAIN_MONOTONICITY_WEIGHT \
      --projection_manifold stiefel_qk \
      --gap_mode \$GAP_MODE_CFG \
      --metric_every $METRIC_EVERY \
      --autotune_batch_size \
      --autotune_max_per_device_batch_size $AUTOTUNE_MAX_PER_DEVICE_BATCH_SIZE \
      --batch_size_file screen_checkpoints/pfam_batch_size_\$RUN_TAG.json \
      --checkpoint_every $CHECKPOINT_EVERY 2>&1 | tee logs/\$RUN_TAG.log ; \
  done && \
  python3 scripts/summarize_benchmark_logs.py logs/*.log | tee logs/sinkhorn_screen_summary.txt" --quiet --ssh-flag="-o StrictHostKeyChecking=no" --ssh-flag="-o UserKnownHostsFile=/dev/null"

echo "✓ Sinkhorn screen complete!"
