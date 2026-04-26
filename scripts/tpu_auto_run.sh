#!/bin/bash
# scripts/tpu_auto_run.sh
set -euo pipefail
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_ID="${PROJECT_ID:-sinkhorn-neurips}"
ZONE="${ZONE:-us-east5-a}"
NODE_ID="${NODE_ID:-sinkhorn-benchmark-cluster}"
RESOURCE_ID="${RESOURCE_ID:-sinkhorn-queue}"
RUNTIME_VERSION="${RUNTIME_VERSION:-v2-alpha-tpuv6e}"
BUCKET_NAME="${BUCKET_NAME:-sinkhorn-neurips}"
GCS_CKPT_DIR="gs://$BUCKET_NAME/checkpoints"
BAND_WIDTH="${BAND_WIDTH:-1024}"
SINKHORN_MAX_HOURS="${SINKHORN_MAX_HOURS:-3.0}"
FLASH_MAX_HOURS="${FLASH_MAX_HOURS:-3.0}"
SINKHORN_GRADIENT_MODE="${SINKHORN_GRADIENT_MODE:-tail_refinement}"
SINKHORN_REFINE_STEPS="${SINKHORN_REFINE_STEPS:-2}"
SINKHORN_EPSILON="${SINKHORN_EPSILON:-1.0}"
SINKHORN_EPSILON_SCALING_STEPS="${SINKHORN_EPSILON_SCALING_STEPS:-0}"
SINKHORN_EPSILON_SCALING_FACTOR="${SINKHORN_EPSILON_SCALING_FACTOR:-2.0}"
TRAIN_RECONSTRUCTION_WEIGHT="${TRAIN_RECONSTRUCTION_WEIGHT:-1.0}"
TRAIN_SPARSE_CE_WEIGHT="${TRAIN_SPARSE_CE_WEIGHT:-0.0}"
TRAIN_MONOTONICITY_WEIGHT="${TRAIN_MONOTONICITY_WEIGHT:-0.0}"
PROJECTION_MANIFOLD="${PROJECTION_MANIFOLD:-stiefel_qk}"
GAP_MODE="${GAP_MODE:-none}"
SUPPORT_RECONSTRUCTION="${SUPPORT_RECONSTRUCTION:-none}"
METRIC_EVERY="${METRIC_EVERY:-50}"
AUTOTUNE_MAX_PER_DEVICE_BATCH_SIZE="${AUTOTUNE_MAX_PER_DEVICE_BATCH_SIZE:-8}"
JAX_TPU_INSTALL_SPEC="${JAX_TPU_INSTALL_SPEC:-jax[tpu]}"
JAX_TPU_PROBE_RETRIES="${JAX_TPU_PROBE_RETRIES:-10}"
JAX_TPU_PROBE_SLEEP_SECS="${JAX_TPU_PROBE_SLEEP_SECS:-30}"
ACTIVE_ACCOUNT="${ACTIVE_ACCOUNT:-$(gcloud config get-value account 2>/dev/null || true)}"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"
SOURCE_ARCHIVE="${SOURCE_ARCHIVE:-/tmp/sinkhorn-aligner-${SOURCE_COMMIT}.tar.gz}"
REMOTE_SOURCE_ARCHIVE="${REMOTE_SOURCE_ARCHIVE:-/tmp/sinkhorn-aligner-${SOURCE_COMMIT}.tar.gz}"
DELETE_WAITING_QUEUE_ON_EXIT="${DELETE_WAITING_QUEUE_ON_EXIT:-0}"

if [ -z "$ACTIVE_ACCOUNT" ]; then
  echo "✗ No active gcloud account found. Run 'gcloud auth login' or 'gcloud config set account ACCOUNT' before launching TPU jobs."
  exit 1
fi

# Function to guarantee cleanup on exit or failure
cleanup() {
  local status
  status="$(gcloud compute tpus queued-resources describe "$RESOURCE_ID" --project="$PROJECT_ID" --zone="$ZONE" --format='value(state.state)' 2>/dev/null || true)"
  if [ "$status" = "WAITING_FOR_RESOURCES" ] && [ "$DELETE_WAITING_QUEUE_ON_EXIT" != "1" ]; then
    echo "[$(date)] Preserving queued resource $RESOURCE_ID (state: $status). Set DELETE_WAITING_QUEUE_ON_EXIT=1 to force deletion while waiting..."
    return
  fi
  echo "[$(date)] Initiating mandatory cleanup to stop billing..."
  gcloud compute tpus queued-resources delete $RESOURCE_ID --project=$PROJECT_ID --zone=$ZONE --force --quiet
  echo "[$(date)] Cleanup complete."
}
# Trap catches errors, normal exits, or if you cancel it, and runs the cleanup function
trap cleanup EXIT

# 1. Start the queue
echo "Step 1: Queuing TPU..."
EXISTING_STATE="$(gcloud compute tpus queued-resources describe $RESOURCE_ID --project=$PROJECT_ID --zone=$ZONE --format='value(state.state)' 2>/dev/null || true)"
if [ -n "$EXISTING_STATE" ]; then
  echo "✓ Reusing existing queued resource $RESOURCE_ID (state: $EXISTING_STATE)"
else
  gcloud compute tpus queued-resources create $RESOURCE_ID \
    --project=$PROJECT_ID --zone=$ZONE --node-id=$NODE_ID \
    --accelerator-type=v6e-8 --runtime-version="$RUNTIME_VERSION" --quiet
fi

# 2. Wait until ACTIVE
echo "Step 2: Waiting for TPU to become ACTIVE..."
while true; do
  STATUS=$(gcloud compute tpus queued-resources describe $RESOURCE_ID --project=$PROJECT_ID --zone=$ZONE --format="value(state.state)")
  if [ -z "$STATUS" ]; then
    sleep 60
    continue
  fi
  
  echo "[$(date)] Status: $STATUS"
  if [ "$STATUS" == "ACTIVE" ]; then
    echo "✓ TPU is ACTIVE!"
    break
  fi
  
  if [ "$STATUS" == "FAILED" ]; then
    echo "✗ TPU Provisioning FAILED."
    exit 1 # The trap will automatically clean up
  fi
  sleep 60
done

# 3. Install dependencies
echo "Step 3: Installing dependencies..."
gcloud compute tpus tpu-vm ssh $NODE_ID --project=$PROJECT_ID --zone=$ZONE --worker=all --command="
  set -euo pipefail && \
  export DEBIAN_FRONTEND=noninteractive && \
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip && \
  export PATH=\"\$PATH:/home/\$USER/.local/bin\" && \
  pip install -U -q \"$JAX_TPU_INSTALL_SPEC\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
  pip install -U -q optax" --quiet --ssh-flag="-o StrictHostKeyChecking=no" --ssh-flag="-o UserKnownHostsFile=/dev/null"

if [ $? -ne 0 ]; then
  echo "✗ Dependency installation failed."
  exit 1
fi

# 4. Upload exact source snapshot from local checkout
echo "Step 4: Packaging and uploading source snapshot at commit $SOURCE_COMMIT..."
git -C "$REPO_ROOT" archive --format=tar.gz --output="$SOURCE_ARCHIVE" "$SOURCE_COMMIT"
gcloud compute tpus tpu-vm ssh $NODE_ID --project=$PROJECT_ID --zone=$ZONE --worker=all --command="
  set -euo pipefail && \
  rm -rf sinkhorn-aligner && \
  mkdir -p sinkhorn-aligner" --quiet --ssh-flag="-o StrictHostKeyChecking=no" --ssh-flag="-o UserKnownHostsFile=/dev/null"
gcloud compute tpus tpu-vm scp "$SOURCE_ARCHIVE" "$NODE_ID:$REMOTE_SOURCE_ARCHIVE" --project="$PROJECT_ID" --zone="$ZONE" --worker=all --quiet --scp-flag="-o StrictHostKeyChecking=no" --scp-flag="-o UserKnownHostsFile=/dev/null"
gcloud compute tpus tpu-vm ssh $NODE_ID --project=$PROJECT_ID --zone=$ZONE --worker=all --command="
  set -euo pipefail && \
  test -f $REMOTE_SOURCE_ARCHIVE && \
  tar -xzf $REMOTE_SOURCE_ARCHIVE -C sinkhorn-aligner && \
  rm -f $REMOTE_SOURCE_ARCHIVE && \
  echo source_commit=$SOURCE_COMMIT" --quiet --ssh-flag="-o StrictHostKeyChecking=no" --ssh-flag="-o UserKnownHostsFile=/dev/null"

if [ $? -ne 0 ]; then
  echo "✗ Source upload failed."
  exit 1
fi

# 4b. Fetch Pfam Shards from GCS (Rapid Pre-fetching)
echo "Step 4b: Fetching pre-computed Pfam shards from gs://$BUCKET_NAME..."
gcloud compute tpus tpu-vm ssh $NODE_ID --project=$PROJECT_ID --zone=$ZONE --worker=all --command="
  set -euo pipefail && \
  cd sinkhorn-aligner && \
  mkdir -p data/pfam_shards && \
  gsutil -m cp -r gs://$BUCKET_NAME/pfam_shards/* data/pfam_shards/" --quiet --ssh-flag="-o StrictHostKeyChecking=no" --ssh-flag="-o UserKnownHostsFile=/dev/null"

if [ $? -ne 0 ]; then
  echo "✗ GCS Fetch failed. Ensure you have run the offline embedding job first."
  exit 1
fi

echo "Step 4c0: Logging TPU control-plane details..."
gcloud compute tpus queued-resources describe $RESOURCE_ID --project=$PROJECT_ID --zone=$ZONE || true
gcloud compute tpus tpu-vm describe $NODE_ID --project=$PROJECT_ID --zone=$ZONE || true

echo "Step 4c: Verifying JAX TPU runtime..."
gcloud compute tpus tpu-vm ssh $NODE_ID --project=$PROJECT_ID --zone=$ZONE --worker=all --command="
  set -euo pipefail && \
  export PATH=\"\$PATH:/home/\$USER/.local/bin\" && \
  export JAX_FORCE_TPU_INIT=true && \
  cd sinkhorn-aligner && \
  for ATTEMPT in \$(seq 1 $JAX_TPU_PROBE_RETRIES); do \
    if python3 scripts/tpu_jax_probe.py; then \
      exit 0; \
    fi; \
    if [ \"\$ATTEMPT\" -eq $JAX_TPU_PROBE_RETRIES ]; then \
      echo \"✗ JAX TPU probe failed after $JAX_TPU_PROBE_RETRIES attempts.\" >&2; \
      exit 1; \
    fi; \
    echo \"JAX TPU probe failed on attempt \$ATTEMPT/$JAX_TPU_PROBE_RETRIES. Retrying in $JAX_TPU_PROBE_SLEEP_SECS seconds...\" >&2; \
    sleep $JAX_TPU_PROBE_SLEEP_SECS; \
  done" --quiet --ssh-flag="-o StrictHostKeyChecking=no" --ssh-flag="-o UserKnownHostsFile=/dev/null"

# 5. Run training (Pfam Seed Alignment - Production Run)
echo "Step 5: Starting training (with Preemption Checkpointing)..."
gcloud compute tpus tpu-vm ssh $NODE_ID --project=$PROJECT_ID --zone=$ZONE --worker=all --command="
  set -euo pipefail && \
  export PATH=\"\$PATH:/home/\$USER/.local/bin\" && \
  export JAX_FORCE_TPU_INIT=true && \
  cd sinkhorn-aligner && \
  python3 -u src/benchmarks/benchmark_pfam_training.py --method sinkhorn_ot --max_hours $SINKHORN_MAX_HOURS --data_dir data/pfam_shards --gcs_ckpt_dir $GCS_CKPT_DIR --band_width $BAND_WIDTH --n_iters 15 --sinkhorn_gradient_mode $SINKHORN_GRADIENT_MODE --sinkhorn_refine_steps $SINKHORN_REFINE_STEPS --sinkhorn_epsilon $SINKHORN_EPSILON --sinkhorn_epsilon_scaling_steps $SINKHORN_EPSILON_SCALING_STEPS --sinkhorn_epsilon_scaling_factor $SINKHORN_EPSILON_SCALING_FACTOR --support_reconstruction $SUPPORT_RECONSTRUCTION --train_reconstruction_weight $TRAIN_RECONSTRUCTION_WEIGHT --train_sparse_ce_weight $TRAIN_SPARSE_CE_WEIGHT --train_monotonicity_weight $TRAIN_MONOTONICITY_WEIGHT --projection_manifold $PROJECTION_MANIFOLD --gap_mode $GAP_MODE --metric_every $METRIC_EVERY --autotune_batch_size --autotune_max_per_device_batch_size $AUTOTUNE_MAX_PER_DEVICE_BATCH_SIZE --batch_size_file checkpoints/pfam_batch_size.json --checkpoint_every 250 && \
  python3 -u src/benchmarks/benchmark_pfam_training.py --method flash_attention --max_hours $FLASH_MAX_HOURS --data_dir data/pfam_shards --gcs_ckpt_dir $GCS_CKPT_DIR --band_width $BAND_WIDTH --n_iters 15 --projection_manifold $PROJECTION_MANIFOLD --gap_mode $GAP_MODE --metric_every $METRIC_EVERY --batch_size_file checkpoints/pfam_batch_size.json --checkpoint_every 250" --quiet --ssh-flag="-o StrictHostKeyChecking=no" --ssh-flag="-o UserKnownHostsFile=/dev/null"

if [ $? -ne 0 ]; then
  echo "✗ Training script failed. Logs should be above."
  exit 1
fi

echo "✓ Full experiment complete!"
