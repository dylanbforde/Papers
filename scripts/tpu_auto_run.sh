#!/bin/bash
# scripts/tpu_auto_run.sh
set -euo pipefail
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_ID="sinkhorn-neurips"
ZONE="us-east5-a"
NODE_ID="sinkhorn-benchmark-cluster"
RESOURCE_ID="sinkhorn-queue"
BUCKET_NAME="sinkhorn-neurips"
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
ACTIVE_ACCOUNT="${ACTIVE_ACCOUNT:-$(gcloud config get-value account 2>/dev/null || true)}"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"
SOURCE_ARCHIVE="${SOURCE_ARCHIVE:-/tmp/sinkhorn-aligner-${SOURCE_COMMIT}.tar.gz}"

if [ -z "$ACTIVE_ACCOUNT" ]; then
  echo "✗ No active gcloud account found. Run 'gcloud auth login' or 'gcloud config set account ACCOUNT' before launching TPU jobs."
  exit 1
fi

# Function to guarantee cleanup on exit or failure
cleanup() {
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
    --accelerator-type=v6e-8 --runtime-version=tpu-ubuntu2204-base --quiet
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
  pip install -U -q jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
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
gcloud compute tpus tpu-vm scp "$SOURCE_ARCHIVE" "$NODE_ID:~/sinkhorn-aligner.tar.gz" --project="$PROJECT_ID" --zone="$ZONE" --worker=all --quiet --scp-flag="-o StrictHostKeyChecking=no" --scp-flag="-o UserKnownHostsFile=/dev/null"
gcloud compute tpus tpu-vm ssh $NODE_ID --project=$PROJECT_ID --zone=$ZONE --worker=all --command="
  set -euo pipefail && \
  tar -xzf ~/sinkhorn-aligner.tar.gz -C sinkhorn-aligner && \
  rm -f ~/sinkhorn-aligner.tar.gz && \
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

# 5. Run training (Pfam Seed Alignment - Production Run)
echo "Step 5: Starting training (with Preemption Checkpointing)..."
gcloud compute tpus tpu-vm ssh $NODE_ID --project=$PROJECT_ID --zone=$ZONE --worker=all --command="
  set -euo pipefail && \
  export PATH=\"\$PATH:/home/\$USER/.local/bin\" && \
  cd sinkhorn-aligner && \
  python3 -u src/benchmarks/benchmark_pfam_training.py --method sinkhorn_ot --max_hours $SINKHORN_MAX_HOURS --data_dir data/pfam_shards --gcs_ckpt_dir $GCS_CKPT_DIR --band_width $BAND_WIDTH --n_iters 15 --sinkhorn_gradient_mode $SINKHORN_GRADIENT_MODE --sinkhorn_refine_steps $SINKHORN_REFINE_STEPS --sinkhorn_epsilon $SINKHORN_EPSILON --sinkhorn_epsilon_scaling_steps $SINKHORN_EPSILON_SCALING_STEPS --sinkhorn_epsilon_scaling_factor $SINKHORN_EPSILON_SCALING_FACTOR --support_reconstruction $SUPPORT_RECONSTRUCTION --train_reconstruction_weight $TRAIN_RECONSTRUCTION_WEIGHT --train_sparse_ce_weight $TRAIN_SPARSE_CE_WEIGHT --train_monotonicity_weight $TRAIN_MONOTONICITY_WEIGHT --projection_manifold $PROJECTION_MANIFOLD --gap_mode $GAP_MODE --metric_every $METRIC_EVERY --autotune_batch_size --autotune_max_per_device_batch_size $AUTOTUNE_MAX_PER_DEVICE_BATCH_SIZE --batch_size_file checkpoints/pfam_batch_size.json --checkpoint_every 250 && \
  python3 -u src/benchmarks/benchmark_pfam_training.py --method flash_attention --max_hours $FLASH_MAX_HOURS --data_dir data/pfam_shards --gcs_ckpt_dir $GCS_CKPT_DIR --band_width $BAND_WIDTH --n_iters 15 --projection_manifold $PROJECTION_MANIFOLD --gap_mode $GAP_MODE --metric_every $METRIC_EVERY --batch_size_file checkpoints/pfam_batch_size.json --checkpoint_every 250" --quiet --ssh-flag="-o StrictHostKeyChecking=no" --ssh-flag="-o UserKnownHostsFile=/dev/null"

if [ $? -ne 0 ]; then
  echo "✗ Training script failed. Logs should be above."
  exit 1
fi

echo "✓ Full experiment complete!"
