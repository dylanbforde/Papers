#!/bin/bash
# scripts/tpu_auto_run.sh

PROJECT_ID="sinkhorn-neurips"
ZONE="us-east5-a"
NODE_ID="sinkhorn-benchmark-cluster"
RESOURCE_ID="sinkhorn-queue"
BUCKET_NAME="sinkhorn-neurips"
GCS_CKPT_DIR="gs://$BUCKET_NAME/checkpoints"

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
gcloud compute tpus queued-resources create $RESOURCE_ID \
  --project=$PROJECT_ID --zone=$ZONE --node-id=$NODE_ID \
  --accelerator-type=v6e-8 --runtime-version=tpu-ubuntu2204-base 

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
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip git && \
  export PATH=\"\$PATH:/home/\$USER/.local/bin\" && \
  pip install -U -q jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
  pip install -U -q optax transformers torch"

if [ $? -ne 0 ]; then
  echo "✗ Dependency installation failed."
  exit 1
fi

# 4. Clone or update repo
echo "Step 4: Cloning/updating repository..."
gcloud compute tpus tpu-vm ssh $NODE_ID --project=$PROJECT_ID --zone=$ZONE --ssh-flag="-A" --command="
  if [ -d sinkhorn-aligner ]; then
    cd sinkhorn-aligner && git pull
  else
    GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no' git clone git@github.com:dylanbforde/sinkhorn-aligner.git
  fi"

if [ $? -ne 0 ]; then
  echo "✗ Repo clone failed."
  exit 1
fi

# 4b. Fetch Pfam Shards from GCS (Rapid Pre-fetching)
echo "Step 4b: Fetching pre-computed Pfam shards from gs://$BUCKET_NAME..."
gcloud compute tpus tpu-vm ssh $NODE_ID --project=$PROJECT_ID --zone=$ZONE --worker=all --command="
  cd sinkhorn-aligner && \
  mkdir -p data/pfam_shards && \
  gsutil -m cp -r gs://$BUCKET_NAME/pfam_shards/* data/pfam_shards/"

if [ $? -ne 0 ]; then
  echo "✗ GCS Fetch failed. Ensure you have run the offline embedding job first."
  exit 1
fi

# 5. Run training (Pfam Seed Alignment - Production Run)
echo "Step 5: Starting training (with Preemption Checkpointing)..."
gcloud compute tpus tpu-vm ssh $NODE_ID --project=$PROJECT_ID --zone=$ZONE --worker=all --command="
  export PATH=\"\$PATH:/home/\$USER/.local/bin\" && \
  cd sinkhorn-aligner && \
  python3 src/benchmarks/benchmark_pfam_training.py --method sinkhorn_ot --max_hours 3.0 --data_dir data/pfam_shards --gcs_ckpt_dir $GCS_CKPT_DIR && \
  python3 src/benchmarks/benchmark_pfam_training.py --method flash_attention --max_hours 3.0 --data_dir data/pfam_shards --gcs_ckpt_dir $GCS_CKPT_DIR"

if [ $? -ne 0 ]; then
  echo "✗ Training script failed. Logs should be above."
  exit 1
fi

echo "✓ Full experiment complete!"
