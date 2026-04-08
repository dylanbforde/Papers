#!/bin/bash
# scripts/tpu_auto_run.sh
# Automated script to Provision -> Wait -> Setup -> Train -> Shutdown

PROJECT_ID="sinkhorn-neurips"
ZONE="us-east5-a"
NODE_ID="sinkhorn-benchmark-cluster"
RESOURCE_ID="sinkhorn-queue"

# 1. Start the queue
echo "Step 1: Queuing TPU..."
gcloud compute tpus queued-resources create $RESOURCE_ID \
  --project=$PROJECT_ID --zone=$ZONE --node-id=$NODE_ID \
  --accelerator-type=v6e-8 --runtime-version=tpu-ubuntu2204-base --spot

# 2. Wait until ACTIVE
echo "Step 2: Waiting for TPU to become ACTIVE (this may take minutes or hours)..."
echo "You can leave this running in Cloud Shell."
while true; do
  STATUS=$(gcloud compute tpus queued-resources describe $RESOURCE_ID --project=$PROJECT_ID --zone=$ZONE --format="value(state.state)")
  if [ -z "$STATUS" ]; then
    echo "[$(date)] Error: Could not fetch status. Check your project/zone."
    sleep 60
    continue
  fi
  
  echo "[$(date)] Status: $STATUS"
  
  if [ "$STATUS" == "ACTIVE" ]; then
    echo "✓ TPU is ACTIVE!"
    break
  fi
  
  if [ "$STATUS" == "FAILED" ]; then
    echo "✗ TPU Provisioning FAILED. Cleaning up..."
    gcloud compute tpus queued-resources delete $RESOURCE_ID --project=$PROJECT_ID --zone=$ZONE --force --quiet
    exit 1
  fi
  
  sleep 60
done

# 3. Install dependencies (idempotent — safe to re-run)
echo "Step 3: Installing dependencies..."
gcloud compute tpus tpu-vm ssh $NODE_ID --project=$PROJECT_ID --zone=$ZONE --worker=all --command="
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip git && \
  pip install -q jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
  pip install -q optax transformers torch
"

if [ $? -ne 0 ]; then
  echo "✗ Dependency installation failed."
  exit 1
fi

# 4. Clone or update repo (SSH in directly for agent forwarding)
echo "Step 4: Cloning/updating repository..."
gcloud compute tpus tpu-vm ssh $NODE_ID --project=$PROJECT_ID --zone=$ZONE --ssh-flag="-A" --command="
  if [ -d sinkhorn-aligner ]; then
    echo 'Repo exists, pulling latest...'
    cd sinkhorn-aligner && git pull
  else
    git clone git@github.com:dylanbforde/sinkhorn-aligner.git
  fi
"

if [ $? -ne 0 ]; then
  echo "✗ Repository clone/pull failed. Check SSH agent forwarding."
  exit 1
fi

# 5. Run training (both methods chained, auto-shutdown at the end)
echo "Step 5: Starting training (Sinkhorn OT 2.5h → Flash Attention 2.5h → shutdown)..."
gcloud compute tpus tpu-vm ssh $NODE_ID --project=$PROJECT_ID --zone=$ZONE --worker=all --command="
  cd sinkhorn-aligner && \
  python src/benchmarks/benchmark_wallclock_training.py --method sinkhorn_ot --max_hours 2.5 && \
  python src/benchmarks/benchmark_wallclock_training.py --method flash_attention --max_hours 2.5; \
  sudo shutdown -h now
"

echo "✓ Full experiment complete! Results are in gs://sinkhorn-neurips-data/results/"
echo "Cleaning up queued resource..."
gcloud compute tpus queued-resources delete $RESOURCE_ID --project=$PROJECT_ID --zone=$ZONE --force --quiet
