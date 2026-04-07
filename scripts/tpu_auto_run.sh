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

# 3. Setup and Run
# Note: Using ssh-flag="-A" to forward local SSH keys for git clone
# Make sure you have run 'ssh-add' on your local machine first!
echo "Step 3: Running setup and training pipeline on the remote cluster..."
gcloud compute tpus tpu-vm ssh $NODE_ID --project=$PROJECT_ID --zone=$ZONE --worker=all --ssh-flag="-A" --command="
  sudo apt-get update && sudo apt-get install python3-pip -y && \
  pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
  pip install optax transformers torch && \
  git clone git@github.com:dylanbforde/sinkhorn-aligner.git && \
  cd sinkhorn-aligner && \
  python src/benchmarks/benchmark_wallclock_training.py --method sinkhorn_ot --max_hours 2.5 && \
  python src/benchmarks/benchmark_wallclock_training.py --method flash_attention --max_hours 2.5; \
  sudo shutdown -h now
"

echo "✓ Full experiment complete! Results are in gs://sinkhorn-neurips-data/results/"
echo "Cleaning up queued resource..."
gcloud compute tpus queued-resources delete $RESOURCE_ID --project=$PROJECT_ID --zone=$ZONE --force --quiet
