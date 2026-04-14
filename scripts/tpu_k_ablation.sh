#!/bin/bash
# scripts/tpu_k_ablation.sh
set -euo pipefail
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROJECT_ID="${PROJECT_ID:-sinkhorn-neurips}"
ZONE="${ZONE:-us-east5-a}"
NODE_ID="${NODE_ID:-sinkhorn-k-ablation-cluster}"
RESOURCE_ID="${RESOURCE_ID:-sinkhorn-k-ablation-queue}"
BUCKET_NAME="${BUCKET_NAME:-sinkhorn-neurips}"
GCS_CKPT_ROOT="${GCS_CKPT_ROOT:-gs://$BUCKET_NAME/checkpoints_k_ablation}"
K_VALUES="${K_VALUES:-2 4}"
MAX_HOURS_PER_RUN="${MAX_HOURS_PER_RUN:-0.25}"
ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"
SOURCE_ARCHIVE="${SOURCE_ARCHIVE:-/tmp/sinkhorn-aligner-${SOURCE_COMMIT}.tar.gz}"

if [ -z "$ACTIVE_ACCOUNT" ]; then
  echo "✗ No active gcloud account found. Run 'gcloud auth login' or 'gcloud config set account ACCOUNT' before launching TPU jobs."
  exit 1
fi

cleanup() {
  echo "[$(date)] Initiating mandatory cleanup to stop billing..."
  gcloud compute tpus queued-resources delete "$RESOURCE_ID" --project="$PROJECT_ID" --zone="$ZONE" --force --quiet
  echo "[$(date)] Cleanup complete."
}
trap cleanup EXIT

echo "Step 1: Queuing TPU for K ablation..."
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
  STATUS=$(gcloud compute tpus queued-resources describe "$RESOURCE_ID" --project="$PROJECT_ID" --zone="$ZONE" --format="value(state.state)")
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

echo "Step 4b: Fetching pre-computed Pfam shards from gs://$BUCKET_NAME..."
gcloud compute tpus tpu-vm ssh "$NODE_ID" --project="$PROJECT_ID" --zone="$ZONE" --worker=all --command="
  set -euo pipefail && \
  cd sinkhorn-aligner && \
  mkdir -p data/pfam_shards && \
  gsutil -m cp -r gs://$BUCKET_NAME/pfam_shards/* data/pfam_shards/" --quiet --ssh-flag="-o StrictHostKeyChecking=no" --ssh-flag="-o UserKnownHostsFile=/dev/null"

echo "Step 5: Starting Sinkhorn K ablation..."
gcloud compute tpus tpu-vm ssh "$NODE_ID" --project="$PROJECT_ID" --zone="$ZONE" --worker=all --command="
  set -euo pipefail && \
  export PATH=\"\$PATH:/home/\$USER/.local/bin\" && \
  cd sinkhorn-aligner && \
  mkdir -p logs checkpoints && \
  for K in $K_VALUES; do \
    RUN_TAG=\"sinkhorn_k\${K}\" && \
    echo \"Running \${RUN_TAG} for $MAX_HOURS_PER_RUN hours\" && \
    python3 -u src/benchmarks/benchmark_pfam_training.py \
      --method sinkhorn_ot \
      --max_hours $MAX_HOURS_PER_RUN \
      --data_dir data/pfam_shards \
      --ckpt_dir checkpoints/\${RUN_TAG} \
      --gcs_ckpt_dir $GCS_CKPT_ROOT/\${RUN_TAG} \
      --band_width 1024 \
      --n_iters 15 \
      --sinkhorn_gradient_mode tail_refinement \
      --sinkhorn_refine_steps \${K} \
      --sinkhorn_epsilon 1.0 \
      --sinkhorn_epsilon_scaling_steps 0 \
      --sinkhorn_epsilon_scaling_factor 2.0 \
      --projection_manifold stiefel_qk \
      --gap_mode none \
      --metric_every 50 \
      --autotune_batch_size \
      --autotune_max_per_device_batch_size 8 \
      --batch_size_file checkpoints/pfam_batch_size_\${RUN_TAG}.json \
      --checkpoint_every 250 2>&1 | tee logs/\${RUN_TAG}.log ; \
  done" --quiet --ssh-flag="-o StrictHostKeyChecking=no" --ssh-flag="-o UserKnownHostsFile=/dev/null"

echo "✓ K ablation complete!"
