#!/bin/bash
# scripts/prepare_data.sh
# ----------------------
# Run this on a cheap CPU instance or locally to generate Pfam shards
# and upload them to GCS. This avoids wasting expensive TPU MUFU time.

BUCKET_NAME="${BUCKET_NAME:?Set BUCKET_NAME to your GCS bucket name.}"
MAX_PAIRS=50000
SHARD_SIZE=500

echo "🚀 Starting Offline Pfam Embedding Job..."
echo "   Target: gs://$BUCKET_NAME/pfam_shards"

# Check for dependencies
python3 -c "import torch, transformers" 2>/dev/null
if [ $? -ne 0 ]; then
  echo "📦 Installing ESM-2 dependencies..."
  pip install -q torch transformers
fi

# Run the pipeline
python3 data/download_pfam.py \
  --max_pairs $MAX_PAIRS \
  --shard_size $SHARD_SIZE \
  --embed \
  --gcs_bucket $BUCKET_NAME

if [ $? -eq 0 ]; then
  echo "✓ Data preparation complete and synced to GCS!"
  echo "   You can now run 'bash scripts/tpu_auto_run.sh' safely."
else
  echo "✗ Data preparation failed."
  exit 1
fi
