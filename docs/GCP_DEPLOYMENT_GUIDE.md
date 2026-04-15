# Google Cloud Platform (GCP) TPU Deployment Guide

This guide covers everything you need to run the **latency benchmark** (`benchmark_long_context_alignment.py`) and the current **Pfam training workflow** (`benchmark_pfam_training.py` plus the TPU automation scripts) on a Google Cloud TPU v6e (Trillium) instance for the NeurIPS paper.

Because you have never used GCP before, this guide assumes zero prior cloud experience.

The current production TPU workflow is:
1. `scripts/tpu_sinkhorn_screen.sh` for the short Sinkhorn-only Pfam screen.
2. `scripts/tpu_auto_run.sh` for the promoted Pfam Sinkhorn-vs-Flash run.

The older ortholog-DNA wall-clock benchmark remains in the repository as a legacy comparator, but it is not the primary TPU training path anymore.

---

## 1. Initial GCP Account Setup & Billing
Before you can provision a TPU, you need an active GCP account and a linked billing method.

1. Go to [console.cloud.google.com](https://console.cloud.google.com/) and sign in with your Google Account.
2. Accept the terms of service. Google usually gives **$300 in free credits** for 90 days if this is a new account, which is more than enough to run all your benchmarks.
3. Once in the console, **Create a New Project** (e.g., `sinkhorn-neurips`).
4. In the left navigation menu, go to **Billing** and link a credit card. Even with free credits, Google requires a card to verify identity to prevent abuse of ML instances.

## 2. Enabling the Bare Necessary APIs
GCP turns off background services by default to save you money. You need to enable the Compute Engine and TPU APIs.

1. Open the [Google Cloud Shell](https://cloud.google.com/shell/docs/run-gcloud-commands) (the terminal icon `>_` in the top right of the console dashboard).
2. Run these two commands to authorize the APIs for your project:
   ```bash
   gcloud services enable compute.googleapis.com
   gcloud services enable tpu.googleapis.com
   ```

## 3. Provisioning a Multi-TPU Pod Slice
To benchmark your Pallas kernel's sharding capabilities against SOTA contexts, you must provision a **Multi-TPU Pod Slice** (e.g., `v6e-8` or `v6e-16`). 

In the Cloud Shell, run the following command to create your multi-TPU VM cluster in an optimal zone (e.g., `us-central1-a`):

```bash
gcloud compute tpus tpu-vm create sinkhorn-benchmark-cluster \
  --zone=us-central1-a \
  --accelerator-type=v6e-8 \
  --version=tpu-ubuntu2204-base
```
*Note: `v6e-8` gives you 8 interconnected Trillium chips. If `us-central1-a` is out of capacity, try `--zone=us-east1-c` or `--zone=us-east5-a`. If all `v6e` (Trillium) chips are globally out of stock, you can swap the accelerator flag to `--accelerator-type=v5litepod-8` (v5e chips) or `--accelerator-type=v4-8` which run the exact same Pallas kernels!*

## 4. Connecting to the Instance & Code Transfer
Once the instance says `READY`, you can securely SSH into the primary machine of your cluster using the Cloud Shell:

```bash
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=us-east5-a
```

Now you are inside the leader TPU machine. To execute standard code across all hosts in a pod simultaneously (which is required if using slices larger than 8 chips), you use the `--worker=all` flag with your `gcloud` commands.

For the current benchmark workflow, you do **not** need to clone from GitHub on the TPU VM. The automation scripts package your exact local checkout with `git archive`, upload that snapshot, and unpack it on the TPU workers. That keeps the TPU run pinned to the source commit you launched.

If you still want to work manually on the VM, you can clone or copy the repo yourself, but the recommended path is to let `scripts/tpu_sinkhorn_screen.sh` and `scripts/tpu_auto_run.sh` handle source transfer.

## 5. Environment Setup (The Magic JAX Command)
Google Cloud TPU VMs come with Ubuntu, but you must install the **TPU-specific version of JAX** so the library knows how to compile the Pallas kernels for the distributed mesh. You must execute this setup across **ALL** workers in the pod.

Run these exact commands in your local Cloud Shell (not SSH'd inside a single host):

```bash
# Combine the entire installation block into one multi-worker execution command
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=us-east5-a --worker=all --command="sudo apt-get update && sudo apt-get install python3-pip -y && pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && pip install optax transformers torch"
```

## 6. Running the Latency Benchmark
Everything is ready! We baked `jax.distributed.initialize()` into the benchmark scripts so JAX automatically synchronizes across the entire TPU pod slice.

```bash
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=us-east5-a --worker=all --command="cd sinkhorn-aligner && python src/benchmarks/benchmark_long_context_alignment.py --model_name 'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species' --vocab dna"
```

### What to expect:
- **Time to run**: ~5–10 minutes. JAX spends ~3 min on XLA compilation; the tensor math finishes in seconds.
- **Cost**: A `v6e-8` pod costs ~$10–12/hr. The latency benchmark costs roughly **~$2.00**.
- Outputs Table 1 data (Flash Attention vs Sinkhorn OT latency at 1K/4K/8K/16K). Hardware telemetry goes to `/tmp/tensorboard_profiles_DATE/`.

## 6.1 Running the Current Pfam Training Workflow
The current TPU training path is a two-stage Pfam workflow on precomputed shards.

### Step 1: Run the short Sinkhorn screen

```bash
./scripts/tpu_sinkhorn_screen.sh
```

This queues `sinkhorn-screen-queue`, uploads your exact local source snapshot, runs the short Pfam screen, and syncs `logs/` plus `screen_checkpoints/` to GCS before tearing the queue down.

Check queue state while waiting:

```bash
./scripts/tpu_queue_status.sh sinkhorn-screen-queue
```

### Step 2: Promote the winning config into the full Pfam run

Set the winning configuration through environment variables and launch:

```bash
SINKHORN_REFINE_STEPS=2 \
SINKHORN_EPSILON=1.0 \
SUPPORT_RECONSTRUCTION=none \
GAP_MODE=none \
TRAIN_RECONSTRUCTION_WEIGHT=1.0 \
TRAIN_SPARSE_CE_WEIGHT=0.0 \
TRAIN_MONOTONICITY_WEIGHT=0.0 \
./scripts/tpu_auto_run.sh
```

This queues `sinkhorn-queue`, uploads the local snapshot, fetches Pfam shards from GCS, runs Sinkhorn first, then Flash, and checkpoints to GCS so reruns can resume.

### What to expect:
- **Data source**: precomputed Pfam shard dataset stored under `gs://sinkhorn-neurips/pfam_shards/`.
- **Screen artifacts**: synced to `gs://sinkhorn-neurips/sinkhorn_screen/<run-id>/`.
- **Production checkpoints**: synced to `gs://sinkhorn-neurips/checkpoints/`.
- **Batch size**: auto-tuned per device at startup; failures now surface explicitly instead of silently falling back.
- **Total time**: screen depends on `MAX_HOURS_PER_RUN` and number of configs; the main run depends on `SINKHORN_MAX_HOURS` and `FLASH_MAX_HOURS`.

### Manual fallback

If you need to invoke the trainer directly on an already-active TPU slice:

```bash
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=us-east5-a --worker=all --command="\
  cd sinkhorn-aligner && \
  python src/benchmarks/benchmark_pfam_training.py --method sinkhorn_ot --max_hours 0.25 --data_dir data/pfam_shards --autotune_batch_size"
```

## 6.2 Legacy Ortholog-DNA Wall-Clock Benchmark
`src/benchmarks/benchmark_wallclock_training.py` remains available for the older ortholog-DNA comparison. It is no longer the recommended TPU workflow, but it is now consistent with gradient descent after the optimizer-sign fix.

### ⚠️ Billing Safety: Auto-shutdown on completion/failure
If the Python script crashes, the TPU VM **stays running** and you keep paying ~$10-12/hr. To protect against this (especially for overnight runs), wrap the command so the **TPU auto-deletes itself** when the script finishes or fails:

```bash
# Safe overnight run: auto-deletes TPU after script completes (success OR failure)
ZONE=us-east5-a
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=$ZONE --worker=all --command="\
  cd sinkhorn-aligner && \
  python src/benchmarks/benchmark_wallclock_training.py --method sinkhorn_ot --max_hours 2.5; \
  sudo shutdown -h now"
# The VM shuts down after the script exits (success or crash).
# Then delete from your local machine / Cloud Shell:
# gcloud compute tpus tpu-vm delete sinkhorn-benchmark-cluster --zone=$ZONE --quiet
```

Alternatively, set a **hard timeout** from Cloud Shell that kills the VM after a max duration (e.g., 3 hours for a 2.5hr run):
```bash
# Start the training in the background, then schedule deletion after 3 hours
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=us-east5-a --worker=all --command="\
  cd sinkhorn-aligner && \
  nohup python src/benchmarks/benchmark_wallclock_training.py --method sinkhorn_ot --max_hours 2.5 > training.log 2>&1 &"

# Schedule auto-delete from Cloud Shell (10800s = 3 hours)
sleep 10800 && gcloud compute tpus tpu-vm delete sinkhorn-benchmark-cluster --zone=us-east5-a --quiet &
```

**Also recommended:** Set a [GCP Budget Alert](https://console.cloud.google.com/billing/budgets) for your project (e.g., $100) so you get an email if spending exceeds your threshold.

## 6.5 Real-World Training: Epochs & Checkpointing
When you transition from running the benchmark profiling script to doing an actual full-scale training run over Pfam or Nucleotide datasets, you must implement **Checkpointing** to protect against TPU preemptions.

1. **Epochs:** Standard sequence alignment training runs for roughly **50 to 100 epochs** over the dataset. You should wrap your JAX loss in an Optax learning rate scheduler (e.g., `optax.warmup_cosine_decay_schedule`).
2. **Cloud Checkpointing:** Because Spot/Preemptible TPUs can be shut off randomly by Google (to save 70% of the cost!), you must save your model weights and optimizer state using JAX's `orbax.checkpoint` library at the end of *every single epoch*.

## 6.6 Setting Up Google Cloud Storage (GCS) & Data Retrieval
TPU VMs have very little persistent disk space. All large datasets, checkpoints, and benchmark results should be read from and written to a Google Cloud Storage (GCS) Bucket.
1. **Create the Bucket**: Run this in your local Cloud Shell:
   ```bash
   gcloud storage buckets create gs://sinkhorn-neurips --location=europe-west4
   ```
2. **Where is my data?**: You can visually explore, upload, and download files from this bucket anytime by visiting [console.cloud.google.com/storage](https://console.cloud.google.com/storage) in your web browser.
3. **Downloading to your laptop**: To pull all training runs from the cloud bucket to your local machine:
   ```bash
   gcloud storage cp -r gs://sinkhorn-neurips/results ./local_folder
   ```

## 6.7 Determining TPU Batch Sizes (Gradient Accumulation)
Batch size on TPUs is determined strictly by the physical limits of the High Bandwidth Memory (HBM, 32GB on v6e).
* **Multiples of Mesh**: Your global `BATCH_SIZE` must be perfectly divisible by the number of chips (e.g., a multiple of 8 for a `v6e-8` slice) so the data shards equally.
* **Finding the Limit**: Because Sinkhorn Attention requires $O(L \cdot W)$ memory compared to standard tracking $O(L^2)$, you can physically fit a much larger `L`. To find your max batch size, start at 8 and double it until JAX throws an `Out Of Memory (OOM)` error during the initial XLA compilation phase, then step back down.
* **Gradient Accumulation**: If your model physically OOMs above a batch size of 8, but your MUON Optimizer mathematically requires a batch size of 128 for stability, you **must use `optax.MultiSteps`**. This allows you to process a physical micro-batch of 8, but accumulate the gradients in memory over 16 loops before actually applying the optimizer update ($8 \times 16 = 128$).

## 7. CRITICAL: Stop the Instance to Stop Paying
The second you are done generating the tables and pulling the JSON files off the cluster, **you must delete or stop the TPU array**. Multi-TPU pods burn credits extremely fast.

From your local machine or the main GCP Cloud Console Shell:
```bash
# To stop it (keeps the disk data, charges a tiny storage fee)
gcloud compute tpus tpu-vm stop sinkhorn-benchmark-cluster --zone=us-east5-a
```

To delete the cluster entirely (which stops all charges permanently and frees up your internal Google TPU quota):
```bash
gcloud compute tpus tpu-vm delete sinkhorn-benchmark-cluster --zone=us-east5-a
```

## 8. Handling Capacity Issues (v6e Trillium)
TPU Trillium models (`v6e`) are in extremely high demand globally. If you get an `Insufficient Capacity` or `RESOURCE_EXHAUSTED` error, you have three options:

### Option A: Queued Resources (Recommended — "Set it and forget it")
GCP's **Queued Resources** API lets you submit a TPU request that sits in a queue until capacity frees up. GCP automatically provisions your TPU the moment chips become available — no manual retrying needed.

```bash
# Submit a queued request — GCP will auto-provision when capacity is available
gcloud compute tpus queued-resources create sinkhorn-queue \
  --node-id=sinkhorn-benchmark-cluster \
  --zone=us-east5-a \
  --accelerator-type=v6e-8 \
  --runtime-version=tpu-ubuntu2204-base

# Check status (will show WAITING_FOR_RESOURCES → PROVISIONING → ACTIVE)
gcloud compute tpus queued-resources describe sinkhorn-queue --zone=us-east5-a

# Once ACTIVE, SSH in as normal
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=us-east5-a
```

To clean up the queued resource after you're done:
```bash
gcloud compute tpus queued-resources delete sinkhorn-queue --zone=us-east5-a --force
```

**Tip:** You can submit queued requests to multiple zones simultaneously — whichever fills first wins:
```bash
for ZONE in us-east5-a us-central1-a us-east1-c europe-west4-b; do
  gcloud compute tpus queued-resources create sinkhorn-queue-${ZONE} \
    --node-id=sinkhorn-cluster-${ZONE} \
    --zone=${ZONE} \
    --accelerator-type=v6e-8 \
    --runtime-version=tpu-ubuntu2204-base
done
```

### Option B: Spot VMs (60–70% cheaper, better availability)
Spot TPU VMs are heavily discounted and often have better availability since they draw from a separate capacity pool. The tradeoff: Google can preempt (shut down) your VM at any time if they need the chips back.

For a 2.5-hour training run, this is usually fine — preemptions are uncommon for short jobs, and our script checkpoints every 100 steps to GCS.

```bash
# Queued + Spot (cheapest option that auto-provisions)
gcloud compute tpus queued-resources create sinkhorn-spot-queue \
  --node-id=sinkhorn-benchmark-cluster \
  --zone=us-east5-a \
  --accelerator-type=v6e-8 \
  --runtime-version=tpu-ubuntu2204-base \
  --spot
```

Or create directly (if capacity is available right now):
```bash
gcloud compute tpus tpu-vm create sinkhorn-benchmark-cluster \
  --zone=us-east5-a \
  --accelerator-type=v6e-8 \
  --version=tpu-ubuntu2204-base \
  --spot
```

### Option C: Auto-retry script (brute force)
If you prefer not to use the queued API, save this as `provision_tpu.sh` and run it — it tries every zone in a loop until one succeeds:

```bash
#!/bin/bash
ZONES=("us-east5-a" "us-central1-a" "us-east1-c" "europe-west4-b" "us-central2-b")

while true; do
  for ZONE in "${ZONES[@]}"; do
    echo "[$(date)] Trying $ZONE..."
    gcloud compute tpus tpu-vm create sinkhorn-benchmark-cluster \
      --zone=$ZONE \
      --accelerator-type=v6e-8 \
      --version=tpu-ubuntu2204-base \
      --spot 2>&1

    if [ $? -eq 0 ]; then
      echo "✓ SUCCESS in $ZONE!"
      exit 0
    fi
  done
  echo "All zones full. Retrying in 5 minutes..."
  sleep 300
done
```

Run it in the background from Cloud Shell:
```bash
chmod +x provision_tpu.sh && nohup ./provision_tpu.sh > provision.log 2>&1 &
```
