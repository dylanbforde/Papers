# Google Cloud Platform (GCP) TPU Deployment Guide

This guide covers everything you need to run both the **latency benchmark** (`benchmark_long_context_alignment.py`) and the **wall-clock training experiment** (`benchmark_wallclock_training.py`) on a Google Cloud TPU v6e (Trillium) instance for the NeurIPS paper.

Because you have never used GCP before, this guide assumes zero prior cloud experience.

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

To distribute your code to all hosts in the cluster natively, you must use a GitHub **Personal Access Token (PAT)**. This allows the non-interactive `--worker=all` command to bypass browser authentication.

**CRITICAL SECURITY WARNING:** NEVER hardcode or commit your actual PAT into any scripts or markdown files in this repository. Always use placeholders like `YOUR_PAT_TOKEN` in documentation. 

1. Go to Github.com → Settings → Developer Settings → Personal Access Tokens → Generate new token (classic). Give it the `repo` scope.
2. Run this command from your Cloud Shell, placing your PAT securely into the cloning URL (replace `YOUR_GITHUB_USERNAME:YOUR_PAT_TOKEN` with your actual credentials):

```bash
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=us-east5-a --worker=all --command="git clone https://YOUR_GITHUB_USERNAME:YOUR_PAT_TOKEN@github.com/dylanbforde/sinkhorn-aligner.git"
```

Whenever updates are pushed to the repository in the future, simply pull them instantly across the cluster:
```bash
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=us-east5-a --worker=all --command="cd sinkhorn-aligner && git pull"
```

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

## 6.1 Running the Wall-Clock Training Experiment
This is the full 2.5-hour training comparison: Flash Attention vs Sinkhorn OT on real human/mouse orthologous DNA sequences. Both methods train for equal wall-clock time; we compare loss curves.

### Step 1: Pull the latest code
```bash
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=us-east5-a --worker=all --command="cd sinkhorn-aligner && git pull"
```

### Step 2: Run Sinkhorn OT training (2.5 hours)
```bash
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=us-east5-a --worker=all --command="cd sinkhorn-aligner && python src/benchmarks/benchmark_wallclock_training.py --method sinkhorn_ot --max_hours 2.5"
```

### Step 3: Run Flash Attention training (2.5 hours)
```bash
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=us-east5-a --worker=all --command="cd sinkhorn-aligner && python src/benchmarks/benchmark_wallclock_training.py --method flash_attention --max_hours 2.5"
```

### What to expect:
- **First run**: The script auto-downloads 30 orthologous gene CDS pairs from the Ensembl REST API (~2 min), embeds them with the Nucleotide Transformer, then begins training. Subsequent runs use cached data from `./data/dna_orthologs/`.
- **Batch size**: Auto-detected via binary search to maximize TPU memory utilization.
- **Sequence length**: 8,192bp by default (padded to mod-128 for Pallas).
- **Checkpoints**: Saved every 100 steps to `gs://sinkhorn-neurips-data/checkpoints/{method}/`.
- **Results**: Loss curve CSV and JSON summary saved to `gs://sinkhorn-neurips-data/results/`.
- **Total time**: ~5 hours for both methods (2.5h each, run sequentially).
- **Cost**: At ~$10–12/hr for v6e-8, expect roughly **~$50–60** for the full experiment.

### Step 4: Retrieve results
Pull the loss curves and checkpoints to your local machine:
```bash
# Loss curves and summaries
gcloud storage cp -r gs://sinkhorn-neurips-data/results/ ./results/

# Model checkpoints (optional — can be large)
gcloud storage cp -r gs://sinkhorn-neurips-data/checkpoints/ ./checkpoints/
```

### Optional flags:
| Flag | Default | Description |
|------|---------|-------------|
| `--seq_len` | `8192` | Sequence length (will be padded to mod-128) |
| `--lr` | `1e-3` | Learning rate |
| `--D_proj` | `64` | Projection dimension for Q/K/V heads |
| `--checkpoint_every` | `100` | Steps between GCS checkpoints |
| `--no_gcs` | off | Disable GCS upload (local only, for testing) |

## 6.5 Real-World Training: Epochs & Checkpointing
When you transition from running the benchmark profiling script to doing an actual full-scale training run over Pfam or Nucleotide datasets, you must implement **Checkpointing** to protect against TPU preemptions.

1. **Epochs:** Standard sequence alignment training runs for roughly **50 to 100 epochs** over the dataset. You should wrap your JAX loss in an Optax learning rate scheduler (e.g., `optax.warmup_cosine_decay_schedule`).
2. **Cloud Checkpointing:** Because Spot/Preemptible TPUs can be shut off randomly by Google (to save 70% of the cost!), you must save your model weights and optimizer state using JAX's `orbax.checkpoint` library at the end of *every single epoch*.

## 6.6 Setting Up Google Cloud Storage (GCS) & Data Retrieval
TPU VMs have very little persistent disk space. All large datasets, checkpoints, and benchmark results should be read from and written to a Google Cloud Storage (GCS) Bucket.
1. **Create the Bucket**: Run this in your local Cloud Shell:
   ```bash
   gcloud storage buckets create gs://sinkhorn-neurips-data --location=europe-west4
   ```
2. **Where is my data?**: You can visually explore, upload, and download files from this bucket anytime by visiting [console.cloud.google.com/storage](https://console.cloud.google.com/storage) in your web browser.
3. **Downloading to your laptop**: To pull all training runs from the cloud bucket to your local machine:
   ```bash
   gcloud storage cp -r gs://sinkhorn-neurips-data/results ./local_folder
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

## 8. Troubleshooting: Bouncing Capacity (v6e Trillium)
TPU Trillium models (`v6e`) are in extremely high demand globally. 

If you `stop` your machine, you release the physical hardware. When you attempt to `start` your stopped machine later, Google must find 8 available chips in your exact datacenter. If the zone is currently sold out, you will receive an `Insufficient Capacity` error.

**What to do:**
1. **Wait and Retry:** Hardware frees up identically to how it gets claimed. Run the `tpu-vm start ...` command repeatedly every 5-10 minutes. A cluster will almost always eventually catch a released pocket of availability.
2. **Delete and Move:** If you cannot wait, you must completely `delete` the stopped cluster and recreate a fresh one in a different zone that has current capacity (like `us-central1-a` or `us-east1-c`), then strictly re-run the `apt-get` installation block and clone the code again.
