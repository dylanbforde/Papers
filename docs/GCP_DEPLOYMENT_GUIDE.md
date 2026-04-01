# Google Cloud Platform (GCP) TPU Deployment Guide

This guide covers everything you need to run the `benchmark_esm2_alignment.py` script on a Google Cloud TPU v6e (Trillium) instance, which you referenced in your NeurIPS paper. 

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

In the Cloud Shell, run the following command to create your multi-TPU VM cluster in an optimal zone (e.g., `europe-west4-a`):

```bash
gcloud compute tpus tpu-vm create sinkhorn-benchmark-cluster \
  --zone=europe-west4-a \
  --accelerator-type=v6e-8 \
  --version=tpu-ubuntu2204-base
```
*Note: `v6e-8` gives you 8 interconnected Trillium chips. Larger sizes like `v6e-16`, `v6e-32` span multiple physical host machines automatically. By requesting this slice, you are provisioning a true distributed cluster.*

## 4. Connecting to the Instance & Code Transfer
Once the instance says `READY`, you can securely SSH into the primary machine of your cluster using the Cloud Shell:

```bash
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=europe-west4-a
```

Now you are inside the leader TPU machine. To execute standard code across all hosts in a pod simultaneously (which is required if using slices larger than 8 chips), you use the `--worker=all` flag with your `gcloud` commands.

To distribute your code to all hosts in the cluster, use standard git cloning across all workers:
```bash
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=europe-west4-a --worker=all --command="git clone https://github.com/your-username/sinkhorn-aligner.git"
```

## 5. Environment Setup (The Magic JAX Command)
Google Cloud TPU VMs come with Ubuntu, but you must install the **TPU-specific version of JAX** so the library knows how to compile the Pallas kernels for the distributed mesh. You must execute this setup across **ALL** workers in the pod.

Run these exact commands in your local Cloud Shell (not SSH'd inside a single host):

```bash
# Combine the entire installation block into one multi-worker execution command
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=europe-west4-a --worker=all --command="sudo apt-get update && sudo apt-get install python3-pip -y && pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && pip install optax transformers torch"
```

## 6. Running the Distributed Benchmark
Everything is ready! We baked `jax.distributed.initialize()` into the `benchmark_esm2_alignment.py` script so it automatically synchronizes the JAX mesh across the entire TPU pod slice. Because it executes via SPMD (Single Program, Multiple Data), you just run the standard Python command across all workers simultaneously.

```bash
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=europe-west4-a --worker=all --command="cd sinkhorn-aligner && python src/benchmarks/benchmark_long_context_alignment.py --model_name 'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species' --vocab dna"
```

### What to expect:
- **Time to run**: The whole script takes about **5-10 minutes**. The longest part is JAX spending ~3 minutes on "XLA Compilation" to physically compile your Pallas kernels to the TPU architecture; the actual tensor math finishes in seconds.
- **Cost**: A `v6e-8` Trillium pod costs roughly **~$10 - $12 per hour** on demand. Because the benchmark finishes in 10 minutes, your actual cost to generate the NeurIPS tables will be barely **~$2.00**, which Google's $300 free tier will easily cover.
- All TPU hosts will independently spin up and coordinate their device ranks to form a global mesh (e.g., 8 chips globally).
- The ESM-2 weights will download, and JAX will collectively shard the protein context layers across the chip memories, allowing scaling that is physically impossible on a single chip.
- The logs will unify, printing out your benchmark matrix and dumping the hardware telemetry to `/tmp/tensorboard_profiles_DATE/`. To pull the logs back to your computer:
  ```bash
  gcloud compute tpus tpu-vm scp sinkhorn-benchmark-cluster:/tmp/tensorboard_profiles_DATE/ ./local_profiles/ --zone=europe-west4-a
  ```

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
gcloud compute tpus tpu-vm stop sinkhorn-benchmark-cluster --zone=europe-west4-a

# To completely delete it forever (stops ALL billing)
gcloud compute tpus tpu-vm delete sinkhorn-benchmark-cluster --zone=us-central1-a
```
