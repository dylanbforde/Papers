# Wall-Clock Training Deployment Runbook

Copy-paste each step in order from **Cloud Shell**.

---

## 0. Initial Setup (Run Once)

Set your project context and ensure the bucket exists:
```bash
# Set your project (replace with your actual project ID)
gcloud config set project [YOUR_PROJECT_ID]

# Create the bucket (if not already created)
gcloud storage buckets create gs://sinkhorn-neurips-data --location=us-east5
```

## 1. Queue the TPU (auto-provisions when capacity is available)

```bash
gcloud compute tpus queued-resources create sinkhorn-queue \
  --node-id=sinkhorn-benchmark-cluster \
  --zone=us-east5-a \
  --accelerator-type=v6e-8 \
  --runtime-version=tpu-ubuntu2204-base \
  --spot
```

## 2. Check status (repeat until ACTIVE)

```bash
gcloud compute tpus queued-resources describe sinkhorn-queue --zone=us-east5-a --format="value(state.state)"
```

## 3. Install dependencies

```bash
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=us-east5-a --worker=all --command="sudo apt-get update && sudo apt-get install python3-pip -y && pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && pip install optax transformers torch"
```

## 4. SSH in and clone repo

**Important**: Make sure your local SSH agent is running and has your GitHub key added so it can be forwarded to the TPU VM.
```bash
# On your local machine (Cloud Shell), start agent and add your key
eval `ssh-agent -s`
ssh-add ~/.ssh/id_ed25519  # Replace with your actual key path
```

Now SSH into the TPU VM with agent forwarding enabled:
```bash
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=us-east5-a --ssh-flag="-A"
```

Then from inside the TPU VM:
```bash
git clone git@github.com:dylanbforde/sinkhorn-aligner.git
```

## 5. Run Full Experiment (5 hours total)

**Why chain them?** Chaining both methods into one command ensures you hold the TPU hardware for the full 5 hours. If you run them separately and the machine shuts down, you might not be able to get it back due to capacity limits.

```bash
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=us-east5-a --worker=all --command="\
  cd sinkhorn-aligner && \
  python src/benchmarks/benchmark_wallclock_training.py --method sinkhorn_ot --max_hours 2.5 && \
  python src/benchmarks/benchmark_wallclock_training.py --method flash_attention --max_hours 2.5; \
  sudo shutdown -h now"
```
*The command will run Sinkhorn OT first, then Flash Attention, then shut down the VM automatically.*

## 6. Download results & checkpoints

Once the VM is shut down (SSH will disconnect), pull everything to your local machine:

```bash
# Pull results (CSV loss curves and JSON summaries)
gcloud storage cp -r gs://sinkhorn-neurips-data/results/ ./results/

# Pull checkpoints (optional)
gcloud storage cp -r gs://sinkhorn-neurips-data/checkpoints/ ./checkpoints/
```

## 7. Delete the queue & cluster (stop all charges)

```bash
# Delete the queued resource
gcloud compute tpus queued-resources delete sinkhorn-queue --zone=us-east5-a --force

# Delete the TPU VM itself
gcloud compute tpus tpu-vm delete sinkhorn-benchmark-cluster --zone=us-east5-a --quiet
```
