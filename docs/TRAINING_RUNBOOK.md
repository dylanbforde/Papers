# Wall-Clock Training Deployment Runbook

Copy-paste each step in order from **Cloud Shell**.

---

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

```bash
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=us-east5-a
```

Then from inside the TPU VM:
```bash
git clone git@github.com:dylanbforde/sinkhorn-aligner.git
```

## 5. Run Sinkhorn OT (2.5 hours, auto-shutdown after)

```bash
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=us-east5-a --worker=all --command="\
  cd sinkhorn-aligner && \
  python src/benchmarks/benchmark_wallclock_training.py --method sinkhorn_ot --max_hours 2.5; \
  sudo shutdown -h now"
```

## 6. Run Flash Attention (2.5 hours, auto-shutdown after)

First restart the VM (it shut itself down in step 5):
```bash
gcloud compute tpus tpu-vm start sinkhorn-benchmark-cluster --zone=us-east5-a
```

Then run:
```bash
gcloud compute tpus tpu-vm ssh sinkhorn-benchmark-cluster --zone=us-east5-a --worker=all --command="\
  cd sinkhorn-aligner && \
  python src/benchmarks/benchmark_wallclock_training.py --method flash_attention --max_hours 2.5; \
  sudo shutdown -h now"
```

## 7. Download results

```bash
gcloud storage cp -r gs://sinkhorn-neurips-data/results/ ./results/
gcloud storage cp -r gs://sinkhorn-neurips-data/checkpoints/ ./checkpoints/
```

## 8. Delete everything (stop paying)

```bash
gcloud compute tpus queued-resources delete sinkhorn-queue --zone=us-east5-a --force
gcloud compute tpus tpu-vm delete sinkhorn-benchmark-cluster --zone=us-east5-a --quiet
```
