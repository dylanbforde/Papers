# TPU Training Deployment Runbook

The current production TPU workflow is a two-stage Pfam pipeline:
1. Run a short Sinkhorn-only screen on Pfam shards with `scripts/tpu_sinkhorn_screen.sh`.
2. Promote the winning config into the full Pfam Sinkhorn-vs-Flash run with `scripts/tpu_auto_run.sh`.

---

## Recommended Flow

### 1. Verify project and bucket

```bash
gcloud config set project sinkhorn-neurips
gcloud storage buckets create gs://sinkhorn-neurips --location=us-east5 || true
```

### 2. Prepare or verify the Pfam shards

The TPU workflows expect precomputed Pfam shards in:

```bash
gs://sinkhorn-neurips/pfam_shards/
```

If they are missing or need to be regenerated, run the offline CPU preparation step first:

```bash
./scripts/prepare_data.sh
```

### 3. Launch the Pfam screen

This script:
- waits for `sinkhorn-screen-queue` to become active,
- uploads an exact source snapshot from your current checkout,
- runs a JAX TPU probe on the TPU VM before training,
- runs the short Sinkhorn config sweep on Pfam shards,
- syncs `logs/` and `screen_checkpoints/` to GCS before deleting the queue.

```bash
./scripts/tpu_sinkhorn_screen.sh
```

If you are launching from `command-center` and want the machine to reconcile the current queue state for you:

```bash
./scripts/tpu_screen_ensure.sh
```

Useful overrides:

```bash
MAX_HOURS_PER_RUN=0.25 ./scripts/tpu_sinkhorn_screen.sh
AUTOTUNE_MAX_PER_DEVICE_BATCH_SIZE=4 ./scripts/tpu_sinkhorn_screen.sh
RESOURCE_ID=my-screen-queue NODE_ID=my-screen-cluster ./scripts/tpu_sinkhorn_screen.sh
```

While the queue is waiting:

```bash
./scripts/tpu_queue_status.sh sinkhorn-screen-queue
```

Screen artifacts land under:

```bash
gs://sinkhorn-neurips/sinkhorn_screen/<run-id>/
```

### 4. Optional: run the fixed-budget `K` ablation

```bash
./scripts/tpu_k_ablation.sh
```

This script:
- launches the current tail-refinement ablation grid,
- verifies the JAX TPU runtime before training,
- keeps per-`K` logs and checkpoints isolated under `logs/` and `checkpoints/`.

### 5. Promote the winning config into the full run

Set the winning Sinkhorn config via environment variables, then launch the automated production run. Prefer the multi-zone race wrapper here too, so the first zone to activate hands off the already-provisioned queue/node pair to `tpu_auto_run.sh`:

```bash
SINKHORN_REFINE_STEPS=2 \
SINKHORN_EPSILON=1.0 \
SUPPORT_RECONSTRUCTION=none \
GAP_MODE=none \
TRAIN_RECONSTRUCTION_WEIGHT=1.0 \
TRAIN_SPARSE_CE_WEIGHT=0.0 \
TRAIN_MONOTONICITY_WEIGHT=0.0 \
RUN_FLASH_BASELINE=0 \
./scripts/tpu_multi_zone_race.sh ./scripts/tpu_auto_run.sh
```

This script:
- waits for a zoned `sinkhorn-queue-<zone>` winner,
- uploads the current local source snapshot,
- fetches Pfam shards from GCS,
- verifies the JAX TPU runtime before training,
- runs Sinkhorn first, then optionally Flash, with GCS checkpointing enabled.

Useful overrides:

```bash
RUN_FLASH_BASELINE=1 ./scripts/tpu_multi_zone_race.sh ./scripts/tpu_auto_run.sh
RACE_ZONES="us-east5-a us-east5-b" ./scripts/tpu_multi_zone_race.sh ./scripts/tpu_auto_run.sh
RESOURCE_ID=my-promoted-queue NODE_ID=my-promoted-node ./scripts/tpu_auto_run.sh
```

Important launch invariants:
- When `tpu_auto_run.sh` is launched from `tpu_multi_zone_race.sh`, it now requires the handed-off queued resource to already exist and already be `ACTIVE`.
- It also checks that the handed-off `RESOURCE_ID` is bound to the handed-off `NODE_ID` before doing any TPU work.
- If either invariant fails, the launcher exits instead of silently creating a second canonical queue.

Production checkpoints land under:

```bash
gs://sinkhorn-neurips/checkpoints/
```

---

## Manual Status Checks

```bash
./scripts/tpu_queue_status.sh sinkhorn-screen-queue
./scripts/tpu_queue_status.sh sinkhorn-queue
```

If you need the raw `gcloud` call:

```bash
gcloud compute tpus queued-resources describe sinkhorn-screen-queue \
  --project=sinkhorn-neurips \
  --zone=us-east5-a \
  --format="yaml(name,state.state,state.stateInitiatedTime)"
```

---

## Notes

- The automation scripts use `git archive` from your local checkout. You do not need to clone from GitHub on the TPU VM.
- `scripts/tpu_sinkhorn_screen.sh` now preserves screen logs/checkpoints before teardown. Review the synced GCS artifacts before promoting a config.
- `RUN_FLASH_BASELINE=0` is the preferred default when the goal is a promoted Sinkhorn-only run for the paper. Use `RUN_FLASH_BASELINE=1` only when you deliberately want the chained Flash baseline in the same launcher session.
- If a promoted run is already bound to an `ACTIVE` queue, verify the pairing before restarting:

```bash
gcloud compute tpus queued-resources describe sinkhorn-queue-us-east5-a \
  --project=sinkhorn-neurips \
  --zone=us-east5-a \
  --format="yaml(name,state.state,tpu.nodeSpec)"

gcloud compute tpus tpu-vm describe sinkhorn-benchmark-cluster-us-east5-a \
  --project=sinkhorn-neurips \
  --zone=us-east5-a \
  --format="yaml(name,state,queuedResource)"
```
