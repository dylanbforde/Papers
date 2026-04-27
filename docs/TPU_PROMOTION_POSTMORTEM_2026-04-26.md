# TPU Promotion Postmortem: 2026-04-26

## Summary

The first promoted `k2_eps1` TPU run launched through `scripts/tpu_multi_zone_race.sh` did not enter training. The race wrapper correctly provisioned `sinkhorn-queue-us-east5-a`, but `scripts/tpu_auto_run.sh` then created and waited on a second canonical queue, `sinkhorn-queue`, instead of reusing the handed-off zoned resource. This stranded an `ACTIVE` TPU while the launcher polled a different queue for hours. Billing later locked the project before the queued resources could be deleted cleanly.

This was an orchestration bug, not a TPU runtime or model bug.

## Impact

- One `v6e-8` queued resource became `ACTIVE` and remained allocated while no training was running.
- A second queued resource remained `WAITING_FOR_RESOURCES`.
- The promoted run produced no valid empirical result in that first attempt.
- Cleanup was delayed by a billing/write lock, which increased the operational risk and made direct remediation slower.

## Timeline

All timestamps UTC.

- `2026-04-25 22:57:57`: multi-zone race created `sinkhorn-queue-us-east5-a`
- `2026-04-25 22:58:04`: multi-zone race created `sinkhorn-queue-us-east5-b`
- `2026-04-25 23:02:23`: `sinkhorn-queue-us-east5-a` became `ACTIVE`
- `2026-04-25 23:02:26`: losing queue in `us-east5-b` was deleted
- `2026-04-25 23:02:29`: race wrapper handed off to `scripts/tpu_auto_run.sh`
- `2026-04-25 23:02:29` onward: `tpu_auto_run.sh` created a fresh unsuffixed `sinkhorn-queue` and waited on it instead of the already-`ACTIVE` zoned queue
- `2026-04-26`: billing/write lock prevented queue deletion and TPU SSH operations until billing was restored
- After billing was restored, `tpu_auto_run.sh` was rebound directly to the existing `sinkhorn-queue-us-east5-a` / `sinkhorn-benchmark-cluster-us-east5-a` pair and training started normally

## Root Cause

The root cause was a launcher handoff mismatch between the multi-zone wrapper and the promoted-run launcher.

`scripts/tpu_multi_zone_race.sh` handed off:

- `ZONE=us-east5-a`
- `RESOURCE_ID=sinkhorn-queue-us-east5-a`
- `NODE_ID=sinkhorn-benchmark-cluster-us-east5-a`

But `scripts/tpu_auto_run.sh` originally hardcoded:

- `ZONE="us-east5-a"`
- `RESOURCE_ID="sinkhorn-queue"`
- `NODE_ID="sinkhorn-benchmark-cluster"`

So the promoted launcher ignored the handed-off resource/node names and treated the run as a fresh single-zone launch.

## Contributing Factors

1. No invariant check on handoff reuse  
   `tpu_auto_run.sh` did not require an existing handed-off queued resource when called from the race wrapper.

2. No resource-to-node consistency check  
   The launcher did not verify that the chosen `RESOURCE_ID` was actually bound to the chosen `NODE_ID`.

3. Chained Flash baseline in the same launcher  
   `tpu_auto_run.sh` always chained `sinkhorn_ot` into `flash_attention`, which complicated operational control and encouraged ad hoc watchdog logic.

4. Cleanup depended on live write access  
   Once billing/write access was disabled, the queued resources could still be observed but not deleted from CLI.

## Detection

The incident became obvious from the contradiction between:

- GCP control plane:
  - `sinkhorn-queue-us-east5-a`: `ACTIVE`
- launcher log:
  - repeated `Status: WAITING_FOR_RESOURCES`

That mismatch is the signature of a handoff bug: an allocated TPU exists, but the launcher is polling a different resource.

## What Was Fixed

### 1. Handoff variables are now honored

`scripts/tpu_auto_run.sh` now reads:

- `PROJECT_ID="${PROJECT_ID:-...}"`
- `ZONE="${ZONE:-...}"`
- `RESOURCE_ID="${RESOURCE_ID:-...}"`
- `NODE_ID="${NODE_ID:-...}"`

instead of overwriting them unconditionally.

### 2. Race handoff now requires reuse

`scripts/tpu_multi_zone_race.sh` now exports:

- `REQUIRE_EXISTING_RESOURCE=1`
- `REQUIRE_ACTIVE_RESOURCE=1`

when it hands off the winning queue to the real launcher.

This means the promoted launcher must reuse the handed-off `ACTIVE` resource; it is not allowed to silently create a replacement queue.

### 3. Resource/node binding is checked explicitly

`scripts/tpu_auto_run.sh` now verifies:

- the existing `RESOURCE_ID` is bound to the expected `NODE_ID`
- the `ACTIVE` TPU VM reports the expected `queuedResource`

If those checks fail, the launcher exits before any TPU work starts.

### 4. Flash can now be skipped cleanly

`RUN_FLASH_BASELINE=0` now disables the chained Flash phase inside `tpu_auto_run.sh`, so a promoted Sinkhorn-only run no longer requires a watchdog kill at the phase boundary.

## What We Should Do Differently

1. Always run promoted jobs through the same zoned handoff contract  
   If a launcher can be used behind `tpu_multi_zone_race.sh`, it must treat `ZONE`, `RESOURCE_ID`, and `NODE_ID` as authoritative.

2. Fail closed on resource reuse  
   Any launcher handed an existing queue should refuse to create a new one unless the operator explicitly opts into that behavior.

3. Separate paper-critical runs from optional baselines  
   Promoted Sinkhorn runs should default to `RUN_FLASH_BASELINE=0`. The Flash baseline should be its own deliberate run.

4. Verify control-plane consistency before expensive steps  
   Before dependency install or shard download:
   - confirm queued resource state
   - confirm node name
   - confirm TPU VM `queuedResource`

5. Treat `ACTIVE` + polling mismatch as a stop signal  
   If GCP shows an `ACTIVE` queue but the launcher log still prints `WAITING_FOR_RESOURCES`, stop and inspect immediately.

## Operational Guardrails

Before reusing an active promoted queue, verify:

```bash
gcloud compute tpus queued-resources describe "$RESOURCE_ID" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --format="yaml(name,state.state,tpu.nodeSpec)"

gcloud compute tpus tpu-vm describe "$NODE_ID" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --format="yaml(name,state,queuedResource)"
```

For promoted Sinkhorn-only runs, prefer:

```bash
RUN_FLASH_BASELINE=0 ./scripts/tpu_multi_zone_race.sh ./scripts/tpu_auto_run.sh
```

## Paper Handling

The invalid first promoted attempt should not be counted as empirical evidence. The valid empirical basis remains:

- the completed `2026-04-25` short screen
- any later promoted run that actually reaches training under the corrected launcher path

Do not blend the failed handoff attempt into the empirical narrative.
