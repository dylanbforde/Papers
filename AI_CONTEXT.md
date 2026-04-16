# AI Context & General Development Environment

## Environment 
- **Host OS**: NixOS
- **Local Dev Environment**: Distrobox container named `devbox` (Ubuntu). Use this if standard Linux tools/environments are explicitly required locally.
- **Compute Target**: 
  - Immediate/Local testing in `devbox`.
  - Final execution and multi-GPU/multi-TPU training will be done on **GCP (Google Cloud Platform)**. 

## Project Goals
- **Refinement for NeurIPS Acceptance**: Fix the theoretical, empirical, and presentation issues highlighted in the paper's reviews.
- **Training from Scratch**: We will conduct full training runs to correct issues natively, ensuring no shortcuts are taken in gathering empirical data.
- **Benchmarking**: Establish a clean repository structure specifically tailored for standardized, reproducible benchmarking against baselines like DEDAL and PLASMA.
- **Reproducibility**: Need to re-run figures and tables to ensure they correctly match the updated experimental setups and hardware (e.g., TPU v6e-1).

## Key Context Points
- Ensure that memory and latency comparisons are properly characterized, explicitly documenting tradeoffs (like the Sinkhorn 15-iteration latency).
- Avoid conflating approximations (e.g., truncated k=2 unrolling) with exact solutions in descriptions.
- Use multi-seed runs for all statistical claims.
- Any new implementation should seamlessly support scaling out to TPU/GPU pods.
- Treat the **Pfam shard pipeline** (`benchmark_pfam_training.py`, `scripts/tpu_sinkhorn_screen.sh`, `scripts/tpu_auto_run.sh`) as the current production TPU benchmark path. The ortholog-DNA wall-clock script is retained as a legacy comparator, not the primary training workflow.

## GCP / TPU Operations
- The practical control plane is the `command-center` GCE VM in `us-east5-a`; use it to launch and monitor TPU queued resources.
- On `command-center`, `gcloud` may default back to the VM service account (`83568222167-compute@developer.gserviceaccount.com`), which does not necessarily have access to the TPU project. Before TPU work, explicitly run:
  - `gcloud auth login`
  - `gcloud config set account forde.dylan@gmail.com`
  - `gcloud config set project sinkhorn-neurips`
  - `gcloud config set compute/zone us-east5-a`
- `scripts/tpu_sinkhorn_screen.sh`, `scripts/tpu_auto_run.sh`, and `scripts/tpu_k_ablation.sh` now read the active account via `gcloud config get-value account`, not `gcloud auth list --filter=status:ACTIVE`.
- Queued resources are zonal. Canonical screen resource: `sinkhorn-screen-queue` in `us-east5-a`. Use `scripts/tpu_queue_status.sh sinkhorn-screen-queue` or direct `gcloud compute tpus queued-resources describe ... --zone=us-east5-a`.
- Older launcher behavior could delete a queued TPU on any shell exit because of `trap cleanup EXIT`. Current TPU launchers preserve resources that are still `WAITING_FOR_RESOURCES` by default. Set `DELETE_WAITING_QUEUE_ON_EXIT=1` only if deliberate teardown while waiting is desired.
- Preferred detached launch pattern on `command-center`: `bash scripts/start_tpu_job_detached.sh scripts/tpu_sinkhorn_screen.sh` (or `... tpu_k_ablation.sh`, `... tpu_auto_run.sh`). This wrapper uses `nohup`, writes a log under `logs/`, and persists across SSH disconnects without requiring an attached terminal multiplexer.
- For TPU scarcity, prefer the multi-zone race wrapper `scripts/tpu_multi_zone_race.sh`. Start with same-region zones (`us-east5-a us-east5-b`) so the first `ACTIVE` zone wins and the losers are deleted before handoff to the real launcher. This reduces queue latency without immediately introducing cross-region data locality issues.
- `tmux` is optional, not required. If using `tmux` on `command-center`, export `TERM=xterm-256color` first because `xterm-kitty` is not recognized there.
- The screen launcher log already includes queue status while the job is in Step 2 (`WAITING_FOR_RESOURCES` / `ACTIVE`), so separate queue polling is only needed for independent verification or after launcher disconnects.

## TPU Workflow Order
- Preferred order is staged, not monolithic:
  1. `scripts/tpu_sinkhorn_screen.sh` to validate queue behavior, dependency install, source upload, artifact sync, and shortlist promising Sinkhorn configs.
  2. `scripts/tpu_k_ablation.sh` only after the screen is healthy and there is a concrete reason to compare `K` values more systematically.
  3. `scripts/tpu_auto_run.sh` for the promoted longer run once the configuration is chosen.
- Do not chain screen + k-ablation + full production in one fragile launcher until the TPU control-plane behavior is stable. Operationally, the screen is the right first script because it exercises the full path at lower cost and shorter wall-clock.
