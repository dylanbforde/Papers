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
