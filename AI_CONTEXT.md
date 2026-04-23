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

## Publication Strategy
- The current arXiv preprint is the methods-first balanced paper in `paper_neurips/main.tex` and was submitted on **2026-04-22**. It is intentionally narrow: balanced fixed-support tail refinement, exact `R=2` staircase factorization, centered-vs-ungauged equivalence, and exactness validation.
- The follow-up draft lives in `paper_neurips/innovation_track.tex`. It is not a separate conference paper plan right now; it is the theory/program draft for the next layer of the **same overall NeurIPS story**.
- The **central theoretical contribution** for the unified NeurIPS paper should remain the balanced fixed-support `R=2` tail-refinement factorization. That is the signature theorem. Everything else should be justified relative to it:
  - dustbin augmentation is the minimum theorem bridge to the current cloud experiments;
  - local surrogate-bias analysis supports the stopped-base construction;
  - true unbalanced OT is currently a secondary, dense-reference theorem layer until there is a scalable implementation;
  - support geometry and multimodal routing stay out of the central contribution unless they later become both theorem-level and empirically necessary.
- Do **not** submit `innovation_track.tex` as a second arXiv preprint right now. Keep it as the internal staging draft for the unified NeurIPS paper until the dustbin-augmented theorem and the cloud results are mature enough to merge cleanly.
- Current recommendation: aim for **one unified NeurIPS paper**, not two overlapping NeurIPS papers. The combined paper should contain:
  - the balanced fixed-support theorem from the arXiv preprint;
  - the dustbin-augmented / gap-aware extension that matches the current TPU runs;
  - the full systems results and Pfam empirical results once the cloud runs complete.
- Working publication flow:
  1. keep `paper_neurips/main.tex` as the public narrow arXiv record;
  2. continue theorem and validation development in `paper_neurips/innovation_track.tex`;
  3. finish the dustbin-augmented theorem, surrogate-bias analysis, and TPU empirical runs;
  4. merge the balanced theorem, dustbin-augmented theorem, and the final empirical sections into one NeurIPS manuscript;
  5. only then decide whether to post a major arXiv version update/replacement aligned with the unified paper.
- Do **not** try to promote soft support geometry and multimodal routing into the main NeurIPS contribution unless they become both theorem-level and empirically justified. For now they remain sequenced follow-up extensions after the dustbin-augmented theorem.

## Key Context Points
- Ensure that memory and latency comparisons are properly characterized, explicitly documenting tradeoffs (like the Sinkhorn 15-iteration latency).
- Avoid conflating approximations (e.g., truncated k=2 unrolling) with exact solutions in descriptions.
- Use multi-seed runs for all statistical claims.
- Any new implementation should seamlessly support scaling out to TPU/GPU pods.
- Treat the **Pfam shard pipeline** (`benchmark_pfam_training.py`, `scripts/tpu_sinkhorn_screen.sh`, `scripts/tpu_auto_run.sh`) as the current production TPU benchmark path. The ortholog-DNA wall-clock script is retained as a legacy comparator, not the primary training workflow.
- The current innovation-track theorem should be grounded in the **existing** `dustbin_block` implementation, not in a hypothetical future UOT solver. The live code path already appends one active dustbin token inside an extra 128-token block and then runs the same balanced tail-refinement solver on the augmented state space.
- This means the current cloud `dustbin_block` runs are already relevant to the follow-up theory: they are not merely engineering probes, but the natural first experiments for the dustbin-augmented theorem and the eventual unified NeurIPS paper.
- Be precise about the current cloud loss: the `muot_dustbin_*` objectives are **not** purely tail-refinement objectives. The reconstruction term uses the dustbin-augmented tail surrogate, while sparse CE and barycenter-monotonicity are computed from the exact augmented `T`-step base solve. Do not describe the full objective as a pure fixed-depth surrogate.
- Canonical local verifier for the merged dustbin theorem layer: `scripts/validate_dustbin_augmentation.py`. It now checks
  1. gap-mode vs manual-augmentation equality for reconstruction-only, `muot_dustbin_mono`, and `muot_dustbin_hybrid` objective mixes;
  2. numerical single-base-plan factorization on the augmented staircase;
  3. tail-refinement vs exact-autodiff gradient agreement on the reduced-length local dustbin problem (`L=128`, `T=15`, `R=2`, cloud-aligned augmentation settings).
- There is now also a dense fixed-support true-UOT prototype theorem/validator layer:
  - dense reference module in `src/unbalanced_ot.py`
  - theorem draft in `paper_neurips/innovation_track.tex`
  - numeric verifier in `scripts/validate_unbalanced_tail_refinement.py`
  It validates the KL-penalized fixed-support `R=2` tail adjoint and single-base-plan reduction on a small dense problem. This is **not** yet the current cloud path; treat it as the next theorem layer beyond the present `dustbin_block` production route.
- There is also a local bridge script, `scripts/compare_dustbin_and_true_unbalanced.py`, for comparing the current cloud-style `dustbin_block` path against the dense true-unbalanced reference on the same small problem. Current reading:
  - the two are **not** equivalent in general;
  - at moderate unbalanced strength (`tau_q=tau_k=1`) the gap is large;
  - in the near-balanced regime (`tau_q=tau_k=20` to `100`) the gap narrows materially but remains nonzero.
  Use this as evidence that `dustbin_block` is a production bridge, not a theorem-identical replacement for true KL-unbalanced OT.

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
- Because the detached launcher runs under `nohup` on `command-center`, a local laptop sleeping or an SSH disconnect does not stop the TPU launcher. `tmux` is therefore optional for unattended runs. This does not protect against the `command-center` VM itself rebooting or being stopped.
- For TPU scarcity, prefer the multi-zone race wrapper `scripts/tpu_multi_zone_race.sh`. Start with same-region zones (`us-east5-a us-east5-b`) so the first `ACTIVE` zone wins and the losers are deleted before handoff to the real launcher. This reduces queue latency without immediately introducing cross-region data locality issues.
- The screen launcher's `SCREEN_CONFIGS` already includes the two gap-aware candidates that matter for the follow-up theorem:
  - `muot_dustbin_mono`
  - `muot_dustbin_hybrid`
  Both use `gap_mode=dustbin_block`, `support_reconstruction=none`, and `sinkhorn_gradient_mode=tail_refinement`.
- `tmux` is optional, not required. If using `tmux` on `command-center`, export `TERM=xterm-256color` first because `xterm-kitty` is not recognized there.
- The screen launcher log already includes queue status while the job is in Step 2 (`WAITING_FOR_RESOURCES` / `ACTIVE`), so separate queue polling is only needed for independent verification or after launcher disconnects.
- Current stable recovery pattern: if `sinkhorn-screen-queue` already exists in `us-east5-a` and is still `WAITING_FOR_RESOURCES`, do not replace it with new queue experiments. Kill any stray helper wrapper, keep the canonical queue, and restart only the detached `scripts/tpu_sinkhorn_screen.sh` launcher so it reuses the existing queue.

## TPU Workflow Order
- Preferred order is staged, not monolithic:
  1. `scripts/tpu_sinkhorn_screen.sh` to validate queue behavior, dependency install, source upload, artifact sync, and shortlist promising Sinkhorn configs.
  2. `scripts/tpu_k_ablation.sh` only after the screen is healthy and there is a concrete reason to compare `K` values more systematically.
  3. `scripts/tpu_auto_run.sh` for the promoted longer run once the configuration is chosen.
- Do not chain screen + k-ablation + full production in one fragile launcher until the TPU control-plane behavior is stable. Operationally, the screen is the right first script because it exercises the full path at lower cost and shorter wall-clock.
- For the publication plan, interpret the workflow as:
  1. screen balanced and dustbin-augmented candidates;
  2. use `K` ablation to support the unified NeurIPS method section;
  3. promote the best balanced and/or dustbin-augmented candidate into the full long-budget run that will supply the empirical backbone of the combined paper.
