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
- The **central theoretical contribution** for the unified NeurIPS paper should remain the balanced fixed-support `R=2` tail-refinement adjoint and one-reference-tile backward schedule. In the NeurIPS draft, treat it as the signature implementation result rather than over-marketing the raw row/column orbit identity as a deep theorem. Everything else should be justified relative to it:
  - dustbin augmentation is the minimum theorem bridge to the current cloud experiments;
  - local surrogate-bias analysis supports the stopped-base construction;
  - true unbalanced OT is currently a secondary, dense-reference theorem layer until there is a scalable implementation;
  - support geometry and multimodal routing stay out of the central contribution unless they later become both theorem-level and empirically necessary.
- Do **not** submit `innovation_track.tex` as a second arXiv preprint right now. Keep it as the internal staging draft for the unified NeurIPS paper until the dustbin-augmented theorem and the cloud results are mature enough to merge cleanly.
- Current recommendation: aim for **one unified NeurIPS paper**, not two overlapping NeurIPS papers. The combined paper should contain:
  - the balanced fixed-support tail adjoint and one-reference-tile schedule from the arXiv preprint;
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
- As of **2026-04-27**, the NeurIPS draft should describe the promoted run conservatively: it improves held-out reconstruction and sparse CE relative to step 0, but the monotonicity violation metric does not improve and sparse CE was diagnostic rather than optimized (`lambda_ce=0`). Do not describe this as downstream alignment-quality evidence.
- The highest-value mathematical strengthening path is to replace the assumed contraction constant in the surrogate-bias theorem with either a projective-metric Sinkhorn contraction certificate or an a posteriori cotangent-decay certificate. True KL-unbalanced OT remains secondary unless it becomes the implemented cloud path.
- As of **2026-04-27**, the first a posteriori certificate layer is implemented in `scripts/validate_tail_bias_certificate.py` and drafted into `paper_neurips/neurips_body.tex`. It proves/checks that the full-vs-stopped gradient remainder is exactly `D_x z_T^T eta_R`, where `eta_R` is the measured base-state tail cotangent. The local validation shows certificate residuals at numerical precision (`<= 2.5e-10` at `L=128`, seeds `0,1,2`) and rapid cotangent/bias decay through `R=2`. With max-absolute tolerance `1e-5`, the certified tail-depth selector chooses `R=2` on all three local seeds; in the paper this is now stated as the cheapest certified tail depth for any monotone per-depth implementation cost over the tested candidate set.
- `scripts/validate_orbit_factorization.py` validates the stronger transport-orbit calculus: all base, tail, and certificate half-step plans live in the row/column scaling orbit of one reference plan, and plan-weighted score cotangent sums can be accumulated by streaming one reference transport tile with row/column orbit coefficients. At `T=15`, `R=2`, `L=128`, the max log-domain reconstruction error across all `2(T+R)` checked plans is `1.91e-6`.
- As of **2026-04-28**, two additional reviewer-facing diagnostics are implemented and drafted into Appendix C of `paper_neurips/neurips_body.tex`:
  - `scripts/benchmark_four_plan_adjoint.py` isolates the `R=2` score-adjoint stage after potentials are available. On local dense-JAX CPU at `W=256`, the one-reference representation matches direct four-plan score cotangents to `<=4.8e-7`, reduces logical active plan-factor storage by exactly `4x`, and gives mean post-compilation score-adjoint speedups of `2.20x` (`L=512`), `1.29x` (`L=1024`), and `1.53x` (`L=2048`). Treat this as a controlled materialization diagnostic, not a TPU or FlashSinkhorn wall-clock claim.
  - `scripts/compute_pfam_contraction_certificate.py` computes production-checkpoint Birkhoff-Hopf certificates on the local Pfam test shards. Over `100` held-out examples and `335` strictly positive active blocks at `epsilon=1`, the promoted step-1437 checkpoint has median exact projective `rho_H=0.081`, P95 `0.279`, and max `0.492`; the coarser range bound has median `0.267` and max `0.673`. This directly addresses reviewer concern that the contraction certificate was uncalibrated on the Pfam distribution.
- Rebuttal-use theorem direction for reviewer concerns:
  - Do **not** claim empirical wall-clock dominance over FlashSinkhorn without benchmarks. The defensible claim is a structural IO separation for the differentiated staircase object: any flash-style implementation that treats the `m` backward staircase/certificate plans independently needs `m` transport-normalization/streaming passes, whereas the transport-orbit calculus streams one reference plan tile and accumulates all plan-weighted cotangent terms through row/column orbit coefficients. For `R=2`, `m=4`.
  - Even this should be phrased carefully: say "a direct `R=2` adjoint must evaluate four distinct staircase plan factors" rather than "must materialize four plans." A clever baseline could stream/recompute the four factors without storing them simultaneously. The precise paper claim is that the four factors lie in one row/column scaling orbit, so each block can be evaluated from one reference transport tile plus vector modifiers, avoiding separate plan-tile formation for the remaining three terms.
  - To address the unverified Euclidean contraction assumption, replace or supplement the local `rho < 1` assumption with a projective Hilbert-metric certificate via Birkhoff-Hopf. Under the paper's scaled-score convention, for a strictly positive active kernel `K = exp(S)` on the balanced or dustbin-augmented support, the two-half-step Sinkhorn map contracts with
    `rho_H = tanh(Delta(K)/4) tanh(Delta(K^T)/4) < 1`,
    where `Delta` is projective diameter. If the active block score range is `Omega = max S - min S`, then a computable sufficient bound is
    `rho_H <= tanh^2(Omega / (2 epsilon))`.
  - This projective statement is mathematically stronger than an unverified local Euclidean assumption, but it should be phrased carefully: it guarantees contraction in Hilbert/oscillation metric on strictly positive active blocks, not arbitrary Euclidean contraction on masked supports with disconnected zero entries.
- As of **2026-04-27**, the NeurIPS draft now incorporates that rebuttal path directly: the introduction explicitly separates FlashSinkhorn-style IO-aware forward solving from the paper's differentiated backward-staircase object, and Section 2 includes a Birkhoff-Hopf projective contraction certificate with the strict-positivity/masked-support caveat. The theorem uses the paper's scaled score convention `K=exp(S)` and states the unscaled-logit equivalent `rho_H <= tanh^2(Omega_raw / (2 epsilon))`.
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
- As of **2026-04-25**, the short balanced TPU screen has now completed end-to-end for:
  - `k2_eps1`
  - `k4_eps1`
  on a TPU v6e-8 slice with `pairs=50000`, `T=15`, `band_width=1024`, and global batch size `64`. At step 50, the two runs were effectively identical (`loss ≈ 3.5778/3.5779`, `CE ≈ 5.3223`, `mono ≈ 0.3353`), with `k2_eps1` slightly faster (`8.46 ex/s` vs `8.38 ex/s`). Treat this as the first real systems-facing empirical support for the central `R=2` claim.
- A later **2026-04-25** rerun completed the two core dustbin-augmented configs as well:
  - `muot_dustbin_mono`
  - `muot_dustbin_hybrid`
  Step-50 snapshots were:
  - `muot_dustbin_mono`: `loss 3.9030`, `CE 5.3258`, `mono 0.4155`, `8.41 ex/s`
  - `muot_dustbin_hybrid`: `loss 4.6250`, `CE 5.3244`, `mono 0.4162`, `8.45 ex/s`
  This means the four core screen configs that matter most for the unified paper have all now completed successfully on TPU.
- The first promoted long-budget `tpu_auto_run.sh` attempt on **2026-04-26** did **not** start training successfully. Root cause: when launched through `scripts/tpu_multi_zone_race.sh`, `tpu_auto_run.sh` still used its canonical unsuffixed `ZONE` / `RESOURCE_ID` / `NODE_ID` defaults instead of the handed-off suffixed winner names, so it left `sinkhorn-queue-us-east5-a` active while polling a separate `sinkhorn-queue`. Treat that first promoted attempt as invalid; it does not contribute empirical results.
- A project billing/write lock then blocked cleanup and TPU-VM SSH temporarily. Billing was later restored and the valid recovery path was to rebind `tpu_auto_run.sh` directly to the already-active suffixed resource/node pair instead of launching another queue.
- Current launcher safeguards after that incident:
  - `scripts/tpu_auto_run.sh` now honors handed-off `PROJECT_ID` / `ZONE` / `RESOURCE_ID` / `NODE_ID` env overrides and logs them at startup;
  - `scripts/tpu_multi_zone_race.sh` now exports `REQUIRE_EXISTING_RESOURCE=1` and `REQUIRE_ACTIVE_RESOURCE=1` when handing off a winner;
  - `scripts/tpu_auto_run.sh` now verifies that the queued resource is already present, already `ACTIVE`, and bound to the expected node before doing any TPU work;
  - `RUN_FLASH_BASELINE=0` now cleanly skips the chained Flash phase for promoted Sinkhorn-only runs.
- Canonical written postmortem for this incident: `docs/TPU_PROMOTION_POSTMORTEM_2026-04-26.md`. Use it as the operational reference for future promoted runs.
- The same successful rerun still failed later on `narrow128_diag`. That failure was a TPU VMEM overrun in the narrower-band ablation path, not in the balanced+dustbin core screen. The current production understanding is:
  - the theorem-level method remains `tail_refinement` with `R=2`;
  - the fused `K=2` backward is only appropriate for genuinely banded regimes;
  - full-support and near-full-support cases, including the dustbin-augmented screen path, must use the orbit-factorized fixed-`K` implementation instead;
  - the remaining unstable branch is the narrow-band/scout ablation path, which is no longer on the critical path for the main paper.
- Current code implements this via `_use_fused_k2_fast_path(...)` in `src/ot_kernel.py`, with the near-full-support guard `band_blocks + 1 < num_blocks`. This is now part of the production TPU policy and should be reflected in the unified NeurIPS draft as an implementation detail, not a theorem change.
- `tmux` is optional, not required. If using `tmux` on `command-center`, export `TERM=xterm-256color` first because `xterm-kitty` is not recognized there.
- The screen launcher log already includes queue status while the job is in Step 2 (`WAITING_FOR_RESOURCES` / `ACTIVE`), so separate queue polling is only needed for independent verification or after launcher disconnects.
- Current stable recovery pattern: if `sinkhorn-screen-queue` already exists in `us-east5-a` and is still `WAITING_FOR_RESOURCES`, do not replace it with new queue experiments. Kill any stray helper wrapper, keep the canonical queue, and restart only the detached `scripts/tpu_sinkhorn_screen.sh` launcher so it reuses the existing queue.
- For promoted runs, the default paper-facing launch should now be:
  - `RUN_FLASH_BASELINE=0 ./scripts/tpu_multi_zone_race.sh ./scripts/tpu_auto_run.sh`
  This keeps the handoff invariant checks, avoids chained Flash by default, and makes the promoted Sinkhorn run a standalone artifact.

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
- Given the completed **2026-04-25** screen, the operational recommendation is now:
  - do **not** rerun the whole screen immediately;
  - do **not** block the paper on fixing `narrow128_diag` / scout ablations;
  - use the completed **2026-04-25** short screen as the present empirical basis for paper consolidation;
  - promotion priority is the best balanced candidate (`k2_eps1`) first, then the best dustbin candidate (currently `muot_dustbin_hybrid`, slightly better CE than `muot_dustbin_mono`);
  - if billing or project write access ever locks again, stop launching new queues immediately and reconcile any already-`ACTIVE` resource before attempting recovery.
