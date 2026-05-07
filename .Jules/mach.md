## 2026-04-07 - [Broadcast Potential Differences in VMEM Kernels]
**Learning:** Repeated full-block `safe_exp` operations in VMEM bounded Pallas kernels (like the multi-transport-plan backward kernels) bottleneck SRAM throughput due to heavy MUFU unit usage.
**Action:** Always compute the base full-block matrix once (e.g., P22 via `safe_exp`), then derive other related transport matrices (P21, P11, P10) using O(N) broadcast multiplications of exponentiated potential differences. This minimizes redundant full-block FMA operations and significantly saves SRAM throughput.

## 2026-04-08 - [Factorized Multi-Transport Matrix Updates]
**Learning:** Explicitly computing derived transport matrices (like P21, P11, P10 from P22) via broadcasting still materializes full (128x128) blocks in VMEM, starving SRAM bandwidth.
**Action:** Apply algebraic distributive properties to multi-transport dS equations, hoisting loop-invariant exponentiated potential differences out of the inner loop and replacing multiple block materializations with a single nested broadcast multiplication.

## 2026-04-09 - [Hoisting Loop-Invariant Operations from Inner Loops]
**Learning:** In nested multi-transport dS equations, potential arrays corresponding to the block index of the outer loop (e.g., `u1_m`, `u2_m` when outer looping over `m`) are invariant to the inner unrolled loop. Leaving exponentiations of these invariant arrays inside the inner loop wastes cycles recomputing the exact same array operations repeatedly.
**Action:** Always inspect the indices of the inner unrolled loops in Pallas kernels. Hoist any operations (especially expensive ones like `jnp.exp`) that depend strictly on the outer loop block index out of the inner loop, minimizing redundant compute overhead.

## 2026-04-10 - [Hoisting Matrix Scaling out of Inner Loops]
**Learning:** XLA inside Pallas loops does not auto-hoist simple dot-product scaling. Pre-scaling a matrix block outside of the loop yields tangible inner-loop FLOP reduction without memory overhead.
**Action:** Always pre-scale base tensor blocks outside of inner loops to avoid O(M*N) elementwise scalar multiplications during each iteration.
