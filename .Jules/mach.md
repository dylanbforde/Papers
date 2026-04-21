## 2026-04-07 - [Broadcast Potential Differences in VMEM Kernels]
**Learning:** Repeated full-block `safe_exp` operations in VMEM bounded Pallas kernels (like the multi-transport-plan backward kernels) bottleneck SRAM throughput due to heavy MUFU unit usage.
**Action:** Always compute the base full-block matrix once (e.g., P22 via `safe_exp`), then derive other related transport matrices (P21, P11, P10) using O(N) broadcast multiplications of exponentiated potential differences. This minimizes redundant full-block FMA operations and significantly saves SRAM throughput.

## 2026-04-08 - [Factorized Multi-Transport Matrix Updates]
**Learning:** Explicitly computing derived transport matrices (like P21, P11, P10 from P22) via broadcasting still materializes full (128x128) blocks in VMEM, starving SRAM bandwidth.
**Action:** Apply algebraic distributive properties to multi-transport dS equations, hoisting loop-invariant exponentiated potential differences out of the inner loop and replacing multiple block materializations with a single nested broadcast multiplication.

## 2026-04-09 - [Hoisting Loop-Invariant Operations from Inner Loops]
**Learning:** In nested multi-transport dS equations, potential arrays corresponding to the block index of the outer loop (e.g., `u1_m`, `u2_m` when outer looping over `m`) are invariant to the inner unrolled loop. Leaving exponentiations of these invariant arrays inside the inner loop wastes cycles recomputing the exact same array operations repeatedly.
**Action:** Always inspect the indices of the inner unrolled loops in Pallas kernels. Hoist any operations (especially expensive ones like `jnp.exp`) that depend strictly on the outer loop block index out of the inner loop, minimizing redundant compute overhead.

## 2026-04-21 - [Hoisting Matrix Scaling out of Pallas Inner Loops]
**Learning:** XLA does not automatically factor and hoist scaling operations (like O(M*N) scaling by a factor inside an einsum or dot product) across the `pl.pallas_call` boundary or outside of custom unrolled Pallas inner loops. Leaving an operation like `jnp.einsum("md,nd->mn", q_block, k_block) * scale` inside the loop causes repeated `O(M*N)` scaling.
**Action:** Always pre-scale the query (`q`) or key (`k`) block by constant factors outside of inner loops and before computing attention dot products. This replaces `O(M*N)` inner-loop operations with a single `O(M*D)` or `O(N*D)` operation.
