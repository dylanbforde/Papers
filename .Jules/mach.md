## 2026-04-07 - [Broadcast Potential Differences in VMEM Kernels]
**Learning:** Repeated full-block `safe_exp` operations in VMEM bounded Pallas kernels (like the multi-transport-plan backward kernels) bottleneck SRAM throughput due to heavy MUFU unit usage.
**Action:** Always compute the base full-block matrix once (e.g., P22 via `safe_exp`), then derive other related transport matrices (P21, P11, P10) using O(N) broadcast multiplications of exponentiated potential differences. This minimizes redundant full-block FMA operations and significantly saves SRAM throughput.

## 2026-04-08 - [Factorized Multi-Transport Matrix Updates]
**Learning:** Explicitly computing derived transport matrices (like P21, P11, P10 from P22) via broadcasting still materializes full (128x128) blocks in VMEM, starving SRAM bandwidth.
**Action:** Apply algebraic distributive properties to multi-transport dS equations, hoisting loop-invariant exponentiated potential differences out of the inner loop and replacing multiple block materializations with a single nested broadcast multiplication.
## 2026-04-10 - [Hoist Exponentiated Differences in Backward Kernels]
**Learning:** Computing loop-invariant exponentiated potential differences (like `exp(u1 - u2)` or `exp(v1 - v2)`) inside the inner `UNROLL` loops of Pallas block-level backward kernels triggers redundant and expensive XLA JVP/MUFU instructions.
**Action:** Always manually analyze loop dependencies in `pallas_call` kernels. Hoist calculations like `jnp.exp(u1_m - u2_m)` out of inner unrolled loops when they strictly depend on the outer kernel block index. This heavily reduces redundant SRAM compute overhead.
