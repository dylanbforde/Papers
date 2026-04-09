## 2026-04-07 - [Broadcast Potential Differences in VMEM Kernels]
**Learning:** Repeated full-block `safe_exp` operations in VMEM bounded Pallas kernels (like the multi-transport-plan backward kernels) bottleneck SRAM throughput due to heavy MUFU unit usage.
**Action:** Always compute the base full-block matrix once (e.g., P22 via `safe_exp`), then derive other related transport matrices (P21, P11, P10) using O(N) broadcast multiplications of exponentiated potential differences. This minimizes redundant full-block FMA operations and significantly saves SRAM throughput.

## 2026-04-08 - [Factorized Multi-Transport Matrix Updates]
**Learning:** Explicitly computing derived transport matrices (like P21, P11, P10 from P22) via broadcasting still materializes full (128x128) blocks in VMEM, starving SRAM bandwidth.
**Action:** Apply algebraic distributive properties to multi-transport dS equations, hoisting loop-invariant exponentiated potential differences out of the inner loop and replacing multiple block materializations with a single nested broadcast multiplication.

## 2026-04-09 - [Hoisting Loop-Invariant Potential Differences in VMEM Kernels]
**Learning:** During multi-transport-plan backward evaluation within a `pl.pallas_call`, exponentiated potential differences mapped to constant outer dimensions (`m` in `dq_kernel_multi`, and `n` in `dk_dv_kernel_multi`) are evaluated inside the inner loop, creating redundant MUFU allocations and stealing memory bandwidth.
**Action:** Always verify variables sliced from matrices mapped to the parallel outer dimension and evaluate their operations (like `jnp.exp(u1_m - u2_m)`) before the inner iteration loop, effectively hoisting them to skip redundant computations.
