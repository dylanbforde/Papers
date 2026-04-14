## 2026-04-07 - [Broadcast Potential Differences in VMEM Kernels]
**Learning:** Repeated full-block `safe_exp` operations in VMEM bounded Pallas kernels (like the multi-transport-plan backward kernels) bottleneck SRAM throughput due to heavy MUFU unit usage.
**Action:** Always compute the base full-block matrix once (e.g., P22 via `safe_exp`), then derive other related transport matrices (P21, P11, P10) using O(N) broadcast multiplications of exponentiated potential differences. This minimizes redundant full-block FMA operations and significantly saves SRAM throughput.

## 2026-04-08 - [Factorized Multi-Transport Matrix Updates]
**Learning:** Explicitly computing derived transport matrices (like P21, P11, P10 from P22) via broadcasting still materializes full (128x128) blocks in VMEM, starving SRAM bandwidth.
**Action:** Apply algebraic distributive properties to multi-transport dS equations, hoisting loop-invariant exponentiated potential differences out of the inner loop and replacing multiple block materializations with a single nested broadcast multiplication.

## 2026-04-09 - [Hoisting Loop-Invariant Operations from Inner Loops]
**Learning:** In nested multi-transport dS equations, potential arrays corresponding to the block index of the outer loop (e.g., `u1_m`, `u2_m` when outer looping over `m`) are invariant to the inner unrolled loop. Leaving exponentiations of these invariant arrays inside the inner loop wastes cycles recomputing the exact same array operations repeatedly.
**Action:** Always inspect the indices of the inner unrolled loops in Pallas kernels. Hoist any operations (especially expensive ones like `jnp.exp`) that depend strictly on the outer loop block index out of the inner loop, minimizing redundant compute overhead.

## 2024-04-14 - Pre-scaling Reference Blocks outside of Loops
**Learning:** In multi-transport and forward Pallas kernels (`dq_kernel_multi`, `dk_dv_kernel_multi`, `tile_sinkhorn_kernel`), `s_block = jnp.einsum("md,nd->mn", q, k) * scale` scales the entire resulting attention matrix inside the unrolled inner loop. This applies $O(M \times N)$ scalar multiplications, starving TPU scalar arithmetic units during hot paths.
**Action:** Always pre-scale the static outer-loop reference blocks by constants (e.g. `q_scaled = q_block * scale` or `k_scaled = k_block * scale`) outside the inner loop to replace $O(M \times N)$ inner loop multiplies with $O(M \times D)$ or $O(N \times D)$ outer loop multiplies, saving critical compute cycles without altering gradients.
