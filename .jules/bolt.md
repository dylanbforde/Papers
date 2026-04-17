## 2024-05-24 - Faster Tensor Packing with jnp.pad
**Learning:** When packing multiple tensors together where some channels need to be padded with zeros (e.g. for hardware alignment mod-8), using `jnp.stack` with multiple `jnp.zeros_like()` calls forces JAX to allocate large explicitly zeroed arrays. Using `jnp.stack` only on active variables and then padding the last dimension with `jnp.pad(active, ((0, 0), ..., (0, N)))` bypasses these allocations and translates to much faster slice-update ops in XLA (up to ~1.8x faster on CPU during JIT).
**Action:** When creating mod-N padded tensor structures for kernels, always prefer `jnp.pad` over explicit `jnp.zeros_like` stacking.

## 2024-05-25 - Tensor Pre-scaling for Kernel Optimization
**Learning:** In attention and kernel computation, when evaluating dense blocks such as `jnp.einsum("md,nd->mn", q_block, k_block) * scale`, it's computationally more efficient to scale one of the input block tensors before executing the block. This converts an $O(M \times N)$ array multiplication into an $O(M \times D)$ scaling step.
**Action:** When implementing or refactoring customized Pallas kernels and dense grid operations, ensure that scalar scaling operations are pushed through matrix multiplications inside nested unrolled loops and applied to incoming input tensors beforehand.
