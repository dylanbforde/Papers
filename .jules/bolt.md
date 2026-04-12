## 2024-05-24 - Faster Tensor Packing with jnp.pad
**Learning:** When packing multiple tensors together where some channels need to be padded with zeros (e.g. for hardware alignment mod-8), using `jnp.stack` with multiple `jnp.zeros_like()` calls forces JAX to allocate large explicitly zeroed arrays. Using `jnp.stack` only on active variables and then padding the last dimension with `jnp.pad(active, ((0, 0), ..., (0, N)))` bypasses these allocations and translates to much faster slice-update ops in XLA (up to ~1.8x faster on CPU during JIT).
**Action:** When creating mod-N padded tensor structures for kernels, always prefer `jnp.pad` over explicit `jnp.zeros_like` stacking.

## 2024-05-24 - Pre-scale query and key blocks
**Learning:** Multiplying the score matrix by `scale` inside inner loops (e.g., `jnp.einsum("md,nd->mn", q, k) * scale`) results in O(M x N) scalar multiplication operations.
**Action:** Pre-scale the query or key block outside of inner loops and before computing attention dot products (e.g., `q_scaled = q * scale`). This replaces O(M x N) scaling operations with O(M x D) or O(N x D), significantly saving FLOPs and compute cycles.
