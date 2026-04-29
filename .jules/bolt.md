## 2024-05-24 - Faster Tensor Packing with jnp.pad
**Learning:** When packing multiple tensors together where some channels need to be padded with zeros (e.g. for hardware alignment mod-8), using `jnp.stack` with multiple `jnp.zeros_like()` calls forces JAX to allocate large explicitly zeroed arrays. Using `jnp.stack` only on active variables and then padding the last dimension with `jnp.pad(active, ((0, 0), ..., (0, N)))` bypasses these allocations and translates to much faster slice-update ops in XLA (up to ~1.8x faster on CPU during JIT).
**Action:** When creating mod-N padded tensor structures for kernels, always prefer `jnp.pad` over explicit `jnp.zeros_like` stacking.
## 2026-04-29 - String Translation and Vectorized Masking
**Learning:** Using `str.translate()` with a pre-computed translation table is ~75x faster than generator expressions for string filtering. Vectorizing matrix construction with boolean masks is ~2x faster than a Python loop.
**Action:** Always prefer `str.translate()` for bulk character removal and numpy vectorization for matrix construction.
