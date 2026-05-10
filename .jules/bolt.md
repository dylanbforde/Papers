## 2024-05-24 - Faster Tensor Packing with jnp.pad
**Learning:** When packing multiple tensors together where some channels need to be padded with zeros (e.g. for hardware alignment mod-8), using `jnp.stack` with multiple `jnp.zeros_like()` calls forces JAX to allocate large explicitly zeroed arrays. Using `jnp.stack` only on active variables and then padding the last dimension with `jnp.pad(active, ((0, 0), ..., (0, N)))` bypasses these allocations and translates to much faster slice-update ops in XLA (up to ~1.8x faster on CPU during JIT).
**Action:** When creating mod-N padded tensor structures for kernels, always prefer `jnp.pad` over explicit `jnp.zeros_like` stacking.

## 2025-02-28 - Optimizing string filtering with str.translate()
**Learning:** Python generator expressions (`"".join(c for c in ali if c not in set(".-~"))`) are extremely slow for processing long strings. `str.translate()` with a pre-computed translation table (`str.maketrans("", "", ".-~")`) executes its logic in optimized C, yielding ~75x faster execution times (~0.03s vs ~2.3s per 10k ops for protein sequences).
**Action:** Always prefer `str.translate()` for removing or substituting characters in large datasets over Python-level loops or generator expressions. Also, hoist constant set/table creations outside of loop bodies.
