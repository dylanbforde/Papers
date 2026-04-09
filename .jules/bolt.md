## 2024-05-24 - Faster Tensor Packing with jnp.pad
**Learning:** When packing multiple tensors together where some channels need to be padded with zeros (e.g. for hardware alignment mod-8), using `jnp.stack` with multiple `jnp.zeros_like()` calls forces JAX to allocate large explicitly zeroed arrays. Using `jnp.stack` only on active variables and then padding the last dimension with `jnp.pad(active, ((0, 0), ..., (0, N)))` bypasses these allocations and translates to much faster slice-update ops in XLA (up to ~1.8x faster on CPU during JIT).
**Action:** When creating mod-N padded tensor structures for kernels, always prefer `jnp.pad` over explicit `jnp.zeros_like` stacking.
## 2024-05-19 - Vectorize Data Loading Code for Performance
**Learning:** Python loops over sequence rows in data generation code (such as numpy arrays for padding logic) can be very slow, especially as sequence lengths get larger (e.g. 8192).
**Action:** Replace Python loops iterating over sequence rows with vectorized boolean indexing (e.g. `padded[np.sum(padded, axis=1) == 0] = value`).
