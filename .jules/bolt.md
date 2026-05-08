## 2024-05-24 - Faster Tensor Packing with jnp.pad
**Learning:** When packing multiple tensors together where some channels need to be padded with zeros (e.g. for hardware alignment mod-8), using `jnp.stack` with multiple `jnp.zeros_like()` calls forces JAX to allocate large explicitly zeroed arrays. Using `jnp.stack` only on active variables and then padding the last dimension with `jnp.pad(active, ((0, 0), ..., (0, N)))` bypasses these allocations and translates to much faster slice-update ops in XLA (up to ~1.8x faster on CPU during JIT).
**Action:** When creating mod-N padded tensor structures for kernels, always prefer `jnp.pad` over explicit `jnp.zeros_like` stacking.

## 2024-05-24 - Faster Synthetic Sequence Generation via Vectorization
**Learning:** In synthetic data generation routines (e.g., benchmark scripts or data pipelines), using Python standard lib loops (`for idx in random.choice(L): seq[idx] = ...`) with string manipulations is a major bottleneck (~0.42s per 100k length sequence). Vectorizing random character replacements using NumPy array indexing (`seq_arr[indices] = np.random.choice(...)`) eliminates Python loops and list conversions, speeding up execution by ~6x (~0.07s per 100k sequence).
**Action:** Always favor vectorized NumPy array indexing over Python `for` loops for character replacement in sequence generation.
