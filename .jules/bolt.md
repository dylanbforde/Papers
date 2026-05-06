## 2024-05-24 - Faster Tensor Packing with jnp.pad
**Learning:** When packing multiple tensors together where some channels need to be padded with zeros (e.g. for hardware alignment mod-8), using `jnp.stack` with multiple `jnp.zeros_like()` calls forces JAX to allocate large explicitly zeroed arrays. Using `jnp.stack` only on active variables and then padding the last dimension with `jnp.pad(active, ((0, 0), ..., (0, N)))` bypasses these allocations and translates to much faster slice-update ops in XLA (up to ~1.8x faster on CPU during JIT).
**Action:** When creating mod-N padded tensor structures for kernels, always prefer `jnp.pad` over explicit `jnp.zeros_like` stacking.

## 2026-05-06 - Fast Character Filtering in Python
**Learning:** Using str.translate() with a pre-computed str.maketrans() table is ~75x faster than generator expressions (~0.03s vs ~2.3s per 10k ops) for removing characters like gaps in sequences. Hoisting constant set instantiations also avoids redundant object creation in tight loops.
**Action:** Always use str.translate with a pre-computed table for removing characters in Python strings, and hoist constant sets outside of inner loops.
