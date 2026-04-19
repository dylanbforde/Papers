## 2024-05-24 - Faster Tensor Packing with jnp.pad
**Learning:** When packing multiple tensors together where some channels need to be padded with zeros (e.g. for hardware alignment mod-8), using `jnp.stack` with multiple `jnp.zeros_like()` calls forces JAX to allocate large explicitly zeroed arrays. Using `jnp.stack` only on active variables and then padding the last dimension with `jnp.pad(active, ((0, 0), ..., (0, N)))` bypasses these allocations and translates to much faster slice-update ops in XLA (up to ~1.8x faster on CPU during JIT).
**Action:** When creating mod-N padded tensor structures for kernels, always prefer `jnp.pad` over explicit `jnp.zeros_like` stacking.

## 2024-05-24 - Faster String Processing with str.translate
**Learning:** For character removal from strings in Python loops (e.g. stripping alignment gaps), `str.translate` with a pre-computed `str.maketrans` table is significantly faster (often >30x) than list comprehensions or generator expressions. Additionally, hoisting constant set instantiations out of inner loops avoids redundant object creation.
**Action:** When performing character replacement or removal on large datasets, always prefer `str.translate()` over character iteration.
