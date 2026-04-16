## 2024-05-24 - Faster Tensor Packing with jnp.pad
**Learning:** When packing multiple tensors together where some channels need to be padded with zeros (e.g. for hardware alignment mod-8), using `jnp.stack` with multiple `jnp.zeros_like()` calls forces JAX to allocate large explicitly zeroed arrays. Using `jnp.stack` only on active variables and then padding the last dimension with `jnp.pad(active, ((0, 0), ..., (0, N)))` bypasses these allocations and translates to much faster slice-update ops in XLA (up to ~1.8x faster on CPU during JIT).
**Action:** When creating mod-N padded tensor structures for kernels, always prefer `jnp.pad` over explicit `jnp.zeros_like` stacking.

## 2024-05-25 - Faster Character Removal with str.translate
**Learning:** Using a list comprehension combined with `set()` inside `"".join()` to remove specific characters from strings (e.g. `raw = "".join(c for c in ali if c not in set(".-~"))`) is computationally expensive and slow for long strings or in tight loops. Python's built-in `str.translate` using a pre-computed translation table (via `str.maketrans()`) executes this operation entirely in C and bypasses Python interpreter overhead, resulting in a >100x performance increase.
**Action:** Always use `str.translate()` with `str.maketrans()` instead of list comprehensions when filtering out specific characters from strings.
