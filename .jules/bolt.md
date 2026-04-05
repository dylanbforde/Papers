## 2025-04-05 - Optimize JAX Tensor Padding

**Learning:** When aligning block sizes or packing mod-8 channels for XLA/TPU compilers, using `jnp.pad` on active tensors is significantly more efficient than using `jnp.stack` with explicitly allocated `jnp.zeros_like()` structures. `jnp.stack` triggers memory allocations for the zero blocks, while `jnp.pad` safely delegates this to faster slice-update ops under the hood in XLA.

**Action:** Whenever padding tensors along the channel or sequence dimension for hardware alignment constraints (like mod-8 or mod-128 blocks for Trillium), prefer `jnp.pad` over concatenation or stack operations with manual zeros.
