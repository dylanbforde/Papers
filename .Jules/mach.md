## 2024-05-20 - [Transport Matrix Re-use]
**Learning:** In Pallas backward kernels for exact Sinkhorn, calculating multiple transport matrices via independent full-block `safe_exp(S + u + v)` calls unnecessarily consumes MUFU/FMA throughput.
**Action:** Compute the base full-block transport matrix ($P^{(2,2)}$) once, then derive remaining matrices ($P^{(2,1)}, P^{(1,1)}, P^{(1,0)}$) using O(N) broadcast multiplications of `jnp.exp(potential_diff)`. This eliminates redundant full-block exponential evaluations and directly targets compute constraints.
