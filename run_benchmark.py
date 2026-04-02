import jax
import jax.numpy as jnp
import time
from src.ot_kernel import pallas_sinkhorn_bwd_fused, dq_kernel_multi, dk_dv_kernel_multi

B, H, M, N, D, E = 2, 4, 1024, 1024, 64, 64
Q = jax.random.normal(jax.random.PRNGKey(0), (B, H, M, D))
K = jax.random.normal(jax.random.PRNGKey(1), (B, H, N, D))
V = jax.random.normal(jax.random.PRNGKey(2), (B, H, N, E))
dO = jax.random.normal(jax.random.PRNGKey(3), (B, H, M, E))
M_data = jax.random.normal(jax.random.PRNGKey(4), (B, H, M, 8))
N_data = jax.random.normal(jax.random.PRNGKey(5), (B, H, N, 8))
W = 256

print("Compiling original...")
t0 = time.time()
jax.block_until_ready(pallas_sinkhorn_bwd_fused(Q, K, V, dO, M_data, N_data, W))
print("Compilation time:", time.time() - t0)

print("Running original benchmark...")
t0 = time.time()
for _ in range(10):
    jax.block_until_ready(pallas_sinkhorn_bwd_fused(Q, K, V, dO, M_data, N_data, W))
print("Original time:", time.time() - t0)
