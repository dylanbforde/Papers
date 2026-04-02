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

import src.ot_kernel
import pallas_kernels.pallas_sinkhorn

def opt_dq_kernel_multi(Q_ref, K_ref, V_ref, M_data_ref, N_data_ref, dO_ref, W_ref, dQ_ref):
    pass # we'll replace the code via regex in python memory or just copy-paste
