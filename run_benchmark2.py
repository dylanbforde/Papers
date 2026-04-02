import jax
import jax.numpy as jnp
import time
from src.ot_kernel import pallas_sinkhorn_bwd_fused

def safe_exp(x):
    return jnp.exp(x) # We'll just test the logic

# Test the P22 computation logic
B, M, N = 2, 128, 128
s_block = jax.random.normal(jax.random.PRNGKey(0), (B, M, N))
u1_block = jax.random.normal(jax.random.PRNGKey(1), (B, M))
v1_block = jax.random.normal(jax.random.PRNGKey(2), (B, N))
u2_block = jax.random.normal(jax.random.PRNGKey(3), (B, M))
v2_block = jax.random.normal(jax.random.PRNGKey(4), (B, N))
v0_block = jax.random.normal(jax.random.PRNGKey(5), (B, N))

@jax.jit
def orig():
    P22 = safe_exp(s_block + u2_block[:, :, None] + v2_block[:, None, :])
    P21 = safe_exp(s_block + u2_block[:, :, None] + v1_block[:, None, :])
    P11 = safe_exp(s_block + u1_block[:, :, None] + v1_block[:, None, :])
    P10 = safe_exp(s_block + u1_block[:, :, None] + v0_block[:, None, :])
    return P22, P21, P11, P10

@jax.jit
def opt():
    P11 = safe_exp(s_block + u1_block[:, :, None] + v1_block[:, None, :])
    exp_du = jnp.exp(u2_block - u1_block)[:, :, None]
    P21 = P11 * exp_du
    P22 = P21 * jnp.exp(v2_block - v1_block)[:, None, :]
    P10 = P11 * jnp.exp(v0_block - v1_block)[:, None, :]
    return P22, P21, P11, P10

orig()
opt()

t0 = time.time()
for _ in range(100):
    orig()
print("Orig time:", time.time() - t0)

t0 = time.time()
for _ in range(100):
    opt()
print("Opt time:", time.time() - t0)
