import jax
import jax.numpy as jnp

def safe_exp(x):
    return jnp.exp(x) # Simplified for testing logic

B, M, N = 2, 4, 4
s_block = jax.random.normal(jax.random.PRNGKey(0), (B, M, N))
u1 = jax.random.normal(jax.random.PRNGKey(1), (B, M))
v1 = jax.random.normal(jax.random.PRNGKey(2), (B, N))
u2 = jax.random.normal(jax.random.PRNGKey(3), (B, M))
v2 = jax.random.normal(jax.random.PRNGKey(4), (B, N))
v0 = jax.random.normal(jax.random.PRNGKey(5), (B, N))

# Original
P22_orig = safe_exp(s_block + u2[:, :, None] + v2[:, None, :])
P21_orig = safe_exp(s_block + u2[:, :, None] + v1[:, None, :])
P11_orig = safe_exp(s_block + u1[:, :, None] + v1[:, None, :])
P10_orig = safe_exp(s_block + u1[:, :, None] + v0[:, None, :])

# Optimized
P11_opt = safe_exp(s_block + u1[:, :, None] + v1[:, None, :])
exp_du = jnp.exp(u2 - u1)[:, :, None]
P21_opt = P11_opt * exp_du
P22_opt = P21_opt * jnp.exp(v2 - v1)[:, None, :]
P10_opt = P11_opt * jnp.exp(v0 - v1)[:, None, :]

print("P22 diff:", jnp.max(jnp.abs(P22_orig - P22_opt)))
print("P21 diff:", jnp.max(jnp.abs(P21_orig - P21_opt)))
print("P11 diff:", jnp.max(jnp.abs(P11_orig - P11_opt)))
print("P10 diff:", jnp.max(jnp.abs(P10_orig - P10_opt)))
