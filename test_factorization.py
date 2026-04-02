import jax
import jax.numpy as jnp

B, M, N = 2, 4, 4
s_block = jax.random.normal(jax.random.PRNGKey(0), (B, M, N))
u1_block = jax.random.normal(jax.random.PRNGKey(1), (B, M))
v1_block = jax.random.normal(jax.random.PRNGKey(2), (B, N))
u2_block = jax.random.normal(jax.random.PRNGKey(3), (B, M))
v2_block = jax.random.normal(jax.random.PRNGKey(4), (B, N))
v0_block = jax.random.normal(jax.random.PRNGKey(5), (B, N))
g_v_block = jax.random.normal(jax.random.PRNGKey(6), (B, N))
del_v1_block = jax.random.normal(jax.random.PRNGKey(7), (B, N))
del_u2_block = jax.random.normal(jax.random.PRNGKey(8), (B, M))
del_u1_block = jax.random.normal(jax.random.PRNGKey(9), (B, M))
z_block = jax.random.normal(jax.random.PRNGKey(10), (B, M, N))

def safe_exp(x):
    return jnp.exp(x)

P22 = safe_exp(s_block + u2_block[:, :, None] + v2_block[:, None, :])
P21 = safe_exp(s_block + u2_block[:, :, None] + v1_block[:, None, :])
P11 = safe_exp(s_block + u1_block[:, :, None] + v1_block[:, None, :])
P10 = safe_exp(s_block + u1_block[:, :, None] + v0_block[:, None, :])

ds_block_orig = (
    P22 * z_block
    - P22 * g_v_block[:, None, :]
    - P21 * del_u2_block[:, :, None]
    - P11 * del_v1_block[:, None, :]
    - P10 * del_u1_block[:, :, None]
)

P11_opt = safe_exp(s_block + u1_block[:, :, None] + v1_block[:, None, :])
exp_du21 = jnp.exp(u2_block - u1_block)
exp_dv21 = jnp.exp(v2_block - v1_block)
exp_dv01 = jnp.exp(v0_block - v1_block)

# ds_block = P11 * (exp_du21 * exp_dv21 * z_block - exp_du21 * exp_dv21 * g_v_block - exp_du21 * del_u2_block - del_v1_block - exp_dv01 * del_u1_block)
# Let's write it carefully
term_P22 = exp_du21[:, :, None] * exp_dv21[:, None, :]
term_P21 = exp_du21[:, :, None]
term_P11 = 1.0
term_P10 = exp_dv01[:, None, :]

ds_block_opt = P11_opt * (
    term_P22 * z_block
    - term_P22 * g_v_block[:, None, :]
    - term_P21 * del_u2_block[:, :, None]
    - del_v1_block[:, None, :]
    - term_P10 * del_u1_block[:, :, None]
)

print(jnp.max(jnp.abs(ds_block_orig - ds_block_opt)))
