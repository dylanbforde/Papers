import jax
import jax.numpy as jnp
from src.ot_kernel import pallas_sinkhorn_bwd_fused, jax_ref_sinkhorn_bwd, _jax_ref_compute_next_uv

def test_math_equiv():
    B, H, M, N, D, E = 1, 1, 128, 128, 16, 16
    Q = jax.random.normal(jax.random.PRNGKey(0), (B, H, M, D))
    K = jax.random.normal(jax.random.PRNGKey(1), (B, H, N, D))
    V = jax.random.normal(jax.random.PRNGKey(2), (B, H, N, E))
    dO = jax.random.normal(jax.random.PRNGKey(3), (B, H, M, E))

    # Init potentials
    u0 = jnp.zeros((B, H, M))
    v0 = jnp.zeros((B, H, N))

    # We'll just run jax_ref_sinkhorn_bwd to get reference gradients
    res = (u0, v0, Q, K, V)
    dQ_ref, dK_ref, dV_ref = jax_ref_sinkhorn_bwd(15, res, dO)

    # Now setup data for pallas fused
    scale = 1.0 / jnp.sqrt(D)
    S = jnp.einsum("bhmd,bhnd->bhmn", Q, K) * scale
    u1, v1 = _jax_ref_compute_next_uv(S, u0, v0)
    u2, v2 = _jax_ref_compute_next_uv(S, u1, v1)

    Z = jnp.matmul(dO, V.transpose(0, 1, 3, 2))
    P22 = jnp.exp(S + u2[..., :, None] + v2[..., None, :])
    P21 = jnp.exp(S + u2[..., :, None] + v1[..., None, :])
    P11 = jnp.exp(S + u1[..., :, None] + v1[..., None, :])
    P10 = jnp.exp(S + u1[..., :, None] + v0[..., None, :])

    T_Z = P22 * Z
    g_u = jnp.sum(T_Z, axis=-1)
    g_v = jnp.sum(T_Z, axis=-2)

    del_u2 = g_u - jnp.einsum("bhmn,bhn->bhm", P22, g_v)
    del_v1 = -jnp.einsum("bhmn,bhm->bhn", P21, del_u2)
    del_u1 = -jnp.einsum("bhmn,bhn->bhm", P11, del_v1)

    M_data = jnp.stack([
        u0, u1, u2, del_u1, del_u2,
        jnp.zeros_like(u0), jnp.zeros_like(u0), jnp.zeros_like(u0)
    ], axis=-1)  # (B, H, M, 8)

    N_data = jnp.stack([
        v0, v1, v2, g_v, del_v1,
        jnp.zeros_like(v0), jnp.zeros_like(v0), jnp.zeros_like(v0)
    ], axis=-1)  # (B, H, N, 8)

    dQ_pallas, dK_pallas, dV_pallas = pallas_sinkhorn_bwd_fused(Q, K, V, dO, M_data, N_data, None)

    print("dQ max diff:", jnp.max(jnp.abs(dQ_ref - dQ_pallas)))
    print("dK max diff:", jnp.max(jnp.abs(dK_ref - dK_pallas)))
    print("dV max diff:", jnp.max(jnp.abs(dV_ref - dV_pallas)))

test_math_equiv()
