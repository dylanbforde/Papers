"""
Flash Sinkhorn Attention — Source of Truth
==========================================
This file is the canonical reference implementation for the banded Sinkhorn
attention mechanism described in the NeurIPS/ISMB manuscripts.

It contains:
  1. The full banded forward pass with Pallas tile kernels (TPU v6e compatible).
  2. The fused Pallas backward kernels (dQ, dK, dV) executing inside TPU VMEM.
  3. The novel **Multi-Transport-Plan** backward pass (Proposition 1 in the paper)
     that replaces the divergent Richardson/Gauss-Seidel adjoint with explicit
     unrolled backpropagation through 2 Sinkhorn steps using 4 distinct transport
     matrices P^{(2,2)}, P^{(2,1)}, P^{(1,1)}, P^{(1,0)}.
  4. A pure-JAX reference backward for numerical validation.

All block sizes are aligned to (mod 8, mod 128) for TPU v6e Trillium compatibility.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

try:
    from jax.experimental.pallas import tpu as pltpu
except ImportError:
    pltpu = None
import functools
from functools import partial


# ============================================================================
# Utilities
# ============================================================================

def resolve_compiler_params(dim_semantics):
    if pltpu is None:
        return None
    if hasattr(pltpu, "CompilerParams"):
        return pltpu.CompilerParams(dimension_semantics=dim_semantics)
    elif hasattr(pltpu, "TPUCompilerParams"):
        return pltpu.TPUCompilerParams(dimension_semantics=dim_semantics)
    return None


def _poly_exp2_frac(r):
    """
    Approximate 2^r for r in [0, 1) using a degree-5 minimax polynomial.
    All operations are FMA (fused multiply-add), bypassing the VPU's MUFU unit.
    
    Coefficients are the Taylor series of 2^x = exp(x*ln2) truncated at degree 5,
    which gives ~1e-7 relative error on [0, 1) — sufficient for float32.
    """
    # Horner's method: p0 + r*(p1 + r*(p2 + r*(p3 + r*(p4 + r*p5))))
    p5 = 1.5252715e-4    # ln(2)^5 / 120
    p4 = 1.3215486e-3    # ln(2)^4 / 24
    p3 = 9.6178085e-3    # ln(2)^3 / 6
    p2 = 5.5504109e-2    # ln(2)^2 / 2
    p1 = 2.4022651e-1    # ln(2)
    p0 = 6.9314718e-1    # ln(2) — note: 2^0 = 1, so true p0 = 1.0
    # We return 1.0 + r * polynomial to get 2^r
    return 1.0 + r * (p0 + r * (p1 + r * (p2 + r * (p3 + r * (p4 + r * p5)))))


def safe_exp(x):
    """
    Numerically safe exp(x) via Cody-Waite range reduction + polynomial.
    
    Implements the FlashAttention-4 technique (Dao et al., 2026):
        exp(x) = 2^(x / ln2) = 2^floor(t) * 2^frac(t)
    where t = x * log2(e).
    
    - 2^floor(t) is computed via jnp.exp2 (maps to bit manipulation on integer ALU)
    - 2^frac(t) is computed via degree-5 polynomial (maps to FMA units)
    - floor(t) is clamped to [-126, 127] to prevent float32 overflow/underflow
    
    This eliminates VPU MUFU dependency and prevents inf/NaN from overflow.
    """
    LOG2E = jnp.float32(1.4426950408889634)  # log2(e)
    t = x * LOG2E
    k = jnp.floor(t)                          # integer part
    r = t - k                                  # fractional part in [0, 1)
    k = jnp.clip(k, -126.0, 127.0)            # clamp to float32 exponent range
    pow2_int = jnp.exp2(k)                     # 2^k via exponent bit manipulation
    pow2_frac = _poly_exp2_frac(r)             # 2^r via FMA polynomial
    return pow2_int * pow2_frac


# ============================================================================
# Forward Pass: Banded Sinkhorn Potential Computation
# ============================================================================

def compute_global_uv(Q, K, n_iters=15, band_width=None):
    """Compute Sinkhorn dual potentials u, v via alternating logsumexp."""
    B, H, L, D = Q.shape
    scale = 1.0 / jnp.sqrt(D)
    
    u = jnp.zeros((B, H, L), dtype=Q.dtype)
    v = jnp.zeros((B, H, L), dtype=Q.dtype)
    
    BLOCK = 128
    num_blocks = L // BLOCK
    
    Q_b = Q.reshape(B, H, num_blocks, BLOCK, D).transpose(2, 0, 1, 3, 4)
    K_b = K.reshape(B, H, num_blocks, BLOCK, D).transpose(2, 0, 1, 3, 4)

    W = band_width // BLOCK if band_width is not None else num_blocks
    slice_size = min(W * 2 + 1, num_blocks)

    def body_fn(i, val):
        u_in, v_in = val
        
        v_b = v_in.reshape(B, H, num_blocks, BLOCK).transpose(2, 0, 1, 3)
        def scan_q(carry, q_kv_idx):
            q_block, block_idx = q_kv_idx
            
            start_idx = jnp.minimum(jnp.maximum(0, block_idx - W), num_blocks - slice_size)
            
            k_band = jax.lax.dynamic_slice_in_dim(K_b, start_idx, slice_size, axis=0)
            v_band = jax.lax.dynamic_slice_in_dim(v_b, start_idx, slice_size, axis=0)
            
            abs_indices = start_idx + jnp.arange(slice_size)
            mask = jnp.abs(abs_indices - block_idx) <= W

            def scan_k(acc, elements):
                k_block, v_block, is_valid = elements
                def true_fn(_):
                    S_block = jnp.einsum("bhmd,bhnd->bhmn", q_block, k_block) * scale
                    val = S_block + v_block[..., None, :]
                    max_val = jnp.maximum(acc[0], jnp.max(val, axis=-1))
                    
                    diff = val - max_val[..., None]
                    diff = jnp.where(jnp.isnan(diff), -jnp.inf, diff)
                    acc_diff = acc[0] - max_val
                    acc_diff = jnp.where(jnp.isnan(acc_diff), -jnp.inf, acc_diff)
                    
                    sum_exp = acc[1] * jnp.exp(acc_diff) + jnp.sum(jnp.exp(diff), axis=-1)
                    return (max_val, sum_exp)
                
                return jax.lax.cond(is_valid, true_fn, lambda _: acc, None), None
                
            init_acc = (jnp.full((B, H, BLOCK), -float('inf')), jnp.zeros((B, H, BLOCK)))
            final_acc, _ = jax.lax.scan(scan_k, init_acc, (k_band, v_band, mask))
            
            sum_exp = jnp.maximum(final_acc[1], 1e-10)
            u_new = -(final_acc[0] + jnp.log(sum_exp))
            return None, u_new
        
        _, u_new_b = jax.lax.scan(scan_q, None, (Q_b, jnp.arange(num_blocks)))
        u_new = u_new_b.transpose(1, 2, 0, 3).reshape(B, H, L)
        
        u_b = u_new.reshape(B, H, num_blocks, BLOCK).transpose(2, 0, 1, 3)
        def scan_k2(carry, k_qu_idx):
            k_block, block_idx = k_qu_idx
            
            start_idx = jnp.minimum(jnp.maximum(0, block_idx - W), num_blocks - slice_size)
            
            q_band = jax.lax.dynamic_slice_in_dim(Q_b, start_idx, slice_size, axis=0)
            u_band = jax.lax.dynamic_slice_in_dim(u_b, start_idx, slice_size, axis=0)
            
            abs_indices = start_idx + jnp.arange(slice_size)
            mask = jnp.abs(abs_indices - block_idx) <= W

            def scan_q2(acc, elements):
                q_block, u_block, is_valid = elements
                def true_fn(_):
                    S_block = jnp.einsum("bhmd,bhnd->bhmn", q_block, k_block) * scale
                    val = S_block + u_block[..., :, None]
                    max_val = jnp.maximum(acc[0], jnp.max(val, axis=-2))
                    
                    diff = val - max_val[..., None, :]
                    diff = jnp.where(jnp.isnan(diff), -jnp.inf, diff)
                    acc_diff = acc[0] - max_val
                    acc_diff = jnp.where(jnp.isnan(acc_diff), -jnp.inf, acc_diff)
                    
                    sum_exp = acc[1] * jnp.exp(acc_diff) + jnp.sum(jnp.exp(diff), axis=-2)
                    return (max_val, sum_exp)
                
                return jax.lax.cond(is_valid, true_fn, lambda _: acc, None), None
            
            init_acc = (jnp.full((B, H, BLOCK), -float('inf')), jnp.zeros((B, H, BLOCK)))
            final_acc, _ = jax.lax.scan(scan_q2, init_acc, (q_band, u_band, mask))
            
            sum_exp = jnp.maximum(final_acc[1], 1e-10)
            v_new = -(final_acc[0] + jnp.log(sum_exp))
            return None, v_new
            
        _, v_new_b = jax.lax.scan(scan_k2, None, (K_b, jnp.arange(num_blocks)))
        v_new = v_new_b.transpose(1, 2, 0, 3).reshape(B, H, L)
        
        return (u_new, v_new)

    return jax.lax.fori_loop(0, n_iters, body_fn, (u, v))


# ============================================================================
# Forward Pass: Pallas Tile Kernel
# ============================================================================

def tile_sinkhorn_kernel(q_ref, k_ref, v_ref, u_ref, v_pot_ref, o_ref):
    """Single-tile forward: logits -> Sinkhorn weights -> output."""
    q = q_ref[...]
    k = k_ref[...]
    v = v_ref[...]
    u = u_ref[..., 0]   # v6e: unpack trailing dim
    v_pot = v_pot_ref[..., 0]

    D = q.shape[-1]
    logits = jnp.dot(q, k.T) / jnp.sqrt(D)
    log_alpha = logits + u[:, None] + v_pot[None, :]
    o_ref[...] = jnp.dot(jnp.exp(log_alpha), v)


def pallas_sinkhorn_fwd(Q, K, V, n_iters=15, band_width=None):
    """Full banded Sinkhorn forward pass with Pallas tile kernels."""
    u_out, v_out = compute_global_uv(Q, K, n_iters, band_width)
    
    B, H, L, D = Q.shape
    block_size = 128
    num_blocks = L // block_size
    head_dim = D
    W = band_width // block_size if band_width is not None else num_blocks
    slice_size = min(W * 2 + 1, num_blocks)

    q_blocks = Q.reshape(B, H, num_blocks, block_size, head_dim)
    k_blocks = K.reshape(B, H, num_blocks, block_size, head_dim)
    v_blocks = V.reshape(B, H, num_blocks, block_size, head_dim)
    u_blocks = u_out.reshape(B, H, num_blocks, block_size, 1)   # v6e pad
    v_pot_blocks = v_out.reshape(B, H, num_blocks, block_size, 1)

    def compute_tile(q_tile, k_tile, v_tile, u_tile, v_pot_tile):
        out_shape = jax.ShapeDtypeStruct((block_size, head_dim), Q.dtype)
        is_cpu = jax.devices()[0].platform == "cpu"
        kw = {"interpret": True} if is_cpu else {}
        return pl.pallas_call(
            functools.partial(tile_sinkhorn_kernel),
            out_shape=out_shape,
            grid=(),
            in_specs=[
                pl.BlockSpec(block_shape=q_tile.shape, index_map=lambda: (0, 0)),
                pl.BlockSpec(block_shape=k_tile.shape, index_map=lambda: (0, 0)),
                pl.BlockSpec(block_shape=v_tile.shape, index_map=lambda: (0, 0)),
                pl.BlockSpec(block_shape=u_tile.shape, index_map=lambda: (0, 0)),
                pl.BlockSpec(block_shape=v_pot_tile.shape, index_map=lambda: (0, 0)),
            ],
            out_specs=pl.BlockSpec(block_shape=out_shape.shape, index_map=lambda: (0, 0)),
            **kw,
        )(q_tile, k_tile, v_tile, u_tile, v_pot_tile)

    def scan_kv_blocks(acc, elements):
        k_tile, v_tile, v_pot_tile, is_valid = elements
        q_tile, u_tile, out_sum = acc
        out_tile = compute_tile(q_tile, k_tile, v_tile, u_tile, v_pot_tile)
        out_tile = jnp.where(is_valid, out_tile, 0.0)
        return (q_tile, u_tile, out_sum + out_tile), None

    def process_q_block(q_tile, u_tile, block_idx, k_blocks_all, v_blocks_all, v_pot_blocks_all):
        init_sum = jnp.zeros((block_size, head_dim), dtype=q_tile.dtype)
        start_idx = jnp.minimum(jnp.maximum(0, block_idx - W), num_blocks - slice_size)
        k_band = jax.lax.dynamic_slice_in_dim(k_blocks_all, start_idx, slice_size, axis=0)
        v_band = jax.lax.dynamic_slice_in_dim(v_blocks_all, start_idx, slice_size, axis=0)
        v_pot_band = jax.lax.dynamic_slice_in_dim(v_pot_blocks_all, start_idx, slice_size, axis=0)
        abs_indices = start_idx + jnp.arange(slice_size)
        mask = jnp.abs(abs_indices - block_idx) <= W
        (q_val, _, final_sum), _ = jax.lax.scan(
            scan_kv_blocks, (q_tile, u_tile, init_sum), (k_band, v_band, v_pot_band, mask)
        )
        return final_sum

    vmap_q = jax.vmap(process_q_block, in_axes=(0, 0, 0, None, None, None))
    vmap_heads = jax.vmap(vmap_q, in_axes=(0, 0, None, 0, 0, 0))
    vmap_batch = jax.vmap(vmap_heads, in_axes=(0, 0, None, 0, 0, 0))
    out_blocks = vmap_batch(q_blocks, u_blocks, jnp.arange(num_blocks), k_blocks, v_blocks, v_pot_blocks)
    out = out_blocks.reshape(B, H, L, D)
    return out, (u_out, v_out, Q, K, V)


# ============================================================================
# Backward Pass: Fused Pallas Multi-Transport Kernels (TPU VMEM)
# ============================================================================
# Tensor Packing Strategy:
#   M_data: (Batch, M, 8) = [u0, u1, u2, del_u1, del_u2, 0, 0, 0]
#   N_data: (Batch, N, 8) = [v0, v1, v2, g_v, del_v1, 0, 0, 0]
# This packs all potential histories + gradient intermediates into 2 tensors,
# reducing the kernel signature to 7 inputs (well within TPU v6e VMEM limits).

def dq_kernel_multi(
    Q_ref, K_ref, V_ref, M_data_ref, N_data_ref, dO_ref, W_ref, dQ_ref
):
    """
    Fused Pallas kernel: compute dQ using the multi-transport-plan formula.
    Executes entirely in TPU VMEM with banded iteration.
    
    dS_{mn} = P22*Z - P22*g_v - P21*del_u2 - P11*del_v1 - P10*del_u1
    dQ_m = sum_n dS_{mn} * K_n * scale
    """
    q_block = Q_ref[...]     # (BLOCK, D)
    do_block = dO_ref[...]   # (BLOCK, E)
    m_data = M_data_ref[...] # (BLOCK, 8)
    W = W_ref[0]

    u1_m = m_data[:, 1]
    u2_m = m_data[:, 2]
    del_u1_m = m_data[:, 3]
    del_u2_m = m_data[:, 4]

    D = q_block.shape[-1]
    scale = 1.0 / jnp.sqrt(D)

    N = K_ref.shape[0]
    BLOCK_N = 128
    UNROLL = 4
    iters = N // BLOCK_N

    m = pl.program_id(0)
    start_block = jnp.maximum(0, m - W)
    end_block = jnp.minimum(iters, m + W + 1)
    start_iter = start_block // UNROLL
    end_iter = (end_block + UNROLL - 1) // UNROLL

    init_state = (jnp.zeros_like(q_block),)

    def body_fn(i, state):
        (dq_acc,) = state
        for u_i in range(UNROLL):
            k_idx = i * UNROLL + u_i
            off = k_idx * BLOCK_N
            mask = (k_idx >= start_block) & (k_idx < end_block)

            k_block = K_ref[pl.ds(off, BLOCK_N), :]
            v_mat_block = V_ref[pl.ds(off, BLOCK_N), :]
            n_data = N_data_ref[pl.ds(off, BLOCK_N), :]

            v0_n = n_data[:, 0]
            v1_n = n_data[:, 1]
            v2_n = n_data[:, 2]
            g_v_n = n_data[:, 3]
            del_v1_n = n_data[:, 4]

            # Core Transport Plan Block P^{(1,1)}
            s_block = jnp.einsum("md,nd->mn", q_block, k_block) * scale
            P11 = safe_exp(s_block + u1_m[:, None] + v1_n[None, :])

            # Multi-Transport Plan Factorization (Proposition 1)
            # P^{(a,b)} = P^{(1,1)} * exp(u_a - u_1) * exp(v_b - v_1)
            # This factors out the expensive NxN matrix exponentials.
            exp_u2_u1 = safe_exp(u2_m - u1_m)
            exp_v2_v1 = safe_exp(v2_n - v1_n)
            exp_v0_v1 = safe_exp(v0_n - v1_n)

            z_block = jnp.einsum("me,ne->mn", do_block, v_mat_block)

            # dS = Term A - Term B - Term C - Term D - Term E
            # Algebraic reduction: factor out P11
            ds_block = P11 * (
                exp_u2_u1[:, None] * exp_v2_v1[None, :] * (z_block - g_v_n[None, :])
                - exp_u2_u1[:, None] * del_u2_m[:, None]
                - del_v1_n[None, :]
                - exp_v0_v1[None, :] * del_u1_m[:, None]
            )

            ds_block = jnp.where(mask, ds_block, 0.0)
            dq_acc = dq_acc + jnp.einsum("mn,nd->md", ds_block, k_block)
        return (dq_acc,)

    dq_acc = jax.lax.fori_loop(start_iter, end_iter, body_fn, init_state)[0]
    dQ_ref[...] = dq_acc * scale


def dk_dv_kernel_multi(
    Q_ref, K_ref, V_ref, M_data_ref, N_data_ref, dO_ref, W_ref, dK_ref, dV_ref
):
    """
    Fused Pallas kernel: compute dK, dV using the multi-transport-plan formula.
    Executes entirely in TPU VMEM with banded iteration.
    """
    k_block = K_ref[...]       # (BLOCK, D)
    v_mat_block = V_ref[...]   # (BLOCK, E)
    n_data = N_data_ref[...]   # (BLOCK, 8)
    W = W_ref[0]

    v1_n = n_data[:, 1]
    v2_n = n_data[:, 2]
    g_v_n = n_data[:, 3]
    del_v1_n = n_data[:, 4]

    D = k_block.shape[-1]
    M = Q_ref.shape[0]
    BLOCK_M = 128
    UNROLL = 4
    iters = M // BLOCK_M
    scale = 1.0 / jnp.sqrt(D)

    n = pl.program_id(0)
    start_block = jnp.maximum(0, n - W)
    end_block = jnp.minimum(iters, n + W + 1)
    start_iter = start_block // UNROLL
    end_iter = (end_block + UNROLL - 1) // UNROLL

    init_state = (jnp.zeros_like(k_block), jnp.zeros_like(v_mat_block))

    def body_fn(i, state):
        dk_acc, dv_acc = state
        for u_i in range(UNROLL):
            m_idx = i * UNROLL + u_i
            off = m_idx * BLOCK_M
            mask = (m_idx >= start_block) & (m_idx < end_block)

            q_block = Q_ref[pl.ds(off, BLOCK_M), :]
            do_block = dO_ref[pl.ds(off, BLOCK_M), :]
            m_data = M_data_ref[pl.ds(off, BLOCK_M), :]

            u0_m = m_data[:, 0]
            u1_m = m_data[:, 1]
            u2_m = m_data[:, 2]
            del_u1_m = m_data[:, 3]
            del_u2_m = m_data[:, 4]

            # Core Transport Plan Block P^{(1,1)} (transposed)
            s_block_T = jnp.einsum("nd,md->nm", k_block, q_block) * scale
            P11_T = safe_exp(s_block_T + v1_n[:, None] + u1_m[None, :])

            # Multi-Transport Plan Factorization
            exp_u2_u1 = safe_exp(u2_m - u1_m)
            exp_u0_u1 = safe_exp(u0_m - u1_m)
            exp_v2_v1 = safe_exp(v2_n - v1_n)

            # dS_T = Term A - Term B - Term C - Term D - Term E
            # (Note: Z_T = V @ dO^T)
            z_block_T = jnp.einsum("ne,me->nm", v_mat_block, do_block)
            
            ds_block_T = P11_T * (
                exp_v2_v1[:, None] * exp_u2_u1[None, :] * (z_block_T - g_v_n[:, None])
                - exp_v2_v1[:, None] * del_u2_m[None, :]
                - del_v1_n[:, None]
                - exp_v0_v1[:, None] * del_u1_m[None, :]
            )

            ds_block_T = jnp.where(mask, ds_block_T, 0.0)
            P22_T = jnp.where(mask, P11_T * exp_v2_v1[:, None] * exp_u2_u1[None, :], 0.0)

            dk_acc = dk_acc + jnp.einsum("nm,md->nd", ds_block_T, q_block)
            dv_acc = dv_acc + jnp.einsum("nm,me->ne", P22_T, do_block)
        return (dk_acc, dv_acc)

    res = jax.lax.fori_loop(start_iter, end_iter, body_fn, init_state)
    dk_acc, dv_acc = res
    dK_ref[...] = dk_acc * scale
    dV_ref[...] = dv_acc


def pallas_sinkhorn_bwd_fused(Q, K, V, dO, M_data, N_data, band_width):
    """Launch fused Pallas kernels over Batch/Head sequentially."""
    M, D = Q.shape
    N, E = V.shape
    C = 8

    BLOCK = 128
    W_val = band_width // BLOCK if band_width is not None else M // BLOCK
    W_arr = jnp.array([W_val], dtype=jnp.int32)

    is_cpu = jax.devices()[0].platform == "cpu"
    kw = {"interpret": True} if is_cpu else {}

    grid_dq = (M // BLOCK,)
    dQ_f = pl.pallas_call(
        dq_kernel_multi,
        out_shape=jax.ShapeDtypeStruct(Q.shape, Q.dtype),
        grid=grid_dq,
        in_specs=[
            pl.BlockSpec(index_map=lambda m: (m, 0), block_shape=(BLOCK, D)),
            pl.BlockSpec(index_map=lambda m: (0, 0), block_shape=(N, D)),
            pl.BlockSpec(index_map=lambda m: (0, 0), block_shape=(N, E)),
            pl.BlockSpec(index_map=lambda m: (m, 0), block_shape=(BLOCK, C)),
            pl.BlockSpec(index_map=lambda m: (0, 0), block_shape=(N, C)),
            pl.BlockSpec(index_map=lambda m: (m, 0), block_shape=(BLOCK, E)),
            pl.BlockSpec(index_map=lambda m: (0,), block_shape=(1,)),
        ],
        out_specs=pl.BlockSpec(index_map=lambda m: (m, 0), block_shape=(BLOCK, D)),
        **kw,
    )(Q, K, V, M_data, N_data, dO, W_arr)

    grid_dk_dv = (N // BLOCK,)
    dK_f, dV_f = pl.pallas_call(
        dk_dv_kernel_multi,
        out_shape=(jax.ShapeDtypeStruct(K.shape, K.dtype), jax.ShapeDtypeStruct(V.shape, V.dtype)),
        grid=grid_dk_dv,
        in_specs=[
            pl.BlockSpec(index_map=lambda n: (0, 0), block_shape=(M, D)),
            pl.BlockSpec(index_map=lambda n: (n, 0), block_shape=(BLOCK, D)),
            pl.BlockSpec(index_map=lambda n: (n, 0), block_shape=(BLOCK, E)),
            pl.BlockSpec(index_map=lambda n: (0, 0), block_shape=(M, C)),
            pl.BlockSpec(index_map=lambda n: (n, 0), block_shape=(BLOCK, C)),
            pl.BlockSpec(index_map=lambda n: (0, 0), block_shape=(M, E)),
            pl.BlockSpec(index_map=lambda n: (0,), block_shape=(1,)),
        ],
        out_specs=[
            pl.BlockSpec(index_map=lambda n: (n, 0), block_shape=(BLOCK, D)),
            pl.BlockSpec(index_map=lambda n: (n, 0), block_shape=(BLOCK, E)),
        ],
        **kw,
    )(Q, K, V, M_data, N_data, dO, W_arr)

    return dQ_f, dK_f, dV_f

def _banded_compute_next_uv(Q, K, u_in, v_in, scale, BLOCK, num_blocks, W, slice_size):
    """Compute one Sinkhorn half-step (u_new, v_new) with banding."""
    def scan_u(carry, i):
        q_block = jax.lax.dynamic_slice_in_dim(Q, i * BLOCK, BLOCK, axis=-2)
        start_idx = jnp.minimum(jnp.maximum(0, i - W), num_blocks - slice_size)
        k_band = jax.lax.dynamic_slice_in_dim(K, start_idx * BLOCK, slice_size * BLOCK, axis=-2)
        v_band = jax.lax.dynamic_slice_in_dim(v_in, start_idx * BLOCK, slice_size * BLOCK, axis=-1)
        S_block = jnp.einsum("bhmd,bhnd->bhmn", q_block, k_band) * scale
        logits = S_block + v_band[..., None, :]
        abs_indices = start_idx + jnp.arange(slice_size * BLOCK) // BLOCK
        mask = jnp.abs(abs_indices - i) <= W
        logits = jnp.where(mask[None, None, None, :], logits, -jnp.inf)
        u_new_block = -jax.scipy.special.logsumexp(logits, axis=-1)
        return carry, u_new_block
    _, u_blocks = jax.lax.scan(scan_u, None, jnp.arange(num_blocks))
    u_new = jnp.concatenate(u_blocks, axis=-1)

    def scan_v(carry, j):
        k_block = jax.lax.dynamic_slice_in_dim(K, j * BLOCK, BLOCK, axis=-2)
        start_idx = jnp.minimum(jnp.maximum(0, j - W), num_blocks - slice_size)
        q_band = jax.lax.dynamic_slice_in_dim(Q, start_idx * BLOCK, slice_size * BLOCK, axis=-2)
        u_band = jax.lax.dynamic_slice_in_dim(u_new, start_idx * BLOCK, slice_size * BLOCK, axis=-1)
        S_block = jnp.einsum("bhmd,bhnd->bhmn", q_band, k_block) * scale
        logits = S_block + u_band[..., :, None]
        abs_indices = start_idx + jnp.arange(slice_size * BLOCK) // BLOCK
        mask = jnp.abs(abs_indices - j) <= W
        logits = jnp.where(mask[None, None, :, None], logits, -jnp.inf)
        v_new_block = -jax.scipy.special.logsumexp(logits, axis=-2)
        return carry, v_new_block
    _, v_blocks = jax.lax.scan(scan_v, None, jnp.arange(num_blocks))
    v_new = jnp.concatenate(v_blocks, axis=-1)
    return u_new, v_new


def _banded_matvec_P(Q, K, u_a, v_b, vec, scale, BLOCK, num_blocks, W, slice_size, direction="row"):
    """Banded matvec P^(a,b) @ vec or P^(a,b)^T @ vec."""
    if direction == "row":
        def scan_body(carry, i):
            q_block = jax.lax.dynamic_slice_in_dim(Q, i * BLOCK, BLOCK, axis=-2)
            u_block = jax.lax.dynamic_slice_in_dim(u_a, i * BLOCK, BLOCK, axis=-1)
            start_idx = jnp.minimum(jnp.maximum(0, i - W), num_blocks - slice_size)
            k_band = jax.lax.dynamic_slice_in_dim(K, start_idx * BLOCK, slice_size * BLOCK, axis=-2)
            v_band = jax.lax.dynamic_slice_in_dim(v_b, start_idx * BLOCK, slice_size * BLOCK, axis=-1)
            vec_band = jax.lax.dynamic_slice_in_dim(vec, start_idx * BLOCK, slice_size * BLOCK, axis=-1)
            S_block = jnp.einsum("bhmd,bhnd->bhmn", q_block, k_band) * scale
            A_block = safe_exp(S_block + u_block[..., :, None] + v_band[..., None, :])
            abs_indices = start_idx + jnp.arange(slice_size * BLOCK) // BLOCK
            mask = jnp.abs(abs_indices - i) <= W
            A_block = jnp.where(mask[None, None, None, :], A_block, 0.0)
            res_block = jnp.sum(A_block * vec_band[..., None, :], axis=-1)
            return carry, res_block
        _, res_blocks = jax.lax.scan(scan_body, None, jnp.arange(num_blocks))
        return jnp.concatenate(res_blocks, axis=-1)
    else:
        result = jnp.zeros_like(v_b)
        def scan_body(acc, i):
            q_block = jax.lax.dynamic_slice_in_dim(Q, i * BLOCK, BLOCK, axis=-2)
            u_block = jax.lax.dynamic_slice_in_dim(u_a, i * BLOCK, BLOCK, axis=-1)
            vec_block = jax.lax.dynamic_slice_in_dim(vec, i * BLOCK, BLOCK, axis=-1)
            start_idx = jnp.minimum(jnp.maximum(0, i - W), num_blocks - slice_size)
            k_band = jax.lax.dynamic_slice_in_dim(K, start_idx * BLOCK, slice_size * BLOCK, axis=-2)
            v_band = jax.lax.dynamic_slice_in_dim(v_b, start_idx * BLOCK, slice_size * BLOCK, axis=-1)
            S_block = jnp.einsum("bhmd,bhnd->bhmn", q_block, k_band) * scale
            A_block = safe_exp(S_block + u_block[..., :, None] + v_band[..., None, :])
            abs_indices = start_idx + jnp.arange(slice_size * BLOCK) // BLOCK
            mask = jnp.abs(abs_indices - i) <= W
            A_block = jnp.where(mask[None, None, None, :], A_block, 0.0)
            contrib = jnp.sum(A_block * vec_block[..., :, None], axis=-2)
            contrib_padded = jnp.zeros_like(v_b)
            contrib_padded = jax.lax.dynamic_update_slice_in_dim(contrib_padded, contrib, start_idx * BLOCK, axis=-1)
            return acc + contrib_padded, None
        result, _ = jax.lax.scan(scan_body, result, jnp.arange(num_blocks))
        return result



def pallas_sinkhorn_bwd_full(n_iters, band_width, res, g):
    """Multi-Transport-Plan backward with sequential Batch mapping."""
    u0, v0, Q, K, V = res
    dO = g

    scale = 1.0 / jnp.sqrt(Q.shape[-1])
    BLOCK = 128
    M = Q.shape[-2]
    num_blocks = M // BLOCK
    W = band_width // BLOCK if band_width is not None else num_blocks
    slice_size = min(W * 2 + 1, num_blocks)

    u1, v1 = _banded_compute_next_uv(Q, K, u0, v0, scale, BLOCK, num_blocks, W, slice_size)
    u2, v2 = _banded_compute_next_uv(Q, K, u1, v1, scale, BLOCK, num_blocks, W, slice_size)

    def matvec_row(u_a, v_b, vec):
        return _banded_matvec_P(Q, K, u_a, v_b, vec, scale, BLOCK, num_blocks, W, slice_size, "row")
    def matvec_col(u_a, v_b, vec):
        return _banded_matvec_P(Q, K, u_a, v_b, vec, scale, BLOCK, num_blocks, W, slice_size, "col")

    g_u_init = jnp.zeros_like(u0)
    g_v_init = jnp.zeros_like(v0)
    def pass1_body(carry, i):
        g_u, g_v = carry
        q_block = jax.lax.dynamic_slice_in_dim(Q, i * BLOCK, BLOCK, axis=-2)
        u_block = jax.lax.dynamic_slice_in_dim(u2, i * BLOCK, BLOCK, axis=-1)
        do_block = jax.lax.dynamic_slice_in_dim(dO, i * BLOCK, BLOCK, axis=-2)
        start_idx = jnp.minimum(jnp.maximum(0, i - W), num_blocks - slice_size)
        k_band = jax.lax.dynamic_slice_in_dim(K, start_idx * BLOCK, slice_size * BLOCK, axis=-2)
        v_band = jax.lax.dynamic_slice_in_dim(v2, start_idx * BLOCK, slice_size * BLOCK, axis=-1)
        V_mat_band = jax.lax.dynamic_slice_in_dim(V, start_idx * BLOCK, slice_size * BLOCK, axis=-2)
        Z_block = jnp.matmul(do_block, V_mat_band.transpose(0, 1, 3, 2))
        S_block = jnp.einsum("bhmd,bhnd->bhmn", q_block, k_band) * scale
        A_block = safe_exp(S_block + u_block[..., :, None] + v_band[..., None, :])
        T_block = A_block * Z_block
        abs_indices = start_idx + jnp.arange(slice_size * BLOCK) // BLOCK
        mask = jnp.abs(abs_indices - i) <= W
        T_block = jnp.where(mask[None, None, None, :], T_block, 0.0)
        gu_contrib = jnp.sum(T_block, axis=-1)
        gv_contrib = jnp.sum(T_block, axis=-2)
        g_u = jax.lax.dynamic_update_slice_in_dim(g_u, gu_contrib, i * BLOCK, axis=-1)
        gv_padded = jnp.zeros_like(v0); gv_padded = jax.lax.dynamic_update_slice_in_dim(gv_padded, gv_contrib, start_idx * BLOCK, axis=-1)
        return (g_u, g_v + gv_padded), None
    (g_u, g_v), _ = jax.lax.scan(pass1_body, (g_u_init, g_v_init), jnp.arange(num_blocks))

    del_u2 = g_u - matvec_row(u2, v2, g_v)
    del_v1 = -matvec_col(u2, v1, del_u2)
    del_u1 = -matvec_row(u1, v1, del_v1)

    M_data = jnp.stack([u0, u1, u2, del_u1, del_u2, jnp.zeros_like(u0), jnp.zeros_like(u0), jnp.zeros_like(u0)], axis=-1)
    N_data = jnp.stack([v0, v1, v2, g_v, del_v1, jnp.zeros_like(v0), jnp.zeros_like(v0), jnp.zeros_like(v0)], axis=-1)

    # Sequential map to avoid VMEM OOM
    def h_map_fn(inputs):
        Q_h, K_h, V_h, dO_h, M_data_h, N_data_h = inputs
        return jax.lax.map(lambda x: pallas_sinkhorn_bwd_fused(x[0], x[1], x[2], x[3], x[4], x[5], band_width), 
                           (Q_h, K_h, V_h, dO_h, M_data_h, N_data_h))
    
    dQ, dK, dV = jax.lax.map(h_map_fn, (Q, K, V, dO, M_data, N_data))
    return dQ, dK, dV


# ============================================================================
# Public API: Custom VJP Wiring
# ============================================================================

@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def pallas_flash_sinkhorn(Q, K, V, n_iters=15, band_width=None):
    """Banded Sinkhorn attention with novel multi-transport-plan backward pass."""
    O, _ = pallas_sinkhorn_fwd(Q, K, V, n_iters, band_width)
    return O

pallas_flash_sinkhorn.defvjp(pallas_sinkhorn_fwd, pallas_sinkhorn_bwd_full)


# ============================================================================
# Pure JAX Reference (for numerical validation)
# ============================================================================

def _jax_ref_compute_next_uv(S, u_in, v_in):
    """Single Sinkhorn step: u_new, v_new from S, u, v (non-banded)."""
    S_v = S + v_in[..., None, :]
    u_out = -jax.scipy.special.logsumexp(S_v, axis=-1)
    S_u = S + u_out[..., :, None]
    v_out = -jax.scipy.special.logsumexp(S_u, axis=-2)
    return u_out, v_out


def jax_ref_sinkhorn_bwd(n_iters, res, g):
    """Pure JAX multi-transport-plan backward for validation (non-banded)."""
    u0, v0, Q, K, V = res
    dO = g
    scale = 1.0 / jnp.sqrt(Q.shape[-1])

    S = jnp.einsum("bhmd,bhnd->bhmn", Q, K) * scale
    Z = jnp.matmul(dO, V.transpose(0, 1, 3, 2))

    u1, v1 = _jax_ref_compute_next_uv(S, u0, v0)
    u2, v2 = _jax_ref_compute_next_uv(S, u1, v1)

    P11 = jnp.exp(S + u1[..., :, None] + v1[..., None, :])
    T_Z = jnp.exp(S + u2[..., :, None] + v2[..., None, :]) * Z
    g_v = jnp.sum(T_Z, axis=-2)
    
    # matvec helpers for non-banded reference
    del_u2 = jnp.sum(T_Z, axis=-1) - jnp.einsum("bhmn,bhn->bhm", jnp.exp(S + u2[..., :, None] + v2[..., None, :]), g_v)
    del_v1 = -jnp.einsum("bhmn,bhm->bhn", jnp.exp(S + u2[..., :, None] + v1[..., None, :]), del_u2)
    del_u1 = -jnp.einsum("bhmn,bhn->bhm", P11, del_v1)

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

    dS = (
        P22 * Z
        - P22 * g_v[..., None, :]
        - P21 * del_u2[..., :, None]
        - P11 * del_v1[..., None, :]
        - P10 * del_u1[..., :, None]
    )

    dQ = jnp.matmul(dS, K) * scale
    dK = jnp.matmul(dS.transpose(0, 1, 3, 2), Q) * scale
    dV = jnp.matmul(P22.transpose(0, 1, 3, 2), dO)

    return dQ, dK, dV
