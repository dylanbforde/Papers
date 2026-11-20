"""Strictly causal chunked Topology Hub Optimal Transport (THOT).

Partitions length ``L`` sequences into chunks of size ``B``:
  1. Write OT: maps each chunk's tokens to ``K`` hubs using capacity-capped
     semi-relaxed annealed Sinkhorn with column closure on ``b``.
  2. Exclusive prefix scan: aggregates chunk hub statistics strictly over
     past chunks ``c' < c`` with zero dependence on future tokens.
  3. Read OT: attends within each chunk jointly over the causal local past
     window ``[max(0, i - r), i]`` and strictly past hub summaries using a
     causal prefix-logsumexp Sinkhorn operator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .gates import causal_threshold_gate


@dataclass(frozen=True)
class CausalTHOTConfig:
    """Configuration for the chunked strictly causal THOT operator."""

    chunk_size: int = 32
    radius: int = 4
    hubs: int = 16
    base_steps: int = 1
    tail_steps: int = 2
    eps_factor: float = 1.5
    gamma_write: float = 0.70
    gamma_read: float = 0.25
    prior_mix: float = 0.05
    hub_total_mass: float = 8.0
    write_key_residual: float = 0.80
    remat_chunks: int = 256


def _safe_lse(x: jnp.ndarray, mask: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    mask_b = jnp.broadcast_to(mask, x.shape)
    x_safe = jnp.where(mask_b, x, 0.0)
    has_any = jnp.any(mask_b, axis=axis, keepdims=True)
    m = jax.lax.stop_gradient(
        jnp.max(jnp.where(mask_b, x_safe, -jnp.inf), axis=axis, keepdims=True)
    )
    m_safe = jnp.where(has_any, m, 0.0)
    ex = jnp.where(mask_b, jnp.exp(jnp.where(mask_b, x_safe - m_safe, 0.0)), 0.0)
    s = jnp.sum(ex, axis=axis, keepdims=True)
    res = jnp.where(has_any, m_safe + jnp.log(jnp.maximum(s, 1e-30)), -jnp.inf)
    return jnp.squeeze(res, axis=axis)


def _causal_prefix_lse(
    x: jnp.ndarray, mask: jnp.ndarray, axis: int = -2, *, exclusive: bool = True
) -> jnp.ndarray:
    """Strictly causal logsumexp along ``axis`` without ``-inf`` arithmetic."""
    safe_x = jnp.where(mask, x, 0.0)

    def combine(lhs, rhs):
        a1, m1, s1 = lhs
        a2, m2, s2 = rhs
        a = a1 | a2
        m = jnp.where(a1 & a2, jnp.maximum(m1, m2), jnp.where(a1, m1, m2))
        e1 = jnp.where(a1, jnp.exp(jnp.where(a1, m1 - m, 0.0)), 0.0)
        e2 = jnp.where(a2, jnp.exp(jnp.where(a2, m2 - m, 0.0)), 0.0)
        s = s1 * e1 + s2 * e2
        return a, m, s

    init = (mask, safe_x, mask.astype(x.dtype))
    a_inc, m_inc, s_inc = jax.lax.associative_scan(combine, init, axis=axis)
    if exclusive:
        z_shape = list(x.shape)
        z_shape[axis] = 1
        pad_b = jnp.zeros(z_shape, dtype=bool)
        pad_f = jnp.zeros(z_shape, dtype=x.dtype)
        slc = [slice(None)] * x.ndim
        slc[axis] = slice(0, x.shape[axis] - 1)
        a_out = jnp.concatenate((pad_b, a_inc[tuple(slc)]), axis=axis)
        m_out = jnp.concatenate((pad_f, m_inc[tuple(slc)]), axis=axis)
        s_out = jnp.concatenate((pad_f, s_inc[tuple(slc)]), axis=axis)
    else:
        a_out, m_out, s_out = a_inc, m_inc, s_inc
    return jnp.where(a_out, m_out + jnp.log(jnp.maximum(s_out, 1e-30)), -jnp.inf)


def _exclusive_cumsum(x: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    inc = jnp.cumsum(x, axis=axis)
    pad_shape = list(inc.shape)
    pad_shape[axis] = 1
    zeros = jnp.zeros(pad_shape, dtype=inc.dtype)
    slc = [slice(None)] * inc.ndim
    slc[axis] = slice(0, inc.shape[axis] - 1)
    return jnp.concatenate((zeros, inc[tuple(slc)]), axis=axis)


def _extract_causal_local_chunks(
    k: jnp.ndarray,
    v: jnp.ndarray,
    b: jnp.ndarray,
    valid: jnp.ndarray,
    chunk_size: int,
    radius: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract ``[N_c, r + B, ...]`` local past slices and causal ``[N_c, B, r + B]`` mask."""
    l_pad, _ = k.shape
    n_chunks = l_pad // chunk_size
    r = min(radius, l_pad - 1) if l_pad > 1 else 0

    k_lp = jnp.pad(k, ((r, 0), (0, 0)))
    v_lp = jnp.pad(v, ((r, 0), (0, 0)))
    b_lp = jnp.pad(b, ((r, 0),))
    val_lp = jnp.pad(valid, ((r, 0),), constant_values=False)

    c_off = jnp.arange(n_chunks, dtype=jnp.int32)[:, None] * chunk_size
    w_off = jnp.arange(r + chunk_size, dtype=jnp.int32)[None, :]
    idx = c_off + w_off

    k_loc = k_lp[idx]
    v_loc = v_lp[idx]
    b_loc = b_lp[idx]
    val_loc = val_lp[idx]

    qi = jnp.arange(chunk_size, dtype=jnp.int32)[:, None]
    kj = jnp.arange(r + chunk_size, dtype=jnp.int32)[None, :]
    causal_band = (kj >= qi) & (kj <= qi + r)
    local_edge_mask = causal_band[None, :, :] & val_loc[:, None, :] & (b_loc[:, None, :] > 0)
    return k_loc, v_loc, b_loc, local_edge_mask


def causal_thot_head(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    q_hub: jnp.ndarray,
    k_hub: jnp.ndarray,
    v_hub: jnp.ndarray,
    bias: jnp.ndarray,
    cfg: CausalTHOTConfig = CausalTHOTConfig(),
    *,
    k_write: jnp.ndarray | None = None,
    scale: float | None = None,
) -> jnp.ndarray:
    """Strictly causal block-hub annealed OT for one attention head.

    Args:
        q, k, v: Token tensors of shape ``[L, D]``.
        a, b: Positive capacities of shape ``[L]``.
        q_hub, k_hub, v_hub: Hub slot parameters of shape ``[K, D]``.
        bias: Node-type bias matrix of shape ``[2, 2]``.
        cfg: ``CausalTHOTConfig`` hyperparameters.
        k_write: Optional position-invariant keys of shape ``[L, D]`` for
            writing summaries into hubs.
        scale: Attention logit scale (defaults to ``1 / sqrt(D)``).

    Returns:
        Causal attended outputs of shape ``[L, D]`` in ``q.dtype``.
    """
    length, dim = q.shape
    hubs = q_hub.shape[0]
    B = cfg.chunk_size
    n_chunks = (length + B - 1) // B
    l_pad = n_chunks * B
    pad_len = l_pad - length
    sc = (1.0 / math.sqrt(dim)) if scale is None else float(scale)

    kw = k if k_write is None else k_write
    if pad_len > 0:
        q_p = jnp.pad(q, ((0, pad_len), (0, 0)))
        k_p = jnp.pad(k, ((0, pad_len), (0, 0)))
        kw_p = jnp.pad(kw, ((0, pad_len), (0, 0)))
        v_p = jnp.pad(v, ((0, pad_len), (0, 0)))
        a_p = jnp.pad(a, ((0, pad_len),))
        b_p = jnp.pad(b, ((0, pad_len),))
        valid_p = jnp.arange(l_pad) < length
    else:
        q_p, k_p, kw_p, v_p, a_p, b_p = q, k, kw, v, a, b
        valid_p = jnp.ones((l_pad,), dtype=bool)

    kw_c = kw_p.reshape(n_chunks, B, dim).astype(jnp.float32)
    v_c = v_p.reshape(n_chunks, B, dim).astype(jnp.float32)
    b_c = b_p.reshape(n_chunks, B).astype(jnp.float32)
    val_c = valid_p.reshape(n_chunks, B)
    act_bw = val_c & (b_c > 0)

    qh_f = q_hub.astype(jnp.float32)
    kh_f = k_hub.astype(jnp.float32)
    vh_f = v_hub.astype(jnp.float32)

    dot_w = jnp.einsum("kd,cbd->ckb", qh_f, kw_c, precision=jax.lax.Precision.HIGHEST)
    s_w_full = sc * dot_w + bias[1, 0]
    s_w_base = (sc / cfg.eps_factor) * dot_w + bias[1, 0]

    chunk_mass = jnp.sum(jnp.where(act_bw, b_c, 0.0), axis=-1, keepdims=True)
    max_b_c = jnp.max(jnp.where(act_bw, b_c, 0.0), axis=-1, keepdims=True)
    cap_w = jnp.maximum(jnp.maximum(chunk_mass / float(hubs), max_b_c), 1e-12)
    log_cap_w = jnp.log(cap_w)
    log_b_w = jnp.where(act_bw, jnp.log(jnp.where(act_bw, b_c, 1.0)), -jnp.inf)

    mask_w = act_bw[:, None, :]
    mask_w_full = jnp.broadcast_to(mask_w, s_w_full.shape)
    v_w = jnp.where(act_bw, log_b_w, 0.0)
    u_w = jnp.zeros((n_chunks, hubs), dtype=jnp.float32)

    for _ in range(cfg.base_steps):
        u_w = -cfg.gamma_write * jax.nn.relu(_safe_lse(s_w_base + v_w[:, None, :], mask_w, axis=-1) - log_cap_w)
        v_w = jnp.where(
            act_bw,
            log_b_w - _safe_lse(s_w_base + u_w[:, :, None], mask_w_full, axis=-2),
            0.0,
        )
    if cfg.base_steps > 0:
        u_w = jax.lax.stop_gradient(u_w)
        v_w = jax.lax.stop_gradient(v_w)

    for _ in range(cfg.tail_steps):
        u_w = -cfg.gamma_write * jax.nn.relu(_safe_lse(s_w_full + v_w[:, None, :], mask_w, axis=-1) - log_cap_w)
        v_w = jnp.where(
            act_bw,
            log_b_w - _safe_lse(s_w_full + u_w[:, :, None], mask_w_full, axis=-2),
            0.0,
        )
    v_w = jnp.where(
        act_bw,
        log_b_w - _safe_lse(s_w_full + u_w[:, :, None], mask_w_full, axis=-2),
        0.0,
    )
    log_p_w = jnp.where(mask_w, s_w_full + u_w[:, :, None] + v_w[:, None, :], 0.0)
    p_w = jnp.where(mask_w, jnp.exp(jnp.where(mask_w, log_p_w, 0.0)), 0.0)

    w_val = jnp.einsum("ckb,cbd->ckd", p_w, v_c, precision=jax.lax.Precision.HIGHEST)
    w_key = jnp.einsum("ckb,cbd->ckd", p_w, kw_c, precision=jax.lax.Precision.HIGHEST)
    z_hub = jnp.sum(p_w, axis=-1)

    m_val_past = _exclusive_cumsum(w_val, axis=0)
    m_key_past = _exclusive_cumsum(w_key, axis=0)
    z_past = _exclusive_cumsum(z_hub, axis=0)

    m0 = cfg.prior_mix
    denom = m0 + z_past[..., None]
    v_hub_past = (m0 * vh_f[None, :, :] + m_val_past) / denom
    k_written_past = (m0 * kh_f[None, :, :] + m_key_past) / denom
    beta = cfg.write_key_residual
    k_hub_past = (1.0 - beta) * kh_f[None, :, :] + beta * k_written_past

    z_tot = jnp.sum(z_past, axis=-1, keepdims=True)
    eff_hub_tot = jnp.minimum(z_tot, cfg.hub_total_mass)
    b_hub_past = eff_hub_tot * (z_past / jnp.maximum(z_tot, 1e-12))
    chunk_has_past = (jnp.arange(n_chunks) > 0)[:, None] & (z_past > 0.0)

    k_loc, v_loc, b_loc, loc_mask = _extract_causal_local_chunks(
        k_p.astype(jnp.float32), v_p.astype(jnp.float32), b_p.astype(jnp.float32), valid_p, B, cfg.radius
    )
    q_c = q_p.reshape(n_chunks, B, dim).astype(jnp.float32)
    a_c = a_p.reshape(n_chunks, B).astype(jnp.float32)
    act_a = val_c & (a_c > 0)
    log_a_c = jnp.where(act_a, jnp.log(jnp.where(act_a, a_c, 1.0)), -jnp.inf)

    def read_block(inputs):
        qc, ac, log_ac, act_ac, kl, vl, bl, lmask, kh_c, vh_c, bh_c, hvalid = inputs
        dot_loc = jnp.einsum("mbd,msd->mbs", qc, kl, precision=jax.lax.Precision.HIGHEST)
        dot_hub = jnp.einsum("mbd,mkd->mbk", qc, kh_c, precision=jax.lax.Precision.HIGHEST)

        s_loc_full = sc * dot_loc + bias[0, 0]
        s_hub_full = sc * dot_hub + bias[0, 1]
        s_full = jnp.concatenate((s_loc_full, s_hub_full), axis=-1)

        s_loc_base = (sc / cfg.eps_factor) * dot_loc + bias[0, 0]
        s_hub_base = (sc / cfg.eps_factor) * dot_hub + bias[0, 1]
        s_base = jnp.concatenate((s_loc_base, s_hub_base), axis=-1)

        hub_mask = jnp.broadcast_to(hvalid[:, None, :], (qc.shape[0], B, hubs))
        edge_mask = jnp.concatenate((lmask, hub_mask), axis=-1) & act_ac[:, :, None]

        b_all = jnp.concatenate((bl, bh_c), axis=-1)
        col_active = jnp.any(edge_mask, axis=-2) & (b_all > 0)
        log_b_all = jnp.where(col_active, jnp.log(jnp.where(col_active, b_all, 1.0)), 0.0)

        v_all = jnp.concatenate((vl, vh_c), axis=-2)

        v_r = jnp.where(edge_mask, log_b_all[:, None, :], 0.0)
        u_r = jnp.zeros((qc.shape[0], B), dtype=jnp.float32)

        def sweep_step(s_mat, u_prev, v_prev):
            u_new = jnp.where(
                act_ac,
                jnp.where(act_ac, log_ac, 0.0) - _safe_lse(s_mat + jnp.where(edge_mask, v_prev, 0.0), edge_mask, axis=-1),
                0.0,
            )
            col_lse = _causal_prefix_lse(s_mat + jnp.where(edge_mask, u_new[:, :, None], 0.0), edge_mask, axis=-2, exclusive=True)
            v_new = jnp.where(
                edge_mask,
                log_b_all[:, None, :] - cfg.gamma_read * jnp.where(jnp.isfinite(col_lse), col_lse, 0.0),
                0.0,
            )
            return u_new, v_new

        for _ in range(cfg.base_steps):
            u_r, v_r = sweep_step(s_base, u_r, v_r)
        if cfg.base_steps > 0:
            u_r = jax.lax.stop_gradient(u_r)
            v_r = jax.lax.stop_gradient(v_r)

        for _ in range(cfg.tail_steps):
            u_r, v_r = sweep_step(s_full, u_r, v_r)

        u_fin = jnp.where(
            act_ac,
            jnp.where(act_ac, log_ac, 0.0) - _safe_lse(s_full + jnp.where(edge_mask, v_r, 0.0), edge_mask, axis=-1),
            0.0,
        )
        logits_r = jnp.where(edge_mask, s_full + u_fin[:, :, None] + v_r, 0.0)
        p_r = jnp.where(edge_mask, jnp.exp(logits_r), 0.0)
        raw_out = jnp.einsum("mbs,msd->mbd", p_r, v_all, precision=jax.lax.Precision.HIGHEST)
        return jnp.where(act_ac[:, :, None], raw_out / jnp.maximum(ac[:, :, None], 1e-30), 0.0)

    blk_inputs = (
        q_c, a_c, log_a_c, act_a, k_loc, v_loc, b_loc, loc_mask,
        k_hub_past, v_hub_past, b_hub_past, chunk_has_past,
    )
    if n_chunks > cfg.remat_chunks and n_chunks % cfg.remat_chunks == 0:
        n_mac = n_chunks // cfg.remat_chunks
        grouped = jax.tree.map(lambda x: x.reshape((n_mac, cfg.remat_chunks) + x.shape[1:]), blk_inputs)
        out_grouped = jax.lax.map(jax.checkpoint(read_block), grouped)
        out_c = out_grouped.reshape(n_chunks, B, dim)
    else:
        out_c = read_block(blk_inputs)

    return out_c.reshape(l_pad, dim)[:length].astype(q.dtype)


def causal_thot_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    q_hub: jnp.ndarray,
    k_hub: jnp.ndarray,
    v_hub: jnp.ndarray,
    bias: jnp.ndarray | None = None,
    *,
    cfg: CausalTHOTConfig = CausalTHOTConfig(),
    k_write: jnp.ndarray | None = None,
    scale: float | None = None,
) -> jnp.ndarray:
    """Batched multi-head strictly causal THOT attention.

    Args:
        q, k, v: Token tensors of shape ``[B, L, H, D]``.
        a, b: Positive capacities of shape ``[B, L, H]``.
        q_hub, k_hub, v_hub: Hub tensors of shape ``[B, K, H, D]`` or
            ``[K, H, D]`` (broadcast across the batch dimension).
        bias: Optional node-type bias of shape ``[2, 2]`` or ``[H, 2, 2]``
            (defaults to zeros).
        cfg: ``CausalTHOTConfig`` hyperparameters.
        k_write: Optional position-invariant keys ``[B, L, H, D]`` for writing
            to hubs.
        scale: Optional logit scaling factor (defaults to ``1 / sqrt(D)``).

    Returns:
        Strictly causal outputs of shape ``[B, L, H, D]`` in ``q.dtype``.
    """
    if q.ndim != 4 or q.shape != k.shape or v.shape != q.shape:
        raise ValueError(f"Expected 4D [B, L, H, D] q, k, v, got {q.shape}, {k.shape}, {v.shape}")
    bsz, length, heads, dim = q.shape
    if a.shape != (bsz, length, heads) or b.shape != (bsz, length, heads):
        raise ValueError(f"Expected a, b shape {(bsz, length, heads)}, got {a.shape}, {b.shape}")
    if q_hub.shape != k_hub.shape or q_hub.shape != v_hub.shape:
        raise ValueError("q_hub, k_hub, v_hub must have matching shapes")

    def _expand_hub(h_tensor: jnp.ndarray) -> jnp.ndarray:
        if h_tensor.ndim == 3:
            if h_tensor.shape[1:] != (heads, dim):
                raise ValueError(f"Expected hub shape [K, {heads}, {dim}], got {h_tensor.shape}")
            return jnp.broadcast_to(h_tensor[None], (bsz,) + h_tensor.shape)
        if h_tensor.ndim == 4:
            if h_tensor.shape[0] != bsz or h_tensor.shape[2:] != (heads, dim):
                raise ValueError(f"Expected hub shape [{bsz}, K, {heads}, {dim}], got {h_tensor.shape}")
            return h_tensor
        raise ValueError(f"Expected 3D or 4D hub tensor, got shape {h_tensor.shape}")

    qh = _expand_hub(q_hub)
    kh = _expand_hub(k_hub)
    vh = _expand_hub(v_hub)

    if bias is None:
        bias_h = jnp.zeros((heads, 2, 2), dtype=jnp.float32)
    elif bias.shape == (2, 2):
        bias_h = jnp.broadcast_to(bias.astype(jnp.float32)[None], (heads, 2, 2))
    elif bias.shape == (heads, 2, 2):
        bias_h = bias.astype(jnp.float32)
    else:
        raise ValueError(f"Incompatible bias shape {bias.shape}; expected (2, 2) or ({heads}, 2, 2)")

    kw = k if k_write is None else k_write
    if kw.shape != k.shape:
        raise ValueError(f"Expected k_write shape {k.shape}, got {kw.shape}")

    def per_batch_head(qb, kb, vb, kw_b, ab, bb, qhb, khb, vhb, bi):
        return causal_thot_head(
            qb, kb, vb, ab, bb, qhb, khb, vhb, bi, cfg, k_write=kw_b, scale=scale
        )

    def per_batch(qb, kb, vb, kw_b, ab, bb, qhb, khb, vhb):
        return jax.vmap(per_batch_head, in_axes=(1, 1, 1, 1, 1, 1, 1, 1, 1, 0), out_axes=1)(
            qb, kb, vb, kw_b, ab, bb, qhb, khb, vhb, bias_h
        )

    return jax.lax.map(lambda xs: per_batch(*xs), (q, k, v, kw, a, b, qh, kh, vh))


def verify_causal_invariance(
    *,
    length: int = 100,
    heads: int = 2,
    dim: int = 16,
    hubs: int = 8,
    chunk_size: int = 32,
    radius: int = 4,
    seed: int = 20261119,
) -> dict[str, Any]:
    """Verify exact 0.0 future leakage (perturbation + JAX Jacobian) and finite VJP."""
    rng = np.random.default_rng(seed)
    cfg = CausalTHOTConfig(chunk_size=chunk_size, radius=radius, hubs=hubs, remat_chunks=2)

    q = jnp.asarray(0.4 * rng.standard_normal((1, length, heads, dim)), jnp.float32)
    k = jnp.asarray(0.4 * rng.standard_normal((1, length, heads, dim)), jnp.float32)
    v = jnp.asarray(0.4 * rng.standard_normal((1, length, heads, dim)), jnp.float32)
    rec_s = jnp.asarray(rng.standard_normal((1, length, heads)), jnp.float32)
    src_s = jnp.asarray(rng.standard_normal((1, length, heads)), jnp.float32)
    theta_rec = jnp.asarray(-0.25, jnp.float32)
    theta_src = jnp.asarray(-0.25, jnp.float32)
    q_hub = jnp.asarray(0.3 * rng.standard_normal((1, hubs, heads, dim)), jnp.float32)
    k_hub = jnp.asarray(0.3 * rng.standard_normal((1, hubs, heads, dim)), jnp.float32)
    v_hub = jnp.asarray(0.3 * rng.standard_normal((1, hubs, heads, dim)), jnp.float32)
    bias = jnp.asarray(0.2 * rng.standard_normal((heads, 2, 2)), jnp.float32)

    cutoffs = sorted({1, chunk_size - 1, chunk_size, chunk_size + 1, length - 1})
    cutoffs = [c for c in cutoffs if 0 < c < length]

    a_thr, b_thr = causal_threshold_gate(rec_s, src_s, theta_rec, theta_src, length)
    rec_s_pert = rec_s.at[:, chunk_size - 1 :, :].set(9.0)
    src_s_pert = src_s.at[:, chunk_size - 1 :, :].set(9.0)
    a_thr_p, b_thr_p = causal_threshold_gate(rec_s_pert, src_s_pert, theta_rec, theta_src, length)
    thr_leak = float(
        max(
            jnp.max(jnp.abs(a_thr_p[:, : chunk_size - 1] - a_thr[:, : chunk_size - 1])),
            jnp.max(jnp.abs(b_thr_p[:, : chunk_size - 1] - b_thr[:, : chunk_size - 1])),
        )
    )
    filler_scores = jnp.full((1, 10, 1), -0.5, dtype=jnp.float32)
    _, b_fill = causal_threshold_gate(filler_scores, filler_scores, 0.0, 0.0, 10)
    filler_excess = float(jnp.max(b_fill - 1.0 / 10.0))

    def run_fn(qq, kk, vv, rs, ss, tr, ts, qh, kh, vh, bi):
        aa, bb = causal_threshold_gate(rs, ss, tr, ts, length)
        return causal_thot_attention(qq, kk, vv, aa, bb, qh, kh, vh, bi, cfg=cfg)

    base_fn = jax.jit(run_fn)
    y_base = jax.block_until_ready(
        base_fn(q, k, v, rec_s, src_s, theta_rec, theta_src, q_hub, k_hub, v_hub, bias)
    )

    max_pert_leak = 0.0
    for t_cut in cutoffs:
        q_p = q.at[:, t_cut:, :, :].add(2.5)
        k_p = k.at[:, t_cut:, :, :].add(-1.7)
        v_p = v.at[:, t_cut:, :, :].set(3.1)
        rs_p = rec_s.at[:, t_cut:, :].set(4.0)
        ss_p = src_s.at[:, t_cut:, :].set(4.0)
        y_pert = jax.block_until_ready(
            base_fn(q_p, k_p, v_p, rs_p, ss_p, theta_rec, theta_src, q_hub, k_hub, v_hub, bias)
        )
        max_pert_leak = max(max_pert_leak, float(jnp.max(jnp.abs(y_pert[:, :t_cut] - y_base[:, :t_cut]))))

    def sum_per_t(vv, kk, ss):
        out = run_fn(q, kk, vv, rec_s, ss, theta_rec, theta_src, q_hub, k_hub, v_hub, bias)
        return jnp.sum(out[0], axis=(1, 2))

    jac_v, jac_k, jac_s = jax.jit(jax.jacrev(sum_per_t, argnums=(0, 1, 2)))(v, k, src_s)
    t_idx = jnp.arange(length)[:, None]
    tp_idx = jnp.arange(length)[None, :]
    upper = t_idx < tp_idx
    jac_leak = float(
        max(
            jnp.max(jnp.where(upper[:, None, :, None, None], jnp.abs(jac_v), 0.0)),
            jnp.max(jnp.where(upper[:, None, :, None, None], jnp.abs(jac_k), 0.0)),
            jnp.max(jnp.where(upper[:, None, :, None], jnp.abs(jac_s), 0.0)),
        )
    )

    cot = jnp.asarray(rng.standard_normal(y_base.shape) * 0.1, jnp.float32)

    def loss_all(*xs):
        return jnp.sum(run_fn(*xs) * cot)

    primals = (q, k, v, rec_s, src_s, theta_rec, theta_src, q_hub, k_hub, v_hub, bias)
    val, grads = jax.block_until_ready(
        jax.jit(jax.value_and_grad(loss_all, argnums=tuple(range(len(primals)))))(*primals)
    )
    grad_leaves = jax.tree.leaves(grads)
    grad_finite = bool(jnp.isfinite(val) and all(bool(jnp.isfinite(g).all()) for g in grad_leaves))
    qkv_norm = float(jnp.sqrt(jnp.sum(grads[0] ** 2) + jnp.sum(grads[1] ** 2) + jnp.sum(grads[2] ** 2)))

    passed = bool(
        thr_leak == 0.0
        and filler_excess == 0.0
        and max_pert_leak == 0.0
        and jac_leak == 0.0
        and grad_finite
        and qkv_norm > 0.0
        and bool(jnp.isfinite(y_base).all())
    )
    return dict(
        passed=passed,
        length=length,
        chunk_size=chunk_size,
        radius=radius,
        threshold_future_leak=thr_leak,
        threshold_filler_excess_mass=filler_excess,
        max_perturbation_leak=max_pert_leak,
        max_jacobian_leak=jac_leak,
        grad_finite=grad_finite,
        qkv_grad_norm=qkv_norm,
    )

