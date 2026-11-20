"""Fast batched Triton/Pallas kernels and pure-JAX reference for THOT.

Implements the 1+2 annealed row-closed bipartite Sinkhorn attention operator
("Topology Hub Optimal Transport", THOT) together with a Two-Call
One-Reference-Tile custom VJP executed through batched multi-head kernels:

  - Split-factor backward accumulators (``left_d, right_d`` of width ``D`` and
    ``left_s, right_s`` of width ``2R+1``) avoid ``next_power_of_2(D + 2R + 1)``
    register padding overhead.
  - Batched multi-head grids eliminate per-head ``jax.vmap`` wrappers and
    redundant HBM pad/concat copies.
  - Dedicated 1D scalar weighted-matvec path and fused in-kernel ``dbias``
    reduction avoid materializing auxiliary score indicators.
  - Explicit ``dot_precision`` toggle between ``"bf16_tc"`` Tensor Cores and
    ``"fp32"``, while retaining FP32 dual potentials and LSE accumulators.
"""

from __future__ import annotations

from functools import lru_cache
import inspect
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt


def _pallas_opts(interpret: bool, num_warps: int = 4) -> dict:
    opts: dict = {}
    if not interpret:
        opts["compiler_params"] = plt.CompilerParams(num_warps=num_warps)
        if "backend" in inspect.signature(pl.pallas_call).parameters:
            opts["backend"] = "triton"
    return opts


def _dot_fn(dot_precision: str):
    if dot_precision == "fp32":
        return lambda a, b, trans_b=False: pl.dot(
            a, b, trans_b=trans_b, precision=jax.lax.Precision.HIGHEST
        )
    if dot_precision == "bf16_tc":
        return lambda a, b, trans_b=False: pl.dot(
            a, b, trans_b=trans_b, precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32
        )
    raise ValueError(f"Unsupported dot_precision: {dot_precision!r}")


def _choose_tiles(
    length: int,
    hubs: int,
    radius: int,
    dim: int,
    feature: int,
    *,
    lse: bool,
    scalar: bool,
    factor: bool,
    dot_precision: str,
) -> tuple[int, int, int, int, int, int, int]:
    """Select ``(rq_real, sk_band, sk_spokes, rq_hub, sk_hub, chunk, fblock)``."""
    if lse or scalar:
        rq_real = 32 if (dot_precision == "fp32" and dim > 32) else 64
        sk_band = 64 if (radius >= 16 and not (dot_precision == "fp32" and dim > 32)) else 32
    elif factor:
        if dim <= 32 and dot_precision == "bf16_tc":
            rq_real = 64 if radius >= 16 else 32
        else:
            rq_real = 32
        sk_band = 32
    else:
        if dot_precision == "bf16_tc" and (dim <= 32 or (length >= 2048 and dim <= 64)):
            rq_real = 64
        else:
            rq_real = 32
        sk_band = 64 if (radius >= 16 and dim <= 32 and dot_precision == "bf16_tc") else 32

    rq_real = min(rq_real, max(16, pl.next_power_of_2(length)))
    sk_spokes = min(32, max(16, pl.next_power_of_2(max(1, hubs))))
    rq_hub = min(64, max(16, pl.next_power_of_2(max(1, hubs))))
    sk_hub = 32 if (factor and (dim > 32 or dot_precision == "fp32")) else 64
    chunk = 256 if (length + hubs <= 8192) else 512
    fblock = 1 if (lse or scalar) else min(64, max(16, pl.next_power_of_2(feature)))
    return rq_real, sk_band, sk_spokes, rq_hub, sk_hub, chunk, fblock


def _validate_batched(q, k, other, bias, length, hubs, radius, dot_precision):
    if dot_precision not in ("fp32", "bf16_tc"):
        raise ValueError(f"Unsupported dot_precision: {dot_precision!r}")
    if length < 1 or not 0 <= hubs <= 128 or radius < 0:
        raise ValueError("Expected L >= 1, 0 <= hubs <= 128, and radius >= 0")
    if q.ndim != 3 or q.shape != k.shape or q.shape[1] < length + hubs or q.shape[2] < 1:
        raise ValueError(f"Expected q, k with shape [H, N>={length + hubs}, D], got {q.shape}")
    heads, n_phys, _ = q.shape
    if other.shape[0] != heads or other.shape[1] < length + hubs or other.ndim != 2:
        raise ValueError(f"Expected dual tensor with shape [H, >={length + hubs}], got {other.shape}")
    if bias.shape != (heads, 2, 2):
        raise ValueError(f"Expected bias with shape ({heads}, 2, 2), got {bias.shape}")
    return heads, n_phys


def _load_rows_3d(ref, h, ix, valid, cols, width, target_dtype):
    arr = plt.load(
        ref.at[h, ix[:, None], cols[None, :]],
        mask=valid[:, None] & (cols[None, :] < width),
        other=0.0,
    )
    return arr.astype(target_dtype) if arr.dtype != target_dtype else arr


def _init_tile_state(rq: int, fblock: int, *, lse: bool, scalar: bool, return_bias_grads: bool):
    if lse:
        return (
            jnp.full((rq,), -jnp.inf, jnp.float32),
            jnp.zeros((rq,), jnp.float32),
        )
    acc = jnp.zeros((rq,), jnp.float32) if scalar else jnp.zeros((rq, fblock), jnp.float32)
    if return_bias_grads:
        z = jnp.asarray(0.0, jnp.float32)
        return acc, z, z
    return acc


def _sparse_batched(
    q,
    k,
    own,
    other,
    payload,
    bias,
    left_d,
    right_d,
    left_s,
    right_s,
    *,
    length: int,
    hubs: int,
    radius: int,
    scale: float,
    dot_precision: str,
    interpret: bool,
    lse: bool,
    scalar: bool,
    return_bias_grads: bool,
    out_dtype,
):
    """Batched multi-head sparse kernel with split backward factors."""
    heads, _, dim = q.shape
    live = length + hubs
    radius_eff = min(radius, length // 2)
    feature = 1 if (lse or scalar) else payload.shape[-1]
    has_d = left_d is not None
    has_s = left_s is not None
    factor = has_d or has_s

    rq_real, sk_band, sk_spokes, rq_hub, sk_hub, chunk, fblock = _choose_tiles(
        length, hubs, radius_eff, dim, feature,
        lse=lse, scalar=scalar, factor=factor, dot_precision=dot_precision,
    )

    channels = max(16, pl.next_power_of_2(dim))
    adim_d = max(16, pl.next_power_of_2(left_d.shape[-1])) if has_d else 0
    adim_s = max(16, pl.next_power_of_2(left_s.shape[-1])) if has_s else 0
    width_d = left_d.shape[-1] if has_d else 0
    width_s = left_s.shape[-1] if has_s else 0

    dot = _dot_fn(dot_precision)
    dot_dtype = jnp.bfloat16 if dot_precision == "bf16_tc" else jnp.float32
    res_dtype = jnp.float32 if out_dtype is None else jnp.dtype(out_dtype)

    dummy3 = jnp.zeros((1, 1, 1), jnp.float32)
    dummy2 = jnp.zeros((1, 1), jnp.float32)
    payload_arg = dummy2 if lse else payload

    opts = _pallas_opts(interpret)
    f_tiles = 1 if (lse or scalar) else pl.cdiv(feature, fblock)
    nchunks = pl.cdiv(live, chunk) if hubs else 0
    real_blocks = pl.cdiv(length, rq_real)
    span = min(length, rq_real + 2 * radius_eff)

    def make_tile_step(
        h, validq, qq, uv, dest, ll_d, ll_s, kr, vr, pr, rdr, rsr, sd, ad_d, ad_s
    ):
        def tile_step(state, ix, valid, edges, bb, *, is_spoke=False, is_hub_col=None):
            kk = _load_rows_3d(kr, h, ix, valid, sd, dim, dot_dtype)
            vv = plt.load(vr.at[h, ix], mask=valid, other=-jnp.inf).astype(jnp.float32)
            scores = dot(qq, kk, trans_b=True) * scale + bb + vv[None, :]
            mask = validq[:, None] & valid[None, :] & edges
            if lse:
                oldmax, total = state
                a = jnp.where(mask, scores, -jnp.inf)
                mx = jnp.maximum(oldmax, jnp.max(a, axis=1))
                shift = jnp.where(mx != -jnp.inf, mx, 0.0)
                total = total * jnp.exp(oldmax - shift) + jnp.sum(
                    jnp.exp(a - shift[:, None]), axis=1
                )
                return mx, total

            weight = jnp.exp(jnp.where(mask, scores + uv[:, None], -jnp.inf))
            if has_d or has_s:
                fdot = jnp.zeros_like(weight)
                if has_d:
                    rt_d = _load_rows_3d(rdr, h, ix, valid, ad_d, width_d, dot_dtype)
                    fdot = fdot + dot(ll_d, rt_d, trans_b=True)
                if has_s:
                    rt_s = _load_rows_3d(rsr, h, ix, valid, ad_s, width_s, dot_dtype)
                    fdot = fdot + dot(ll_s, rt_s, trans_b=True)
                weight = weight * fdot

            if return_bias_grads:
                acc, br_sum, bh_sum = state
                if is_hub_col is not None:
                    br_sum = br_sum + jnp.sum(jnp.where(is_hub_col[None, :], 0.0, weight))
                    bh_sum = bh_sum + jnp.sum(jnp.where(is_hub_col[None, :], weight, 0.0))
                elif is_spoke:
                    bh_sum = bh_sum + jnp.sum(weight)
                else:
                    br_sum = br_sum + jnp.sum(weight)
            else:
                acc = state

            if scalar:
                vals = plt.load(pr.at[h, ix], mask=valid, other=0.0).astype(jnp.float32)
                acc_new = acc + jnp.sum(weight * vals[None, :], axis=1)
            else:
                vals = _load_rows_3d(pr, h, ix, valid, dest, feature, dot_dtype)
                acc_new = acc + dot(weight.astype(dot_dtype), vals)

            return (acc_new, br_sum, bh_sum) if return_bias_grads else acc_new

        return tile_step

    def setup_query(h, row, validq, qr, ur, ldr, lsr, f_id):
        sd = jnp.arange(channels)
        qq = _load_rows_3d(qr, h, row, validq, sd, dim, dot_dtype)
        uv = None if lse else plt.load(ur.at[h, row], mask=validq, other=-jnp.inf).astype(jnp.float32)
        dest = None if (lse or scalar) else (f_id * fblock + jnp.arange(fblock))
        ad_d = jnp.arange(adim_d) if has_d else None
        ll_d = _load_rows_3d(ldr, h, row, validq, ad_d, width_d, dot_dtype) if has_d else None
        ad_s = jnp.arange(adim_s) if has_s else None
        ll_s = _load_rows_3d(lsr, h, row, validq, ad_s, width_s, dot_dtype) if has_s else None
        return sd, qq, uv, dest, ad_d, ll_d, ad_s, ll_s

    def run_hub_partial():
        args = (
            q, k, own, other, bias, payload_arg,
            left_d if has_d else dummy3, right_d if has_d else dummy3,
            left_s if has_s else dummy3, right_s if has_s else dummy3,
        )

        def hub_kernel(qr, kr, ur, vr, br, pr, ldr, rdr, lsr, rsr, *outs):
            out_ref = outs[0]
            pid_q = pl.program_id(0)
            if lse or scalar:
                f_id, part, h = 0, pl.program_id(1), pl.program_id(2)
            else:
                f_id, part, h = pl.program_id(1), pl.program_id(2), pl.program_id(3)

            qi = pid_q * rq_hub + jnp.arange(rq_hub)
            validq = qi < hubs
            row = qi + length
            sd, qq, uv, dest, ad_d, ll_d, ad_s, ll_s = setup_query(
                h, row, validq, qr, ur, ldr, lsr, f_id
            )
            tile_step = make_tile_step(
                h, validq, qq, uv, dest, ll_d, ll_s, kr, vr, pr, rdr, rsr, sd, ad_d, ad_s
            )

            b10, b11 = br[h, 1, 0], br[h, 1, 1]
            js = jnp.arange(sk_hub)

            def body(t, state):
                ix = part * chunk + t * sk_hub + js
                valid = ix < live
                is_hub_col = ix >= length
                bb = jnp.where(is_hub_col, b11, b10)
                return tile_step(state, ix, valid, True, bb[None, :], is_hub_col=is_hub_col)

            init_state = _init_tile_state(
                rq_hub, fblock, lse=lse, scalar=scalar, return_bias_grads=return_bias_grads
            )
            niters = min(pl.cdiv(live, sk_hub), chunk // sk_hub)
            end_state = jax.lax.fori_loop(0, niters, body, init_state)

            if lse:
                mx, total = end_state
                ans = mx + jnp.log(jnp.maximum(total, 1.0))
                plt.store(out_ref.at[h, part, qi], ans, mask=validq)
            elif scalar:
                acc = end_state[0] if return_bias_grads else end_state
                plt.store(out_ref.at[h, part, qi], acc, mask=validq)
            else:
                acc = end_state[0] if return_bias_grads else end_state
                plt.store(
                    out_ref.at[h, part, qi[:, None], dest[None, :]],
                    acc,
                    mask=validq[:, None] & (dest[None, :] < feature),
                )

            if return_bias_grads:
                bias_ref = outs[1]
                bvec = jnp.where(jnp.arange(2) == 0, end_state[1], end_state[2])
                if f_tiles == 1:
                    plt.store(bias_ref.at[h, pid_q, part, jnp.arange(2)], bvec)
                else:
                    jax.lax.cond(
                        f_id == 0,
                        lambda: plt.store(bias_ref.at[h, pid_q, part, jnp.arange(2)], bvec),
                        lambda: None,
                    )

        hub_q_blocks = pl.cdiv(hubs, rq_hub)
        if lse or scalar:
            out_shape = jax.ShapeDtypeStruct((heads, nchunks, hubs), jnp.float32)
            grid = (hub_q_blocks, nchunks, heads)
        else:
            out_shape = jax.ShapeDtypeStruct((heads, nchunks, hubs, feature), jnp.float32)
            grid = (hub_q_blocks, f_tiles, nchunks, heads)

        shapes = (
            (out_shape, jax.ShapeDtypeStruct((heads, hub_q_blocks, nchunks, 2), jnp.float32))
            if return_bias_grads
            else out_shape
        )
        res = pl.pallas_call(
            hub_kernel,
            out_shape=shapes,
            grid=grid,
            interpret=interpret,
            name="thot_hub_" + ("lse" if lse else ("smv" if scalar else "mv")),
            **opts,
        )(*args)
        if return_bias_grads:
            partial, bias_h_blocks = res
            bias_h = bias_h_blocks.sum(axis=(1, 2))
        else:
            partial, bias_h = res, None
        hub_reduced = jax.nn.logsumexp(partial, axis=1) if lse else partial.sum(axis=1)
        return hub_reduced, bias_h

    if hubs:
        hub_out, bias_h = run_hub_partial()
    else:
        hub_out = jnp.zeros((heads, 0) if (lse or scalar) else (heads, 0, feature), jnp.float32)
        bias_h = jnp.zeros((heads, 2), jnp.float32) if return_bias_grads else None

    def run_real():
        args = (
            q, k, own, other, bias, payload_arg,
            left_d if has_d else dummy3, right_d if has_d else dummy3,
            left_s if has_s else dummy3, right_s if has_s else dummy3,
            hub_out if hubs else (dummy2 if (lse or scalar) else dummy3),
        )

        def real_kernel(qr, kr, ur, vr, br, pr, ldr, rdr, lsr, rsr, hbr, *outs):
            out_ref = outs[0]
            pid0 = pl.program_id(0)
            if lse or scalar:
                f_id, h = 0, pl.program_id(1)
            else:
                f_id, h = pl.program_id(1), pl.program_id(2)

            qi = pid0 * rq_real + jnp.arange(rq_real)
            validq = qi < length
            qi_safe = jnp.minimum(qi, length - 1)
            sd, qq, uv, dest, ad_d, ll_d, ad_s, ll_s = setup_query(
                h, qi, validq, qr, ur, ldr, lsr, f_id
            )
            tile_step = make_tile_step(
                h, validq, qq, uv, dest, ll_d, ll_s, kr, vr, pr, rdr, rsr, sd, ad_d, ad_s
            )

            b00_val, b01_val = br[h, 0, 0], br[h, 0, 1]
            state = _init_tile_state(
                rq_real, fblock, lse=lse, scalar=scalar, return_bias_grads=return_bias_grads
            )

            js_band = jnp.arange(sk_band)
            start = pid0 * rq_real - radius_eff

            def band_body(t, s):
                off = t * sk_band + js_band
                ix = (start + off + length) % length
                delta = (ix[None, :] - qi_safe[:, None] + length) % length
                edges = (delta <= radius_eff) | (delta >= length - radius_eff)
                return tile_step(s, ix, off < span, edges, b00_val, is_spoke=False)

            state = jax.lax.fori_loop(0, pl.cdiv(span, sk_band), band_body, state)

            if hubs:
                js_spokes = jnp.arange(sk_spokes)

                def spoke_body(t, s):
                    off = t * sk_spokes + js_spokes
                    return tile_step(s, length + off, off < hubs, True, b01_val, is_spoke=True)

                state = jax.lax.fori_loop(0, pl.cdiv(hubs, sk_spokes), spoke_body, state)

            if lse:
                mx, total = state
                ans = mx + jnp.log(jnp.maximum(total, 1.0))
                plt.store(out_ref.at[h, qi], ans.astype(res_dtype), mask=validq)
            elif scalar:
                acc = state[0] if return_bias_grads else state
                plt.store(out_ref.at[h, qi], acc.astype(res_dtype), mask=validq)
            else:
                acc = state[0] if return_bias_grads else state
                plt.store(
                    out_ref.at[h, qi[:, None], dest[None, :]],
                    acc.astype(res_dtype),
                    mask=validq[:, None] & (dest[None, :] < feature),
                )

            if hubs:
                hub_copy_tile = min(64, max(16, pl.next_power_of_2(hubs)))

                def copy_hub():
                    def copy_step(t, _):
                        hi = t * hub_copy_tile + jnp.arange(hub_copy_tile)
                        hmask = hi < hubs
                        if lse or scalar:
                            hv = plt.load(hbr.at[h, hi], mask=hmask, other=0.0)
                            plt.store(out_ref.at[h, length + hi], hv.astype(res_dtype), mask=hmask)
                        else:
                            hv = plt.load(
                                hbr.at[h, hi[:, None], dest[None, :]],
                                mask=hmask[:, None] & (dest[None, :] < feature),
                                other=0.0,
                            )
                            plt.store(
                                out_ref.at[h, (length + hi)[:, None], dest[None, :]],
                                hv.astype(res_dtype),
                                mask=hmask[:, None] & (dest[None, :] < feature),
                            )
                        return None

                    nit_copy = pl.cdiv(hubs, hub_copy_tile)
                    if nit_copy == 1:
                        copy_step(0, None)
                    else:
                        jax.lax.fori_loop(0, nit_copy, copy_step, None)

                jax.lax.cond(pid0 == 0, copy_hub, lambda: None)

            if return_bias_grads:
                bias_ref = outs[1]
                bvec = jnp.where(jnp.arange(2) == 0, state[1], state[2])
                if f_tiles == 1:
                    plt.store(bias_ref.at[h, pid0, jnp.arange(2)], bvec)
                else:
                    jax.lax.cond(
                        f_id == 0,
                        lambda: plt.store(bias_ref.at[h, pid0, jnp.arange(2)], bvec),
                        lambda: None,
                    )

        if lse or scalar:
            out_shape = jax.ShapeDtypeStruct((heads, live), res_dtype)
            grid = (real_blocks, heads)
        else:
            out_shape = jax.ShapeDtypeStruct((heads, live, feature), res_dtype)
            grid = (real_blocks, f_tiles, heads)

        shapes = (out_shape, jax.ShapeDtypeStruct((heads, real_blocks, 2), jnp.float32)) if return_bias_grads else out_shape
        res = pl.pallas_call(
            real_kernel,
            out_shape=shapes,
            grid=grid,
            interpret=interpret,
            name="thot_real_" + ("lse" if lse else ("smv" if scalar else "mv")),
            **opts,
        )(*args)
        if return_bias_grads:
            out, bias_r_blocks = res
            bias_r = bias_r_blocks.sum(axis=1)
            return out, jnp.stack((bias_r, bias_h), axis=1)
        return res

    return run_real()


def batched_logsumexp(
    q: jnp.ndarray,
    k: jnp.ndarray,
    other: jnp.ndarray,
    bias: jnp.ndarray,
    *,
    length: int,
    hubs: int,
    radius: int,
    scale: float,
    dot_precision: str = "bf16_tc",
    transpose: bool = False,
    backend: str = "pallas",
) -> jnp.ndarray:
    """Batched ``[H, N, D]`` sparse row/column log-sum-exp returning ``[H, L+K]`` FP32."""
    _validate_batched(q, k, other, bias, length, hubs, radius, dot_precision)
    if backend not in ("pallas", "interpret"):
        raise ValueError(f"Unsupported backend: {backend!r}")
    if transpose:
        q, k, bias = k, q, jnp.swapaxes(bias, -2, -1)
    return _sparse_batched(
        q, k, other, other, None, bias, None, None, None, None,
        length=length, hubs=hubs, radius=radius, scale=scale,
        dot_precision=dot_precision, interpret=(backend == "interpret"),
        lse=True, scalar=False, return_bias_grads=False, out_dtype=jnp.float32,
    )


def batched_weighted_matvec(
    q: jnp.ndarray,
    k: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    payload: jnp.ndarray,
    bias: jnp.ndarray,
    *,
    length: int,
    hubs: int,
    radius: int,
    scale: float,
    dot_precision: str = "bf16_tc",
    transpose: bool = False,
    left_d: jnp.ndarray | None = None,
    right_d: jnp.ndarray | None = None,
    left_s: jnp.ndarray | None = None,
    right_s: jnp.ndarray | None = None,
    return_bias_grads: bool = False,
    out_dtype=None,
    backend: str = "pallas",
):
    """Batched ``[H, N, D]`` sparse weighted matvec with optional split factors."""
    heads, _ = _validate_batched(q, k, v, bias, length, hubs, radius, dot_precision)
    if backend not in ("pallas", "interpret"):
        raise ValueError(f"Unsupported backend: {backend!r}")
    if u.ndim != 2 or u.shape[0] != heads or u.shape[1] < length + hubs:
        raise ValueError(f"Expected u shape ({heads}, >={length + hubs}), got {u.shape}")
    scalar = payload.ndim == 2
    if scalar:
        if payload.shape[0] != heads or payload.shape[1] < length + hubs:
            raise ValueError(f"Bad scalar payload shape {payload.shape}")
    elif payload.ndim == 3:
        if payload.shape[0] != heads or payload.shape[1] < length + hubs or payload.shape[2] < 1:
            raise ValueError(f"Bad payload shape {payload.shape}")
    else:
        raise ValueError(f"Expected 2D or 3D payload, got {payload.ndim}D")
    if (left_d is None) != (right_d is None):
        raise ValueError("Supply both left_d and right_d")
    if (left_s is None) != (right_s is None):
        raise ValueError("Supply both left_s and right_s")
    if transpose:
        q, k, u, v = k, q, v, u
        bias = jnp.swapaxes(bias, -2, -1)
        left_d, right_d = right_d, left_d
        left_s, right_s = right_s, left_s
    return _sparse_batched(
        q, k, u, v, payload, bias, left_d, right_d, left_s, right_s,
        length=length, hubs=hubs, radius=radius, scale=scale,
        dot_precision=dot_precision, interpret=(backend == "interpret"),
        lse=False, scalar=scalar, return_bias_grads=return_bias_grads, out_dtype=out_dtype,
    )


def _sweep_1_step(q, k, aa, ab, log_a, log_b, b_mat, v_prev, denom_a, *, kw):
    lse_r = batched_logsumexp(q, k, v_prev, b_mat, transpose=False, **kw)
    u_raw = jnp.where(aa, log_a - lse_r, -jnp.inf)
    lse_c = batched_logsumexp(q, k, u_raw, b_mat, transpose=True, **kw)
    v_raw = jnp.where(ab, log_b - lse_c, -jnp.inf)
    shift = jnp.where(aa, u_raw, 0.0).sum(axis=-1, keepdims=True) / denom_a
    u_out = jnp.where(aa, u_raw - shift, -jnp.inf)
    v_out = jnp.where(ab, v_raw + shift, -jnp.inf)
    return u_raw, u_out, v_out, shift


def _make_biases(bias: jnp.ndarray, heads: int, log_deg_prior: float):
    b = bias.astype(jnp.float32)
    if b.ndim == 2:
        b = jnp.broadcast_to(b[None], (heads, 2, 2))
    b_rr = b[:, 0, 0] + log_deg_prior
    b_base = jnp.stack(
        (
            jnp.stack((b_rr, b[:, 0, 1]), axis=-1),
            jnp.stack((b[:, 1, 0], b[:, 1, 1]), axis=-1),
        ),
        axis=-2,
    )
    b_eff = jnp.stack(
        (
            jnp.stack((b_rr, b[:, 0, 1]), axis=-1),
            jnp.stack((b[:, 1, 0], jnp.full_like(b_rr, -20.0)), axis=-1),
        ),
        axis=-2,
    )
    return b_base, b_eff


@lru_cache(maxsize=64)
def _bind_thot_transport(
    length: int,
    radius: int,
    hubs: int,
    scale: float,
    dot_precision: str,
    backend: str,
):
    """Compiled ``custom_vjp`` for the 1+2 annealed row-closed THOT operator."""
    live = length + hubs
    d_deg = min(length, 2 * radius + 1)
    log_deg_prior = math.log(length / d_deg)
    kw = dict(
        length=length,
        hubs=hubs,
        radius=radius,
        scale=scale,
        dot_precision=dot_precision,
        backend=backend,
    )

    def forward(q, k, value, a, b, bias):
        heads = q.shape[0]
        a_act = a[:, :live].astype(jnp.float32)
        b_act = b[:, :live].astype(jnp.float32)
        aa = a_act > 0
        ab = b_act > 0
        denom_a = jnp.maximum(aa.sum(axis=-1, keepdims=True), 1)
        log_a = jnp.log(jnp.where(aa, a_act, 1.0))
        log_b = jnp.log(jnp.where(ab, b_act, 1.0))
        b_base, b_eff = _make_biases(bias, heads, log_deg_prior)

        v0 = jnp.where(ab, log_b, -jnp.inf)
        _, _, base_v, _ = _sweep_1_step(q, k, aa, ab, log_a, log_b, b_base, v0, denom_a, kw=kw)
        base_v = jax.lax.stop_gradient(base_v)

        raw0, _, v0_tail, shift0 = _sweep_1_step(
            q, k, aa, ab, log_a, log_b, b_eff, base_v, denom_a, kw=kw
        )
        raw1, _, v1_tail, shift1 = _sweep_1_step(
            q, k, aa, ab, log_a, log_b, b_eff, v0_tail, denom_a, kw=kw
        )
        lse_fin = batched_logsumexp(q, k, v1_tail, b_eff, transpose=False, **kw)
        u_fin = jnp.where(aa, log_a - lse_fin, -jnp.inf)
        v_fin = v1_tail

        y = batched_weighted_matvec(
            q, k, u_fin, v_fin, value, b_eff, transpose=False, out_dtype=jnp.float32, **kw
        )
        saved = (
            q, k, value, a, b, bias,
            y, base_v, raw0, shift0, v0_tail, raw1, shift1, v_fin, u_fin,
        )
        return y, saved

    @jax.custom_vjp
    def operator_fn(q, k, value, a, b, bias):
        return forward(q, k, value, a, b, bias)[0]

    def backward(saved, g):
        (
            q, k, value, a, b, bias,
            y, base_v, raw0, shift0, v0_tail, raw1, shift1, v_fin, u_fin,
        ) = saved
        heads = q.shape[0]
        g_act = g[:, :live].astype(jnp.float32)
        a_act = a[:, :live].astype(jnp.float32)
        b_act = b[:, :live].astype(jnp.float32)
        aa = a_act > 0
        ab = b_act > 0
        denom_a = jnp.maximum(aa.sum(axis=-1, keepdims=True), 1)
        inv_a = jnp.where(aa, 1.0 / jnp.where(aa, a_act, 1.0), 0.0)
        inv_b = jnp.where(ab, 1.0 / jnp.where(ab, b_act, 1.0), 0.0)
        _, b_eff = _make_biases(bias, heads, log_deg_prior)

        dvalue_f32 = batched_weighted_matvec(
            q, k, u_fin, v_fin, g_act, b_eff, transpose=True, out_dtype=jnp.float32, **kw
        )

        gu = (g_act * y).sum(axis=-1)
        gv = (value[:, :live].astype(jnp.float32) * dvalue_f32).sum(axis=-1)

        u_base = jnp.where(aa, u_fin, 0.0)
        v_base = jnp.where(ab, v_fin, 0.0)

        row_fin = jnp.where(aa, gu, 0.0) * inv_a
        da = row_fin
        db = jnp.zeros_like(b_act)
        gv = gv - batched_weighted_matvec(
            q, k, u_fin, v_fin, row_fin, b_eff, transpose=True, out_dtype=jnp.float32, **kw
        )

        u0 = jnp.where(aa, raw0 - shift0, -jnp.inf)
        u1 = jnp.where(aa, raw1 - shift1, -jnp.inf)

        L_s_cols = [-row_fin]
        R_s_cols = [jnp.where(ab, 1.0, 0.0)]

        history = (
            (raw1, v0_tail, u1, v_fin, True),
            (raw0, base_v, u0, v0_tail, False),
        )
        for raw_t, prev_t, ut, vt, need_prev_gv in history:
            gv = jnp.where(ab, gv, 0.0)
            c_ut = jnp.where(aa, ut - u_base, 0.0).sum(axis=-1, keepdims=True) / denom_a
            c_raw = jnp.where(aa, raw_t - u_base, 0.0).sum(axis=-1, keepdims=True) / denom_a
            dc = gv.sum(axis=-1, keepdims=True) / denom_a
            col = gv * inv_b
            ev_vt_col = jnp.where(
                ab,
                jnp.exp(jnp.clip(jnp.where(ab, (vt - v_base) + c_ut, 0.0), -50.0, 50.0)) * col,
                0.0,
            )
            p_col = batched_weighted_matvec(
                q, k, ut, vt, col, b_eff, transpose=False, out_dtype=jnp.float32, **kw
            )
            row = (jnp.where(aa, dc, 0.0) - p_col) * inv_a
            da = da + row
            db = db + col

            eu_ut = jnp.where(
                aa,
                jnp.exp(jnp.clip(jnp.where(aa, (ut - u_base) - c_ut, 0.0), -50.0, 50.0)),
                0.0,
            )
            eu_raw = jnp.where(
                aa,
                jnp.exp(jnp.clip(jnp.where(aa, (raw_t - u_base) - c_raw, 0.0), -50.0, 50.0)),
                0.0,
            )
            ev_prev = jnp.where(
                ab,
                jnp.exp(jnp.clip(jnp.where(ab, (prev_t - v_base) + c_raw, 0.0), -50.0, 50.0)),
                0.0,
            )
            L_s_cols.extend((-eu_ut, -eu_raw * row))
            R_s_cols.extend((ev_vt_col, ev_prev))

            if need_prev_gv:
                gv = -batched_weighted_matvec(
                    q, k, raw_t, prev_t, row, b_eff, transpose=True, out_dtype=jnp.float32, **kw
                )

        left_s = jnp.stack(L_s_cols, axis=-1)
        right_s = jnp.stack(R_s_cols, axis=-1)

        dq, dbias_h = batched_weighted_matvec(
            q, k, u_fin, v_fin, k, b_eff,
            transpose=False,
            left_d=g_act, right_d=value,
            left_s=left_s, right_s=right_s,
            return_bias_grads=True,
            out_dtype=jnp.float32,
            **kw,
        )
        dk = batched_weighted_matvec(
            q, k, u_fin, v_fin, q, b_eff,
            transpose=True,
            left_d=g_act, right_d=value,
            left_s=left_s, right_s=right_s,
            return_bias_grads=False,
            out_dtype=jnp.float32,
            **kw,
        )

        dq = (dq * scale).astype(q.dtype)
        dk = (dk * scale).astype(k.dtype)
        dvalue = dvalue_f32.astype(value.dtype)
        da = da.astype(a.dtype)
        db = db.astype(b.dtype)
        dbias_h = dbias_h.at[:, 1, 1].set(0.0)
        dbias = (dbias_h.sum(axis=0) if bias.ndim == 2 else dbias_h).astype(bias.dtype)
        return dq, dk, dvalue, da, db, dbias

    operator_fn.defvjp(forward, backward)
    return operator_fn


def _prepare_batched_inputs(q, k, v, a, b, bias, *, length: int, hubs: int):
    if q.ndim not in (2, 3, 4):
        raise ValueError(f"Expected 2D, 3D, or 4D q/k/v, got shape {q.shape}")
    if q.shape != k.shape or v.ndim != q.ndim or v.shape[:-1] != q.shape[:-1]:
        raise ValueError(f"Mismatched q/k/v shapes: {q.shape}, {k.shape}, {v.shape}")
    if a.shape != q.shape[:-1] or b.shape != q.shape[:-1]:
        raise ValueError(f"Expected a, b shape {q.shape[:-1]}, got {a.shape}, {b.shape}")
    if length < 1 or hubs < 0:
        raise ValueError(f"Expected length >= 1 and hubs >= 0, got {length}, {hubs}")

    live = length + hubs
    n_phys = q.shape[-2]
    if n_phys < live:
        raise ValueError(f"Sequence axis {n_phys} < length + hubs ({live})")

    bias_arr = jnp.zeros((2, 2), dtype=jnp.float32) if bias is None else bias
    lead_shape = q.shape[:-2]

    if q.ndim == 2:
        if bias_arr.shape != (2, 2):
            raise ValueError(f"Expected bias shape (2, 2) for 2D inputs, got {bias_arr.shape}")
        q_b, k_b, v_b = q[None], k[None], v[None]
        a_b, b_b = a[None], b[None]
        bias_b = bias_arr
    elif q.ndim == 3:
        heads = lead_shape[0]
        if bias_arr.shape not in ((2, 2), (heads, 2, 2)):
            raise ValueError(f"Incompatible bias shape {bias_arr.shape} for {heads} heads")
        q_b, k_b, v_b, a_b, b_b, bias_b = q, k, v, a, b, bias_arr
    else:
        bsz, heads = lead_shape
        q_b = q.reshape(bsz * heads, n_phys, q.shape[-1])
        k_b = k.reshape(bsz * heads, n_phys, k.shape[-1])
        v_b = v.reshape(bsz * heads, n_phys, v.shape[-1])
        a_b = a.reshape(bsz * heads, n_phys)
        b_b = b.reshape(bsz * heads, n_phys)
        if bias_arr.shape == (2, 2):
            bias_b = bias_arr
        elif bias_arr.shape == (heads, 2, 2):
            bias_b = jnp.broadcast_to(bias_arr[None], (bsz, heads, 2, 2)).reshape(bsz * heads, 2, 2)
        elif bias_arr.shape == (bsz, heads, 2, 2):
            bias_b = bias_arr.reshape(bsz * heads, 2, 2)
        else:
            raise ValueError(f"Incompatible bias shape {bias_arr.shape} for shape {lead_shape}")

    return q_b, k_b, v_b, a_b, b_b, bias_b, lead_shape, n_phys, live


def _restore_shape(out_b: jnp.ndarray, lead_shape: tuple[int, ...], n_phys: int, live: int) -> jnp.ndarray:
    if n_phys > live:
        out_b = jnp.pad(out_b, ((0, 0), (0, n_phys - live), (0, 0)))
    if len(lead_shape) == 0:
        return out_b[0]
    if len(lead_shape) == 2:
        return out_b.reshape(lead_shape[0], lead_shape[1], n_phys, out_b.shape[-1])
    return out_b


def thot_transport(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    bias: jnp.ndarray | None = None,
    *,
    length: int,
    radius: int = 4,
    hubs: int = 16,
    scale: float | None = None,
    dot_precision: str = "bf16_tc",
    backend: str = "pallas",
) -> jnp.ndarray:
    """Compute unnormalized ``P @ v`` for 1+2 annealed row-closed THOT.

    Accepts ``[N, D]``, ``[H, N, D]``, or ``[B, H, N, D]`` inputs with
    ``N >= length + hubs``. Supports GPU Pallas/Triton (``backend="pallas"``)
    and CPU/XLA Pallas interpretation (``backend="interpret"``).
    """
    if dot_precision not in ("fp32", "bf16_tc"):
        raise ValueError(f"Unsupported dot_precision: {dot_precision!r}")
    if backend not in ("pallas", "interpret"):
        raise ValueError(f"Unsupported backend: {backend!r}")

    q_b, k_b, v_b, a_b, b_b, bias_b, lead_shape, n_phys, live = _prepare_batched_inputs(
        q, k, v, a, b, bias, length=length, hubs=hubs
    )
    scale_val = 1.0 / math.sqrt(q.shape[-1]) if scale is None else float(scale)
    fn = _bind_thot_transport(length, radius, hubs, scale_val, dot_precision, backend)
    out_act = fn(q_b[:, :live], k_b[:, :live], v_b[:, :live], a_b[:, :live], b_b[:, :live], bias_b)
    return _restore_shape(out_act, lead_shape, n_phys, live)


def _thot_reference_single(
    q_h: jnp.ndarray,
    k_h: jnp.ndarray,
    v_h: jnp.ndarray,
    a_h: jnp.ndarray,
    b_h: jnp.ndarray,
    b_base_h: jnp.ndarray,
    b_eff_h: jnp.ndarray,
    *,
    length: int,
    radius: int,
    hubs: int,
    scale: float,
) -> jnp.ndarray:
    live = length + hubs
    radius_eff = min(radius, length // 2)
    idx = jnp.arange(live)
    is_real = idx < length

    dist = jnp.abs(idx[:length, None] - idx[None, :length])
    ring_mask = jnp.minimum(dist, length - dist) <= radius_eff
    edge_mask = jnp.ones((live, live), dtype=bool).at[:length, :length].set(ring_mask)

    a_act = a_h[:live].astype(jnp.float32)
    b_act = b_h[:live].astype(jnp.float32)
    aa = a_act > 0
    ab = b_act > 0
    denom_a = jnp.maximum(aa.sum(), 1)
    log_a = jnp.log(jnp.where(aa, a_act, 1.0))
    log_b = jnp.log(jnp.where(ab, b_act, 1.0))

    ti = jnp.where(is_real, 0, 1)[:, None]
    tj = jnp.where(is_real, 0, 1)[None, :]
    base_scores = jnp.matmul(
        q_h[:live].astype(jnp.float32),
        k_h[:live].astype(jnp.float32).T,
        precision=jax.lax.Precision.HIGHEST,
    ) * scale
    s_base = base_scores + b_base_h[ti, tj]
    s_eff = base_scores + b_eff_h[ti, tj]

    def masked_lse(mat: jnp.ndarray, mask: jnp.ndarray, axis: int) -> jnp.ndarray:
        m = jnp.max(jnp.where(mask, mat, -jnp.inf), axis=axis, keepdims=True)
        has = jnp.isfinite(m)
        safe_m = jax.lax.stop_gradient(jnp.where(has, m, 0.0))
        ex = jnp.where(mask & has, jnp.exp(jnp.where(mask & has, mat - safe_m, 0.0)), 0.0)
        total = jnp.sum(ex, axis=axis, keepdims=True)
        out = jnp.where(has, safe_m + jnp.log(jnp.maximum(total, 1e-30)), -jnp.inf)
        return jnp.squeeze(out, axis=axis)

    def lse_r(s_mat, v_prev):
        row_mask = edge_mask & ab[None, :]
        return masked_lse(s_mat + v_prev[None, :], row_mask, axis=1)

    def lse_c(s_mat, u_raw):
        col_mask = edge_mask & aa[:, None]
        return masked_lse(s_mat + u_raw[:, None], col_mask, axis=0)

    def sweep(s_mat, v_prev):
        u_raw = jnp.where(aa, log_a - lse_r(s_mat, v_prev), -jnp.inf)
        v_raw = jnp.where(ab, log_b - lse_c(s_mat, u_raw), -jnp.inf)
        shift = jnp.where(aa, u_raw, 0.0).sum() / denom_a
        return jnp.where(aa, u_raw - shift, -jnp.inf), jnp.where(ab, v_raw + shift, -jnp.inf)

    v0 = jnp.where(ab, log_b, -jnp.inf)
    _, base_v = sweep(s_base, v0)
    base_v = jax.lax.stop_gradient(base_v)

    _, v0_tail = sweep(s_eff, base_v)
    _, v1_tail = sweep(s_eff, v0_tail)
    u_fin = jnp.where(aa, log_a - lse_r(s_eff, v1_tail), -jnp.inf)

    valid_edges = edge_mask & aa[:, None] & ab[None, :]
    plan = jnp.exp(jnp.where(valid_edges, s_eff + u_fin[:, None] + v1_tail[None, :], -jnp.inf))
    return jnp.matmul(plan, v_h[:live].astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)


def thot_reference(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    bias: jnp.ndarray | None = None,
    *,
    length: int,
    radius: int = 4,
    hubs: int = 16,
    scale: float | None = None,
) -> jnp.ndarray:
    """Pure-JAX dense reference for unnormalized ``P @ v`` matching ``thot_transport``.

    Intended for small sequence lengths, CPU tests, and exact-autodiff gradient checks.
    """
    q_b, k_b, v_b, a_b, b_b, bias_b, lead_shape, n_phys, live = _prepare_batched_inputs(
        q, k, v, a, b, bias, length=length, hubs=hubs
    )
    heads = q_b.shape[0]
    scale_val = 1.0 / math.sqrt(q.shape[-1]) if scale is None else float(scale)
    d_deg = min(length, 2 * radius + 1)
    log_deg_prior = math.log(length / d_deg)
    b_base, b_eff = _make_biases(bias_b, heads, log_deg_prior)

    out_act = jax.vmap(
        lambda qh, kh, vh, ah, bh, bbh, beh: _thot_reference_single(
            qh, kh, vh, ah, bh, bbh, beh,
            length=length, radius=radius, hubs=hubs, scale=scale_val,
        )
    )(q_b[:, :live], k_b[:, :live], v_b[:, :live], a_b[:, :live], b_b[:, :live], b_base, b_eff)

    return _restore_shape(out_act, lead_shape, n_phys, live)


def thot_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    bias: jnp.ndarray | None = None,
    *,
    length: int,
    radius: int = 4,
    hubs: int = 16,
    scale: float | None = None,
    dot_precision: str = "bf16_tc",
    backend: str = "pallas",
    normalize_by_a: bool = True,
    return_hubs: bool = False,
) -> jnp.ndarray:
    """Topology Hub Optimal Transport (THOT) attention layer.

    Args:
        q, k, v: Tensors of shape ``[N, D]``, ``[H, N, D]``, or ``[B, H, N, D]``
            with ``N >= length + hubs`` (tokens followed by hub slots).
        a, b: Positive capacity marginals of shape ``[..., N]``.
        bias: Optional node-type bias of shape ``(2, 2)``, ``(H, 2, 2)``, or
            ``(B, H, 2, 2)`` (defaults to zeros).
        length: Number of real sequence tokens ``L``.
        radius: Local sliding-window radius ``r``.
        hubs: Number of shared global hub slots ``K``.
        scale: Score scaling factor (defaults to ``1 / sqrt(D)``).
        dot_precision: Kernel dot precision (``"bf16_tc"`` or ``"fp32"``).
        backend: Execution backend (``"pallas"``, ``"interpret"``, or ``"reference"``).
        normalize_by_a: If ``True``, return row-normalized outputs ``(P @ v) / a``
            so every row forms a convex combination of values.
        return_hubs: If ``False``, slice off hub outputs and return only the
            ``length`` real token outputs.

    Returns:
        Updated representations in ``q.dtype``.
    """
    if backend == "reference":
        raw = thot_reference(
            q, k, v, a, b, bias, length=length, radius=radius, hubs=hubs, scale=scale
        )
    elif backend in ("pallas", "interpret"):
        raw = thot_transport(
            q, k, v, a, b, bias,
            length=length, radius=radius, hubs=hubs, scale=scale,
            dot_precision=dot_precision, backend=backend,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend!r}")

    if normalize_by_a:
        a_f = a.astype(jnp.float32)
        active = a_f > 0
        denom = jnp.where(active, a_f, 1.0)[..., None]
        out = jnp.where(active[..., None], raw / denom, 0.0)
    else:
        out = raw

    if not return_hubs:
        out = out[..., :length, :]
    return out.astype(q.dtype)

