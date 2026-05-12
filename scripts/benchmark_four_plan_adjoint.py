#!/usr/bin/env python3

import argparse
import json
import time

import jax
import jax.numpy as jnp


def _band_mask(length, band_width):
    if band_width is None or band_width >= length:
        return jnp.ones((length, length), dtype=bool)
    idx = jnp.arange(length)
    return jnp.abs(idx[:, None] - idx[None, :]) <= int(band_width)


def _sinkhorn_history(score, support, total_steps):
    neg = jnp.asarray(-1.0e30, dtype=score.dtype)
    masked_score = jnp.where(support, score, neg)
    length = score.shape[0]
    u = jnp.zeros((length,), dtype=score.dtype)
    v = jnp.zeros((length,), dtype=score.dtype)
    u_hist = [u]
    v_hist = [v]
    for _ in range(int(total_steps)):
        u = -jax.scipy.special.logsumexp(masked_score + v[None, :], axis=-1)
        v = -jax.scipy.special.logsumexp(masked_score + u[:, None], axis=-2)
        shift = jnp.mean(u)
        u = u - shift
        v = v + shift
        u_hist.append(u)
        v_hist.append(v)
    return tuple(u_hist), tuple(v_hist)


def _plan(score, support, u, v):
    return jnp.where(support, jnp.exp(score + u[:, None] + v[None, :]), 0.0)


def _direct_four_plan(score, support, v_values, d_out, u0, u1, u2, v0, v1, v2):
    del u0
    p22 = _plan(score, support, u2, v2)
    p21 = _plan(score, support, u2, v1)
    p11 = _plan(score, support, u1, v1)
    p10 = _plan(score, support, u1, v0)
    z = d_out @ v_values.T
    gv = jnp.sum(p22 * z, axis=0)
    gu = jnp.sum(p22 * z, axis=1)
    dv2 = gv
    du2 = gu - p22 @ dv2
    dv1 = -(p21.T @ du2)
    du1 = -(p11 @ dv1)
    score_bar = p22 * z - p22 * dv2[None, :] - p21 * du2[:, None] - p11 * dv1[None, :] - p10 * du1[:, None]
    return score_bar


def _one_reference_tile(score, support, v_values, d_out, u0, u1, u2, v0, v1, v2):
    p22 = _plan(score, support, u2, v2)
    beta = jnp.exp(v1 - v2)
    alpha = jnp.exp(u1 - u2)
    delta = jnp.exp(v0 - v2)
    z = d_out @ v_values.T
    gv = jnp.sum(p22 * z, axis=0)
    gu = jnp.sum(p22 * z, axis=1)
    dv2 = gv
    du2 = gu - p22 @ dv2
    dv1 = -(p22 * beta[None, :]).T @ du2
    du1 = -((p22 * alpha[:, None] * beta[None, :]) @ dv1)
    modifier = z - dv2[None, :] - beta[None, :] * du2[:, None]
    modifier = modifier - alpha[:, None] * beta[None, :] * dv1[None, :]
    modifier = modifier - alpha[:, None] * delta[None, :] * du1[:, None]
    return p22 * modifier


def _time_fn(fn, args, repeats):
    compiled = jax.jit(fn)
    compiled(*args).block_until_ready()
    times = []
    for _ in range(int(repeats)):
        start = time.perf_counter()
        out = compiled(*args).block_until_ready()
        times.append(time.perf_counter() - start)
    return out, min(times), sum(times) / len(times)


def _case(seed, length, dim, band_width, base_steps, repeats):
    key = jax.random.PRNGKey(seed)
    q_key, k_key, v_key, do_key = jax.random.split(key, 4)
    q = jax.random.normal(q_key, (length, dim), dtype=jnp.float32)
    k = jax.random.normal(k_key, (length, dim), dtype=jnp.float32)
    v_values = jax.random.normal(v_key, (length, dim), dtype=jnp.float32)
    d_out = jax.random.normal(do_key, (length, dim), dtype=jnp.float32)
    score = (q @ k.T) / jnp.sqrt(jnp.asarray(dim, dtype=jnp.float32))
    support = _band_mask(length, band_width)
    u_hist, v_hist = _sinkhorn_history(score, support, base_steps + 2)
    history_args = (
        u_hist[base_steps],
        u_hist[base_steps + 1],
        u_hist[base_steps + 2],
        v_hist[base_steps],
        v_hist[base_steps + 1],
        v_hist[base_steps + 2],
    )
    jax.block_until_ready(history_args)
    args = (score, support, v_values, d_out, *history_args)
    direct, direct_min, direct_mean = _time_fn(
        _direct_four_plan,
        args,
        repeats,
    )
    orbit, orbit_min, orbit_mean = _time_fn(
        _one_reference_tile,
        args,
        repeats,
    )
    active_entries = int(jnp.sum(support))
    plan_bytes = active_entries * 4
    return {
        "length": int(length),
        "dim": int(dim),
        "band_width": None if band_width is None else int(band_width),
        "active_entries": active_entries,
        "direct_min_ms": direct_min * 1000.0,
        "direct_mean_ms": direct_mean * 1000.0,
        "orbit_min_ms": orbit_min * 1000.0,
        "orbit_mean_ms": orbit_mean * 1000.0,
        "speedup_min": direct_min / orbit_min,
        "speedup_mean": direct_mean / orbit_mean,
        "score_vjp_max_abs_diff": float(jnp.max(jnp.abs(direct - orbit))),
        "direct_plan_factor_bytes": int(4 * plan_bytes),
        "orbit_plan_factor_bytes": int(plan_bytes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="512,1024,2048")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--band_width", type=int, default=256)
    parser.add_argument("--base_steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    lengths = [int(item) for item in args.lengths.split(",") if item.strip()]
    results = [
        _case(args.seed, length, args.dim, args.band_width, args.base_steps, args.repeats)
        for length in lengths
    ]
    print(json.dumps({"settings": vars(args), "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
