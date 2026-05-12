#!/usr/bin/env python3
# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import ot_kernel


def _make_inputs(seed, length, dim):
    key = jax.random.PRNGKey(seed)
    q_key, k_key = jax.random.split(key, 2)
    shape = (1, 1, length, dim)
    q = jax.random.normal(q_key, shape, dtype=jnp.float32)
    k = jax.random.normal(k_key, shape, dtype=jnp.float32)
    q_valid = int(length * 3 // 4)
    k_valid = int(length * 7 // 8)
    q_mask = (jnp.arange(length) < q_valid)[None, None, :]
    k_mask = (jnp.arange(length) < k_valid)[None, None, :]
    return q, k, q_mask, k_mask


def _plan(q, k, u, v, q_mask, k_mask, band_width, epsilon):
    length = q.shape[-2]
    score = jnp.einsum("bhmd,bhnd->bhmn", q, k) / (jnp.sqrt(q.shape[-1]) * epsilon)
    valid = q_mask[..., :, None] & k_mask[..., None, :]
    if band_width is not None and int(band_width) < length:
        rows = jnp.arange(length)[:, None]
        cols = jnp.arange(length)[None, :]
        valid = valid & (jnp.abs(rows - cols)[None, None, :, :] <= int(band_width))
    return jnp.where(valid, jnp.exp(score + u[..., :, None] + v[..., None, :]), 0.0)


def _log_plan(q, k, u, v, q_mask, k_mask, band_width, epsilon):
    length = q.shape[-2]
    score = jnp.einsum("bhmd,bhnd->bhmn", q, k) / (jnp.sqrt(q.shape[-1]) * epsilon)
    valid = q_mask[..., :, None] & k_mask[..., None, :]
    if band_width is not None and int(band_width) < length:
        rows = jnp.arange(length)[:, None]
        cols = jnp.arange(length)[None, :]
        valid = valid & (jnp.abs(rows - cols)[None, None, :, :] <= int(band_width))
    log_p = score + u[..., :, None] + v[..., None, :]
    return jnp.where(valid, log_p, 0.0), valid


def _history(q, k, q_mask, k_mask, *, total_steps, band_width, epsilon):
    scale = ot_kernel._score_scale(q.shape[-1], epsilon)
    block = 128
    u = jnp.zeros(q.shape[:-1], dtype=q.dtype)
    v = jnp.zeros(k.shape[:-1], dtype=k.dtype)
    u_hist = [u]
    v_hist = [v]
    for _ in range(int(total_steps)):
        u, v = ot_kernel._banded_compute_next_uv(
            q,
            k,
            u,
            v,
            scale,
            block,
            q_mask,
            k_mask,
            band_width,
            support_reconstruction="none",
        )
        u_hist.append(u)
        v_hist.append(v)
    return u_hist, v_hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--base_steps", type=int, default=15)
    parser.add_argument("--tail_steps", type=int, default=2)
    parser.add_argument("--band_width", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=1.0)
    args = parser.parse_args()

    q, k, q_mask, k_mask = _make_inputs(args.seed, args.length, args.dim)
    total_steps = args.base_steps + args.tail_steps
    u_hist, v_hist = _history(
        q,
        k,
        q_mask,
        k_mask,
        total_steps=total_steps,
        band_width=args.band_width,
        epsilon=args.epsilon,
    )
    ref_idx = total_steps
    p_ref = _plan(q, k, u_hist[ref_idx], v_hist[ref_idx], q_mask, k_mask, args.band_width, args.epsilon)
    log_ref, valid = _log_plan(
        q,
        k,
        u_hist[ref_idx],
        v_hist[ref_idx],
        q_mask,
        k_mask,
        args.band_width,
        args.epsilon,
    )

    pairs = []
    for step in range(1, total_steps + 1):
        pairs.append((step, step))
        pairs.append((step, step - 1))

    reports = []
    for a, b in pairs:
        direct = _plan(q, k, u_hist[a], v_hist[b], q_mask, k_mask, args.band_width, args.epsilon)
        direct_log, _ = _log_plan(q, k, u_hist[a], v_hist[b], q_mask, k_mask, args.band_width, args.epsilon)
        row = jnp.exp(u_hist[a] - u_hist[ref_idx])
        col = jnp.exp(v_hist[b] - v_hist[ref_idx])
        reconstructed = p_ref * row[..., :, None] * col[..., None, :]
        reconstructed_log = log_ref + (u_hist[a] - u_hist[ref_idx])[..., :, None] + (
            v_hist[b] - v_hist[ref_idx]
        )[..., None, :]
        log_error = jnp.where(valid, jnp.abs(direct_log - reconstructed_log), 0.0)
        reports.append(
            {
                "a": int(a),
                "b": int(b),
                "plan_max_abs": float(jnp.max(jnp.abs(direct - reconstructed))),
                "log_max_abs": float(jnp.max(log_error)),
            }
        )

    jax.block_until_ready(p_ref)
    print(
        json.dumps(
            {
                "settings": {
                    "seed": args.seed,
                    "length": args.length,
                    "dim": args.dim,
                    "base_steps": args.base_steps,
                    "tail_steps": args.tail_steps,
                    "band_width": args.band_width,
                    "epsilon": args.epsilon,
                    "reference_pair": [ref_idx, ref_idx],
                },
                "plan_max_abs": max(item["plan_max_abs"] for item in reports),
                "log_max_abs": max(item["log_max_abs"] for item in reports),
                "pairs_checked": reports,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
