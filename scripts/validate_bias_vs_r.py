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
    q_key, k_key, v_key, t_key = jax.random.split(key, 4)
    shape = (1, 1, length, dim)
    q = jax.random.normal(q_key, shape, dtype=jnp.float32)
    k = jax.random.normal(k_key, shape, dtype=jnp.float32)
    v = jax.random.normal(v_key, shape, dtype=jnp.float32)
    target = jax.random.normal(t_key, shape, dtype=jnp.float32)
    q_valid = int(length * 3 // 4)
    k_valid = int(length * 7 // 8)
    q_mask = (jnp.arange(length) < q_valid)[None, None, :]
    k_mask = (jnp.arange(length) < k_valid)[None, None, :]
    return q, k, v, target, q_mask, k_mask


def _loss(
    q,
    k,
    v,
    target,
    q_mask,
    k_mask,
    *,
    n_iters,
    band_width,
    refine_steps,
    gradient_mode,
    epsilon,
):
    out = ot_kernel.sinkhorn_attention(
        q,
        k,
        v,
        n_iters=n_iters,
        band_width=band_width,
        q_mask=q_mask,
        k_mask=k_mask,
        refine_steps=refine_steps,
        refine_tolerance=None,
        epsilon=epsilon,
        epsilon_scaling_steps=0,
        epsilon_scaling_factor=2.0,
        gradient_mode=gradient_mode,
        support_reconstruction="none",
    )
    return jnp.sum(out * target) / out.size


def _full_bptt_output(
    q,
    k,
    v,
    q_mask,
    k_mask,
    *,
    n_iters,
    band_width,
    refine_steps,
    epsilon,
):
    score_scale = ot_kernel._score_scale(q.shape[-1], epsilon)
    block = 128
    q_mask = ot_kernel._canonicalize_mask(q_mask, q.shape[0], q.shape[1], q.shape[2], "q_mask")
    k_mask = ot_kernel._canonicalize_mask(k_mask, k.shape[0], k.shape[1], k.shape[2], "k_mask")
    u0, v0 = ot_kernel.compute_global_uv(
        q,
        k,
        n_iters=n_iters,
        band_width=band_width,
        q_mask=q_mask,
        k_mask=k_mask,
        epsilon=epsilon,
        epsilon_scaling_steps=0,
        epsilon_scaling_factor=2.0,
        support_reconstruction="none",
    )
    u_curr, v_curr = u0, v0
    for _ in range(int(refine_steps)):
        u_curr, v_curr = ot_kernel._banded_compute_next_uv(
            q,
            k,
            u_curr,
            v_curr,
            score_scale,
            block,
            q_mask,
            k_mask,
            band_width,
            support_reconstruction="none",
        )
    return ot_kernel._banded_transport_apply(
        q,
        k,
        v,
        u_curr,
        v_curr,
        score_scale,
        block,
        q_mask,
        k_mask,
        band_width,
        support_reconstruction="none",
    )


def _case(q, k, v, target, q_mask, k_mask, *, n_iters, band_width, refine_steps, epsilon):
    def tail_loss(q_, k_, v_):
        return _loss(
            q_,
            k_,
            v_,
            target,
            q_mask,
            k_mask,
            n_iters=n_iters,
            band_width=band_width,
            refine_steps=refine_steps,
            gradient_mode="tail_refinement",
            epsilon=epsilon,
        )

    def full_loss(q_, k_, v_):
        out = _full_bptt_output(
            q_,
            k_,
            v_,
            q_mask,
            k_mask,
            n_iters=n_iters,
            band_width=band_width,
            refine_steps=refine_steps,
            epsilon=epsilon,
        )
        return jnp.sum(out * target) / out.size

    tail_out = ot_kernel.sinkhorn_attention(
        q,
        k,
        v,
        n_iters=n_iters,
        band_width=band_width,
        q_mask=q_mask,
        k_mask=k_mask,
        refine_steps=refine_steps,
        refine_tolerance=None,
        epsilon=epsilon,
        epsilon_scaling_steps=0,
        epsilon_scaling_factor=2.0,
        gradient_mode="tail_refinement",
        support_reconstruction="none",
    )
    full_out = _full_bptt_output(
        q,
        k,
        v,
        q_mask,
        k_mask,
        n_iters=n_iters,
        band_width=band_width,
        refine_steps=refine_steps,
        epsilon=epsilon,
    )
    tail_val, tail_grads = jax.value_and_grad(tail_loss, argnums=(0, 1, 2))(q, k, v)
    full_val, full_grads = jax.value_and_grad(full_loss, argnums=(0, 1, 2))(q, k, v)
    jax.block_until_ready((tail_out, full_out, tail_val, full_val, tail_grads, full_grads))

    dQ = float(jnp.max(jnp.abs(tail_grads[0] - full_grads[0])))
    dK = float(jnp.max(jnp.abs(tail_grads[1] - full_grads[1])))
    dV = float(jnp.max(jnp.abs(tail_grads[2] - full_grads[2])))
    return {
        "refine_steps": int(refine_steps),
        "tail_loss": float(tail_val),
        "full_loss": float(full_val),
        "output_max_abs": float(jnp.max(jnp.abs(tail_out - full_out))),
        "dQ_max_abs": dQ,
        "dK_max_abs": dK,
        "dV_max_abs": dV,
        "grad_max_abs": max(dQ, dK, dV),
    }


def _parse_refine_steps(raw):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("at least one refine_steps value is required")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--n_iters", type=int, default=15)
    parser.add_argument("--band_width", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--refine_steps", type=str, default="0,1,2,4")
    args = parser.parse_args()

    refine_steps_list = _parse_refine_steps(args.refine_steps)
    q, k, v, target, q_mask, k_mask = _make_inputs(args.seed, args.length, args.dim)
    cases = [
        _case(
            q,
            k,
            v,
            target,
            q_mask,
            k_mask,
            n_iters=args.n_iters,
            band_width=args.band_width,
            refine_steps=refine_steps,
            epsilon=args.epsilon,
        )
        for refine_steps in refine_steps_list
    ]

    grad_sequence = [case["grad_max_abs"] for case in cases]
    report = {
        "settings": {
            "seed": args.seed,
            "length": args.length,
            "dim": args.dim,
            "n_iters": args.n_iters,
            "band_width": args.band_width,
            "epsilon": args.epsilon,
            "refine_steps": refine_steps_list,
        },
        "cases": cases,
        "grad_max_abs_monotone_nonincreasing": all(
            grad_sequence[i + 1] <= grad_sequence[i] + 1e-12 for i in range(len(grad_sequence) - 1)
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
