#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jax_implementations import flash_sinkhorn_truth as jax_kernel
from pallas_kernels import pallas_sinkhorn as pallas_kernel
from src.benchmarks.benchmark_pfam_training import apply_projection_manifold
from src import ot_kernel


def _make_inputs(seed, length, dim):
    key = jax.random.PRNGKey(seed)
    q_key, k_key, v_key, t_key, wq_key, wk_key, wv_key = jax.random.split(key, 7)
    shape = (1, 1, length, dim)
    q = jax.random.normal(q_key, shape, dtype=jnp.float32)
    k = jax.random.normal(k_key, shape, dtype=jnp.float32)
    v = jax.random.normal(v_key, shape, dtype=jnp.float32)
    target = jax.random.normal(t_key, shape, dtype=jnp.float32)

    q_valid = int(length * 3 // 4)
    k_valid = int(length * 7 // 8)
    q_mask = (jnp.arange(length) < q_valid)[None, None, :]
    k_mask = (jnp.arange(length) < k_valid)[None, None, :]

    params = {
        "W_q": jax.random.normal(wq_key, (64, 16), dtype=jnp.float32),
        "W_k": jax.random.normal(wk_key, (64, 16), dtype=jnp.float32),
        "W_v": jax.random.normal(wv_key, (64, 16), dtype=jnp.float32),
    }
    return q, k, v, target, q_mask, k_mask, params


def _loss(fn, q, k, v, target, q_mask, k_mask, n_iters, band_width, refine_steps, refine_tolerance, gradient_mode):
    out = fn(
        q,
        k,
        v,
        n_iters=n_iters,
        band_width=band_width,
        q_mask=q_mask,
        k_mask=k_mask,
        refine_steps=refine_steps,
        refine_tolerance=refine_tolerance,
        gradient_mode=gradient_mode,
    )
    return jnp.sum(out * target) / out.size


def _grad_case(name, q, k, v, target, q_mask, k_mask, n_iters, band_width, refine_steps, refine_tolerance):
    def tail_loss(q_, k_, v_):
        return _loss(
            ot_kernel.sinkhorn_attention,
            q_,
            k_,
            v_,
            target,
            q_mask,
            k_mask,
            n_iters,
            band_width,
            refine_steps,
            refine_tolerance,
            "tail_refinement",
        )

    def exact_loss(q_, k_, v_):
        return _loss(
            ot_kernel.sinkhorn_attention,
            q_,
            k_,
            v_,
            target,
            q_mask,
            k_mask,
            n_iters,
            band_width,
            refine_steps,
            refine_tolerance,
            "exact_autodiff",
        )

    tail_out = ot_kernel.sinkhorn_attention(
        q,
        k,
        v,
        n_iters=n_iters,
        band_width=band_width,
        q_mask=q_mask,
        k_mask=k_mask,
        refine_steps=refine_steps,
        refine_tolerance=refine_tolerance,
        gradient_mode="tail_refinement",
    )
    exact_out = ot_kernel.sinkhorn_attention(
        q,
        k,
        v,
        n_iters=n_iters,
        band_width=band_width,
        q_mask=q_mask,
        k_mask=k_mask,
        refine_steps=refine_steps,
        refine_tolerance=refine_tolerance,
        gradient_mode="exact_autodiff",
    )
    tail_val, tail_grads = jax.value_and_grad(tail_loss, argnums=(0, 1, 2))(q, k, v)
    exact_val, exact_grads = jax.value_and_grad(exact_loss, argnums=(0, 1, 2))(q, k, v)
    jax.block_until_ready((tail_out, exact_out, tail_val, exact_val, tail_grads, exact_grads))

    out_diff = float(jnp.max(jnp.abs(tail_out - exact_out)))
    grad_diffs = {
        "dQ_max_abs": float(jnp.max(jnp.abs(tail_grads[0] - exact_grads[0]))),
        "dK_max_abs": float(jnp.max(jnp.abs(tail_grads[1] - exact_grads[1]))),
        "dV_max_abs": float(jnp.max(jnp.abs(tail_grads[2] - exact_grads[2]))),
    }
    return {
        "name": name,
        "tail_loss": float(tail_val),
        "exact_loss": float(exact_val),
        "output_max_abs": out_diff,
        **grad_diffs,
    }


def _module_agreement(q, k, v, q_mask, k_mask, n_iters, band_width):
    kwargs = dict(
        n_iters=n_iters,
        band_width=band_width,
        q_mask=q_mask,
        k_mask=k_mask,
        refine_steps=2,
        refine_tolerance=None,
        gradient_mode="tail_refinement",
    )
    src_out = ot_kernel.sinkhorn_attention(q, k, v, **kwargs)
    jax_out = jax_kernel.sinkhorn_attention(q, k, v, **kwargs)
    pallas_out = pallas_kernel.sinkhorn_attention(q, k, v, **kwargs)
    jax.block_until_ready((src_out, jax_out, pallas_out))
    return {
        "src_vs_jax_max_abs": float(jnp.max(jnp.abs(src_out - jax_out))),
        "src_vs_pallas_max_abs": float(jnp.max(jnp.abs(src_out - pallas_out))),
    }


def _stiefel_case(params):
    projected = apply_projection_manifold(params, "stiefel_qk")
    wq_err = float(jnp.max(jnp.abs(projected["W_q"].T @ projected["W_q"] - jnp.eye(projected["W_q"].shape[1], dtype=projected["W_q"].dtype))))
    wk_err = float(jnp.max(jnp.abs(projected["W_k"].T @ projected["W_k"] - jnp.eye(projected["W_k"].shape[1], dtype=projected["W_k"].dtype))))
    wv_diff = float(jnp.max(jnp.abs(projected["W_v"] - params["W_v"])))
    return {
        "W_q_orth_max_abs": wq_err,
        "W_k_orth_max_abs": wk_err,
        "W_v_change_max_abs": wv_diff,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--n_iters", type=int, default=2)
    parser.add_argument("--band_width", type=int, default=16)
    parser.add_argument("--grad_tol", type=float, default=2e-4)
    parser.add_argument("--out_tol", type=float, default=1e-6)
    parser.add_argument("--orth_tol", type=float, default=1e-5)
    args = parser.parse_args()

    q, k, v, target, q_mask, k_mask, params = _make_inputs(args.seed, args.length, args.dim)
    report = {
        "module_agreement": _module_agreement(q, k, v, q_mask, k_mask, args.n_iters, args.band_width),
        "k2_fast_path": _grad_case(
            "k2_fast_path",
            q,
            k,
            v,
            target,
            q_mask,
            k_mask,
            args.n_iters,
            args.band_width,
            2,
            None,
        ),
        "adaptive_reference": _grad_case(
            "adaptive_reference",
            q,
            k,
            v,
            target,
            q_mask,
            k_mask,
            args.n_iters,
            args.band_width,
            4,
            1e-3,
        ),
        "fixed_k_orbit": _grad_case(
            "fixed_k_orbit",
            q,
            k,
            v,
            target,
            q_mask,
            k_mask,
            args.n_iters,
            args.band_width,
            4,
            None,
        ),
        "stiefel_projection": _stiefel_case(params),
    }

    failures = []
    module_agreement = report["module_agreement"]
    if module_agreement["src_vs_jax_max_abs"] > args.out_tol:
        failures.append(f"src_vs_jax_max_abs={module_agreement['src_vs_jax_max_abs']}")
    if module_agreement["src_vs_pallas_max_abs"] > args.out_tol:
        failures.append(f"src_vs_pallas_max_abs={module_agreement['src_vs_pallas_max_abs']}")

    for case_name in ("k2_fast_path", "adaptive_reference", "fixed_k_orbit"):
        case = report[case_name]
        if case["output_max_abs"] > args.out_tol:
            failures.append(f"{case_name}.output_max_abs={case['output_max_abs']}")
        for grad_key in ("dQ_max_abs", "dK_max_abs", "dV_max_abs"):
            if case[grad_key] > args.grad_tol:
                failures.append(f"{case_name}.{grad_key}={case[grad_key]}")

    stiefel = report["stiefel_projection"]
    if stiefel["W_q_orth_max_abs"] > args.orth_tol:
        failures.append(f"W_q_orth_max_abs={stiefel['W_q_orth_max_abs']}")
    if stiefel["W_k_orth_max_abs"] > args.orth_tol:
        failures.append(f"W_k_orth_max_abs={stiefel['W_k_orth_max_abs']}")
    if stiefel["W_v_change_max_abs"] != 0.0:
        failures.append(f"W_v_change_max_abs={stiefel['W_v_change_max_abs']}")

    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit("validation failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
