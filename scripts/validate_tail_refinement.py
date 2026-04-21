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

from jax_implementations import flash_sinkhorn_truth as jax_kernel
from pallas_kernels import pallas_sinkhorn as pallas_kernel
from src.benchmarks.benchmark_pfam_training import apply_projection_manifold
from src import ot_kernel


def _parse_length_list(raw_lengths):
    lengths = []
    for item in raw_lengths.split(","):
        item = item.strip()
        if not item:
            continue
        lengths.append(int(item))
    if not lengths:
        raise ValueError("at least one validation length is required")
    return lengths


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


def _loss(
    fn,
    q,
    k,
    v,
    target,
    q_mask,
    k_mask,
    n_iters,
    band_width,
    refine_steps,
    refine_tolerance,
    gradient_mode,
    epsilon=1.0,
    epsilon_scaling_steps=0,
    epsilon_scaling_factor=2.0,
    support_reconstruction="none",
):
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
        epsilon=epsilon,
        epsilon_scaling_steps=epsilon_scaling_steps,
        epsilon_scaling_factor=epsilon_scaling_factor,
        gradient_mode=gradient_mode,
        support_reconstruction=support_reconstruction,
    )
    return jnp.sum(out * target) / out.size


def _grad_case(
    name,
    q,
    k,
    v,
    target,
    q_mask,
    k_mask,
    n_iters,
    band_width,
    refine_steps,
    refine_tolerance,
    epsilon=1.0,
    epsilon_scaling_steps=0,
    epsilon_scaling_factor=2.0,
    support_reconstruction="none",
):
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
            epsilon,
            epsilon_scaling_steps,
            epsilon_scaling_factor,
            support_reconstruction,
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
            epsilon,
            epsilon_scaling_steps,
            epsilon_scaling_factor,
            support_reconstruction,
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
        epsilon=epsilon,
        epsilon_scaling_steps=epsilon_scaling_steps,
        epsilon_scaling_factor=epsilon_scaling_factor,
        gradient_mode="tail_refinement",
        support_reconstruction=support_reconstruction,
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
        epsilon=epsilon,
        epsilon_scaling_steps=epsilon_scaling_steps,
        epsilon_scaling_factor=epsilon_scaling_factor,
        gradient_mode="exact_autodiff",
        support_reconstruction=support_reconstruction,
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
        epsilon=1.0,
        epsilon_scaling_steps=0,
        epsilon_scaling_factor=2.0,
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


def _module_agreement_case(seed, length, dim, n_iters, band_width):
    q, k, v, _target, q_mask, k_mask, _params = _make_inputs(seed, length, dim)
    return _module_agreement(q, k, v, q_mask, k_mask, n_iters, band_width)


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


def _epsilon_scaling_case(q, k, v, q_mask, k_mask, band_width):
    common = dict(
        band_width=band_width,
        q_mask=q_mask,
        k_mask=k_mask,
        refine_steps=0,
        refine_tolerance=None,
        gradient_mode="exact_autodiff",
        epsilon=0.5,
    )
    reference = ot_kernel.sinkhorn_attention(q, k, v, n_iters=64, epsilon_scaling_steps=0, epsilon_scaling_factor=2.0, **common)
    plain = ot_kernel.sinkhorn_attention(q, k, v, n_iters=8, epsilon_scaling_steps=0, epsilon_scaling_factor=2.0, **common)
    scaled = ot_kernel.sinkhorn_attention(q, k, v, n_iters=8, epsilon_scaling_steps=2, epsilon_scaling_factor=2.0, **common)
    jax.block_until_ready((reference, plain, scaled))
    return {
        "reference_vs_plain_max_abs": float(jnp.max(jnp.abs(reference - plain))),
        "reference_vs_scaled_max_abs": float(jnp.max(jnp.abs(reference - scaled))),
        "scaled_improves": bool(jnp.max(jnp.abs(reference - scaled)) <= jnp.max(jnp.abs(reference - plain))),
    }


def _offdiag_transport_support_case(seed, length, dim, band_width, shift):
    key = jax.random.PRNGKey(seed)
    base_key, noise_key = jax.random.split(key)
    k = jax.random.normal(base_key, (1, 1, length, dim), dtype=jnp.float32)
    target_cols = jnp.clip(jnp.arange(length, dtype=jnp.int32) + int(shift), 0, length - 1)
    q = jnp.take(k, target_cols, axis=-2) + 0.01 * jax.random.normal(noise_key, k.shape, dtype=k.dtype)
    q_mask = jnp.ones((1, 1, length), dtype=bool)
    k_mask = jnp.ones((1, 1, length), dtype=bool)

    diag_mask = ot_kernel.materialize_support_mask(
        q,
        k,
        band_width=band_width,
        q_mask=q_mask,
        k_mask=k_mask,
        support_reconstruction="none",
    )
    scout_mask = ot_kernel.materialize_support_mask(
        q,
        k,
        band_width=band_width,
        q_mask=q_mask,
        k_mask=k_mask,
        support_reconstruction="transport_scout",
    )
    rows = jnp.arange(length, dtype=jnp.int32)
    return {
        "diagonal_support_recall": float(jnp.mean(diag_mask[rows, target_cols].astype(jnp.float32))),
        "transport_scout_support_recall": float(jnp.mean(scout_mask[rows, target_cols].astype(jnp.float32))),
        "shift_tokens": int(shift),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--module_lengths", type=str, default="512,1024,2048,4096,8192,16384")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--n_iters", type=int, default=2)
    parser.add_argument("--band_width", type=int, default=256)
    parser.add_argument("--grad_tol", type=float, default=2e-4)
    parser.add_argument("--out_tol", type=float, default=1e-6)
    parser.add_argument("--orth_tol", type=float, default=1e-5)
    parser.add_argument("--support_gain_tol", type=float, default=0.0)
    args = parser.parse_args()

    module_lengths = _parse_length_list(args.module_lengths)
    q, k, v, target, q_mask, k_mask, params = _make_inputs(args.seed, args.length, args.dim)
    support_length = max(max(module_lengths), args.length, 2048)
    support_shift = min(max(args.band_width * 2, 256), support_length // 2)
    report = {
        "module_agreement": {
            str(length): _module_agreement_case(args.seed, length, args.dim, args.n_iters, args.band_width)
            for length in module_lengths
        },
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
        "transport_scout_grad": _grad_case(
            "transport_scout_grad",
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
            support_reconstruction="transport_scout",
        ),
        "epsilon_scaling": _epsilon_scaling_case(q, k, v, q_mask, k_mask, args.band_width),
        "offdiag_support": _offdiag_transport_support_case(
            args.seed + 17,
            support_length,
            args.dim,
            args.band_width,
            shift=support_shift,
        ),
        "stiefel_projection": _stiefel_case(params),
    }

    failures = []
    for length_key, module_agreement in report["module_agreement"].items():
        if module_agreement["src_vs_jax_max_abs"] > args.out_tol:
            failures.append(f"L{length_key}.src_vs_jax_max_abs={module_agreement['src_vs_jax_max_abs']}")
        if module_agreement["src_vs_pallas_max_abs"] > args.out_tol:
            failures.append(f"L{length_key}.src_vs_pallas_max_abs={module_agreement['src_vs_pallas_max_abs']}")

    for case_name in ("k2_fast_path", "adaptive_reference", "fixed_k_orbit", "transport_scout_grad"):
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

    offdiag = report["offdiag_support"]
    if offdiag["transport_scout_support_recall"] < offdiag["diagonal_support_recall"] + args.support_gain_tol:
        failures.append(
            "offdiag_support_recall_gain="
            f"{offdiag['transport_scout_support_recall'] - offdiag['diagonal_support_recall']}"
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit("validation failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
