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

from src import ot_kernel
from src.benchmarks.benchmark_pfam_training import (
    DUSTBIN_BLOCK_SIZE,
    _apply_gap_mode,
    _alignment_loss_terms,
    _batch_take_rows,
    _effective_band_width,
    _prepare_sparse_supervision,
    output_reconstruction_loss,
    sinkhorn_outputs,
)


def _make_problem(seed: int, length: int, dim: int, batch: int):
    key = jax.random.PRNGKey(seed)
    q_key, k_key, v_key, gq_key, gk_key, gv_key = jax.random.split(key, 6)

    q = jax.random.normal(q_key, (batch, 1, length, dim), dtype=jnp.float32)
    k = jax.random.normal(k_key, (batch, 1, length, dim), dtype=jnp.float32)
    v = jax.random.normal(v_key, (batch, 1, length, dim), dtype=jnp.float32)

    q_valid = int(length - 2)
    k_valid = int(length - 1)
    q_mask = jnp.broadcast_to(jnp.arange(length) < q_valid, (batch, length))
    k_mask = jnp.broadcast_to(jnp.arange(length) < k_valid, (batch, length))

    base_targets = jnp.broadcast_to(jnp.minimum(jnp.arange(length) + 1, k_valid - 1), (batch, length))
    target_mask = q_mask & ((jnp.arange(length) % 3) != 0)[None, :]
    target_cols = jnp.where(target_mask, base_targets, 0).astype(jnp.int32)

    params = {
        "gap_q": jax.random.normal(gq_key, (dim,), dtype=jnp.float32),
        "gap_k": jax.random.normal(gk_key, (dim,), dtype=jnp.float32),
        "gap_v": jax.random.normal(gv_key, (dim,), dtype=jnp.float32),
    }
    return {
        "Q": q,
        "K": k,
        "V": v,
        "target_cols": target_cols,
        "target_mask": target_mask,
        "q_mask": q_mask,
        "k_mask": k_mask,
        "params": params,
    }


def _reconstruction_loss(
    q,
    k,
    v,
    target_cols,
    target_mask,
    q_mask,
    k_mask,
    *,
    gradient_mode: str,
    n_iters: int,
    band_width: int,
    refine_steps: int,
    epsilon: float,
):
    outputs = sinkhorn_outputs(
        q,
        k,
        v,
        q_mask=q_mask,
        k_mask=k_mask,
        n_iters=n_iters,
        band_width=band_width,
        gradient_mode=gradient_mode,
        refine_steps=refine_steps,
        refine_tolerance=None,
        epsilon=epsilon,
        epsilon_scaling_steps=0,
        epsilon_scaling_factor=2.0,
        support_reconstruction="none",
    )
    safe_cols, active_mask = _prepare_sparse_supervision(target_cols, target_mask, q_mask, k_mask)
    target_v = _batch_take_rows(v, safe_cols)
    sq_error = jnp.sum(jnp.square(outputs - target_v), axis=-1)
    denom = jnp.maximum(jnp.sum(active_mask) * q.shape[1], 1)
    return jnp.sum(jnp.where(active_mask[:, None, :], sq_error, 0.0)) / denom


def _weighted_alignment_loss_from_augmented(
    problem,
    *,
    weights,
    gradient_mode: str,
    n_iters: int,
    band_width: int,
    refine_steps: int,
    epsilon: float,
):
    params = problem["params"]
    q_aug, k_aug, v_aug, target_cols_aug, target_mask_aug, q_mask_aug, k_mask_aug = _apply_gap_mode(
        params,
        problem["Q"],
        problem["K"],
        problem["V"],
        problem["target_cols"],
        problem["target_mask"],
        problem["q_mask"],
        problem["k_mask"],
        "dustbin_block",
    )
    band_width_eff = _effective_band_width(
        band_width,
        "dustbin_block",
        target_cols_aug.shape[-1] - DUSTBIN_BLOCK_SIZE,
    )
    terms = _alignment_loss_terms(
        "sinkhorn_ot",
        q_aug,
        k_aug,
        v_aug,
        target_cols_aug,
        target_mask_aug,
        q_mask_aug,
        k_mask_aug,
        n_iters,
        band_width_eff,
        gradient_mode,
        refine_steps,
        None,
        epsilon,
        0,
        2.0,
        "none",
    )
    return (
        weights["reconstruction"] * terms["reconstruction"]
        + weights["sparse_ce"] * terms["sparse_ce"]
        + weights["monotonicity"] * terms["monotonicity"]
    )


def _weighted_alignment_loss_via_gap_mode(
    problem,
    *,
    weights,
    gradient_mode: str,
    n_iters: int,
    band_width: int,
    refine_steps: int,
    epsilon: float,
):
    return output_reconstruction_loss(
        "sinkhorn_ot",
        problem["Q"],
        problem["K"],
        problem["V"],
        problem["target_cols"],
        problem["target_mask"],
        problem["q_mask"],
        problem["k_mask"],
        n_iters,
        band_width,
        gradient_mode,
        refine_steps,
        None,
        epsilon,
        0,
        2.0,
        "none",
        weights["reconstruction"],
        weights["sparse_ce"],
        weights["monotonicity"],
        "dustbin_block",
        problem["params"],
    )


def _loss_and_grads(loss_fn, problem):
    def wrapped(q, k, v, gap_q, gap_k, gap_v):
        local_problem = {
            **problem,
            "Q": q,
            "K": k,
            "V": v,
            "params": {
                "gap_q": gap_q,
                "gap_k": gap_k,
                "gap_v": gap_v,
            },
        }
        return loss_fn(local_problem)

    value, grads = jax.value_and_grad(wrapped, argnums=(0, 1, 2, 3, 4, 5))(
        problem["Q"],
        problem["K"],
        problem["V"],
        problem["params"]["gap_q"],
        problem["params"]["gap_k"],
        problem["params"]["gap_v"],
    )
    jax.block_until_ready((value, grads))
    return value, grads


def _max_tree_diff(tree_a, tree_b):
    leaves_a = jax.tree_util.tree_leaves(tree_a)
    leaves_b = jax.tree_util.tree_leaves(tree_b)
    return max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(leaves_a, leaves_b))


def _materialize_plan(q, k, u, v, q_mask, k_mask, band_width, epsilon):
    length = q.shape[-2]
    score = jnp.einsum("bhmd,bhnd->bhmn", q, k) / (jnp.sqrt(q.shape[-1]) * epsilon)
    valid = q_mask[:, None, :, None] & k_mask[:, None, None, :]
    if band_width is not None and int(band_width) < length:
        rows = jnp.arange(length, dtype=jnp.int32)[:, None]
        cols = jnp.arange(length, dtype=jnp.int32)[None, :]
        band = jnp.abs(rows - cols) <= int(band_width)
        valid = valid & band[None, None, :, :]
    log_plan = score + u[..., :, None] + v[..., None, :]
    return jnp.where(valid, jnp.exp(log_plan), 0.0)


def _factorization_report(problem, *, n_iters: int, band_width: int, refine_steps: int, epsilon: float):
    if refine_steps != 2:
        raise ValueError("factorization report currently assumes refine_steps=2")

    params = problem["params"]
    q_aug, k_aug, v_aug, target_cols_aug, _target_mask_aug, q_mask_aug, k_mask_aug = _apply_gap_mode(
        params,
        problem["Q"],
        problem["K"],
        problem["V"],
        problem["target_cols"],
        problem["target_mask"],
        problem["q_mask"],
        problem["k_mask"],
        "dustbin_block",
    )
    del v_aug, target_cols_aug
    band_width_eff = _effective_band_width(
        band_width,
        "dustbin_block",
        problem["target_cols"].shape[-1],
    )

    u0, v0 = ot_kernel.compute_global_uv(
        q_aug,
        k_aug,
        n_iters=n_iters,
        band_width=band_width_eff,
        q_mask=q_mask_aug,
        k_mask=k_mask_aug,
        epsilon=epsilon,
        epsilon_scaling_steps=0,
        epsilon_scaling_factor=2.0,
        support_reconstruction="none",
    )
    score_scale = 1.0 / (jnp.sqrt(q_aug.shape[-1]) * epsilon)
    u_hist, v_hist = ot_kernel._fixed_refinement_history(
        q_aug,
        k_aug,
        u0,
        v0,
        refine_steps,
        band_width_eff,
        q_mask_aug,
        k_mask_aug,
        score_scale,
        support_reconstruction="none",
    )

    plans = {
        "P22": _materialize_plan(q_aug, k_aug, u_hist[2], v_hist[2], q_mask_aug, k_mask_aug, band_width_eff, epsilon),
        "P21": _materialize_plan(q_aug, k_aug, u_hist[2], v_hist[1], q_mask_aug, k_mask_aug, band_width_eff, epsilon),
        "P11": _materialize_plan(q_aug, k_aug, u_hist[1], v_hist[1], q_mask_aug, k_mask_aug, band_width_eff, epsilon),
        "P10": _materialize_plan(q_aug, k_aug, u_hist[1], v_hist[0], q_mask_aug, k_mask_aug, band_width_eff, epsilon),
    }

    p22 = plans["P22"]
    reports = {}
    for name, (u_a, v_b, ref_plan) in {
        "P21": (u_hist[2], v_hist[1], plans["P21"]),
        "P11": (u_hist[1], v_hist[1], plans["P11"]),
        "P10": (u_hist[1], v_hist[0], plans["P10"]),
    }.items():
        recon = p22 * jnp.exp(u_a - u_hist[2])[..., :, None] * jnp.exp(v_b - v_hist[2])[..., None, :]
        reports[f"{name}_max_abs"] = float(jnp.max(jnp.abs(ref_plan - recon)))
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--n_iters", type=int, default=15)
    parser.add_argument("--band_width", type=int, default=1024)
    parser.add_argument("--refine_steps", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--loss_tol", type=float, default=1e-6)
    parser.add_argument("--grad_tol", type=float, default=2e-4)
    parser.add_argument("--plan_tol", type=float, default=2e-5)
    args = parser.parse_args()

    problem = _make_problem(args.seed, args.length, args.dim, args.batch)
    configs = {
        "reconstruction_only": {
            "reconstruction": 1.0,
            "sparse_ce": 0.0,
            "monotonicity": 0.0,
        },
        "muot_dustbin_mono": {
            "reconstruction": 1.0,
            "sparse_ce": 0.0,
            "monotonicity": 0.05,
        },
        "muot_dustbin_hybrid": {
            "reconstruction": 0.5,
            "sparse_ce": 0.5,
            "monotonicity": 0.05,
        },
    }

    config_reports = {}
    for name, weights in configs.items():
        weighted_gap_tail = lambda p, weights=weights: _weighted_alignment_loss_via_gap_mode(
            p,
            weights=weights,
            gradient_mode="tail_refinement",
            n_iters=args.n_iters,
            band_width=args.band_width,
            refine_steps=args.refine_steps,
            epsilon=args.epsilon,
        )
        weighted_aug_tail = lambda p, weights=weights: _weighted_alignment_loss_from_augmented(
            p,
            weights=weights,
            gradient_mode="tail_refinement",
            n_iters=args.n_iters,
            band_width=args.band_width,
            refine_steps=args.refine_steps,
            epsilon=args.epsilon,
        )
        weighted_gap_exact = lambda p, weights=weights: _weighted_alignment_loss_via_gap_mode(
            p,
            weights=weights,
            gradient_mode="exact_autodiff",
            n_iters=args.n_iters,
            band_width=args.band_width,
            refine_steps=args.refine_steps,
            epsilon=args.epsilon,
        )

        gap_tail_val, gap_tail_grads = _loss_and_grads(weighted_gap_tail, problem)
        aug_tail_val, aug_tail_grads = _loss_and_grads(weighted_aug_tail, problem)
        gap_exact_val, gap_exact_grads = _loss_and_grads(weighted_gap_exact, problem)
        config_reports[name] = {
            "gap_vs_manual_loss_abs": float(jnp.abs(gap_tail_val - aug_tail_val)),
            "gap_vs_manual_grad_max_abs": _max_tree_diff(gap_tail_grads, aug_tail_grads),
            "tail_vs_exact_loss_abs": float(jnp.abs(gap_tail_val - gap_exact_val)),
            "tail_vs_exact_grad_max_abs": _max_tree_diff(gap_tail_grads, gap_exact_grads),
        }

    factorization = _factorization_report(
        problem,
        n_iters=args.n_iters,
        band_width=args.band_width,
        refine_steps=args.refine_steps,
        epsilon=args.epsilon,
    )

    report = {"configs": config_reports, **factorization}
    print(json.dumps(report, indent=2, sort_keys=True))

    failures = []
    for name, config_report in config_reports.items():
        if config_report["gap_vs_manual_loss_abs"] > args.loss_tol:
            failures.append(f"{name}.gap_vs_manual_loss_abs={config_report['gap_vs_manual_loss_abs']}")
        if config_report["gap_vs_manual_grad_max_abs"] > args.grad_tol:
            failures.append(f"{name}.gap_vs_manual_grad_max_abs={config_report['gap_vs_manual_grad_max_abs']}")
        if config_report["tail_vs_exact_loss_abs"] > args.loss_tol:
            failures.append(f"{name}.tail_vs_exact_loss_abs={config_report['tail_vs_exact_loss_abs']}")
        if config_report["tail_vs_exact_grad_max_abs"] > args.grad_tol:
            failures.append(f"{name}.tail_vs_exact_grad_max_abs={config_report['tail_vs_exact_grad_max_abs']}")
    for key in ("P21_max_abs", "P11_max_abs", "P10_max_abs"):
        if report[key] > args.plan_tol:
            failures.append(f"{key}={report[key]}")

    if failures:
        raise SystemExit("validation failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
