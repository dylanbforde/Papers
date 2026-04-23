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

from src import ot_kernel, unbalanced_ot
from src.benchmarks.benchmark_pfam_training import (
    DUSTBIN_BLOCK_SIZE,
    _apply_gap_mode,
    _effective_band_width,
)


def _make_problem(seed: int, length: int, dim: int):
    key = jax.random.PRNGKey(seed)
    q_key, k_key, v_key, gq_key, gk_key, gv_key = jax.random.split(key, 6)

    q = jax.random.normal(q_key, (1, 1, length, dim), dtype=jnp.float32)
    k = jax.random.normal(k_key, (1, 1, length, dim), dtype=jnp.float32)
    v = jax.random.normal(v_key, (1, 1, length, dim), dtype=jnp.float32)

    q_valid = length - 2
    k_valid = length - 1
    q_mask = jnp.arange(length) < q_valid
    k_mask = jnp.arange(length) < k_valid
    target_mask = q_mask & ((jnp.arange(length) % 4) != 0)
    target_cols = jnp.where(target_mask, jnp.minimum(jnp.arange(length) + 1, k_valid - 1), 0).astype(jnp.int32)

    params = {
        "gap_q": jax.random.normal(gq_key, (dim,), dtype=jnp.float32),
        "gap_k": jax.random.normal(gk_key, (dim,), dtype=jnp.float32),
        "gap_v": jax.random.normal(gv_key, (dim,), dtype=jnp.float32),
    }
    return {
        "Q": q,
        "K": k,
        "V": v,
        "q_mask": q_mask[None, :],
        "k_mask": k_mask[None, :],
        "target_cols": target_cols[None, :],
        "target_mask": target_mask[None, :],
        "params": params,
        "base_length": length,
    }


def _materialize_plan(q, k, u, v, q_mask, k_mask, band_width, epsilon):
    length = q.shape[-2]
    score = jnp.einsum("bhmd,bhnd->bhmn", q, k) / (jnp.sqrt(q.shape[-1]) * epsilon)
    valid = q_mask[:, None, :, None] & k_mask[:, None, None, :]
    if band_width is not None and int(band_width) < length:
        rows = jnp.arange(length, dtype=jnp.int32)[:, None]
        cols = jnp.arange(length, dtype=jnp.int32)[None, :]
        valid = valid & (jnp.abs(rows - cols) <= int(band_width))[None, None, :, :]
    return jnp.where(valid, jnp.exp(score + u[..., :, None] + v[..., None, :]), 0.0)


def _dustbin_reference(problem, *, n_iters, refine_steps, epsilon, band_width):
    q_aug, k_aug, v_aug, target_cols_aug, target_mask_aug, q_mask_aug, k_mask_aug = _apply_gap_mode(
        problem["params"],
        problem["Q"],
        problem["K"],
        problem["V"],
        problem["target_cols"],
        problem["target_mask"],
        problem["q_mask"],
        problem["k_mask"],
        "dustbin_block",
    )
    del target_cols_aug, target_mask_aug
    band_eff = _effective_band_width(
        band_width,
        "dustbin_block",
        q_aug.shape[-2] - DUSTBIN_BLOCK_SIZE,
    )
    u0, v0 = ot_kernel.compute_global_uv(
        q_aug,
        k_aug,
        n_iters=n_iters,
        band_width=band_eff,
        q_mask=q_mask_aug,
        k_mask=k_mask_aug,
        epsilon=epsilon,
        epsilon_scaling_steps=0,
        epsilon_scaling_factor=2.0,
        support_reconstruction="none",
    )
    scale = 1.0 / (jnp.sqrt(q_aug.shape[-1]) * epsilon)
    u_hist, v_hist = ot_kernel._fixed_refinement_history(
        q_aug,
        k_aug,
        u0,
        v0,
        refine_steps,
        band_eff,
        q_mask_aug,
        k_mask_aug,
        scale,
        support_reconstruction="none",
    )
    plan = _materialize_plan(q_aug, k_aug, u_hist[-1], v_hist[-1], q_mask_aug, k_mask_aug, band_eff, epsilon)[0, 0]
    out = (plan @ v_aug[0, 0]).astype(jnp.float32)
    base_len = problem["base_length"]
    dustbin_index = base_len
    q_mask_base = problem["q_mask"][0]
    k_mask_base = problem["k_mask"][0]
    return {
        "output": out[:base_len],
        "plan": plan,
        "row_sums_base": jnp.sum(plan[:base_len, :], axis=-1),
        "col_sums_base": jnp.sum(plan[:, :base_len], axis=0),
        "base_to_dustbin_mass": float(jnp.sum(plan[:base_len, dustbin_index])),
        "dustbin_to_base_mass": float(jnp.sum(plan[dustbin_index, :base_len])),
        "base_active_rows": q_mask_base,
        "base_active_cols": k_mask_base,
    }


def _unbalanced_reference(problem, *, n_iters, refine_steps, epsilon, band_width, tau_q, tau_k):
    q = problem["Q"][0, 0]
    k = problem["K"][0, 0]
    v = problem["V"][0, 0]
    q_mask = problem["q_mask"][0]
    k_mask = problem["k_mask"][0]
    support = q_mask[:, None] & k_mask[None, :]
    if band_width is not None and band_width < q.shape[0]:
        rows = jnp.arange(q.shape[0], dtype=jnp.int32)[:, None]
        cols = jnp.arange(q.shape[0], dtype=jnp.int32)[None, :]
        support = support & (jnp.abs(rows - cols) <= int(band_width))

    log_a, log_b = unbalanced_ot.make_unit_log_masses(q_mask, k_mask)
    alpha_q, alpha_k = unbalanced_ot.alphas_from_taus(tau_q, tau_k, epsilon)
    score = unbalanced_ot.score_from_qk(q, k, epsilon)
    u_hist, v_hist = unbalanced_ot.fixed_tail_history_unbalanced_dense_from_score(
        score,
        support,
        log_a,
        log_b,
        q_mask,
        k_mask,
        n_iters=n_iters,
        refine_steps=refine_steps,
        alpha_q=alpha_q,
        alpha_k=alpha_k,
    )
    plan = unbalanced_ot.plan_from_score_and_duals(score, u_hist[-1], v_hist[-1], support)
    out = unbalanced_ot.surrogate_output_unbalanced_dense_from_score(
        score,
        v,
        support,
        log_a,
        log_b,
        q_mask,
        k_mask,
        n_iters=n_iters,
        refine_steps=refine_steps,
        alpha_q=alpha_q,
        alpha_k=alpha_k,
    )
    row_sums = jnp.sum(plan, axis=-1)
    col_sums = jnp.sum(plan, axis=0)
    a = jnp.where(q_mask, jnp.exp(log_a), 0.0)
    b = jnp.where(k_mask, jnp.exp(log_b), 0.0)
    return {
        "output": out,
        "plan": plan,
        "row_sums": row_sums,
        "col_sums": col_sums,
        "row_gap_l1": float(jnp.sum(jnp.abs(row_sums - a))),
        "col_gap_l1": float(jnp.sum(jnp.abs(col_sums - b))),
        "total_mass": float(jnp.sum(plan)),
        "alpha_q": float(alpha_q),
        "alpha_k": float(alpha_k),
        "target_row_mass": 1.0,
        "target_col_mass": 1.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--n_iters", type=int, default=15)
    parser.add_argument("--refine_steps", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--band_width", type=int, default=1024)
    parser.add_argument("--tau_q", type=float, default=1.0)
    parser.add_argument("--tau_k", type=float, default=1.0)
    args = parser.parse_args()

    if args.length % DUSTBIN_BLOCK_SIZE != 0:
        raise ValueError(
            f"--length must be divisible by {DUSTBIN_BLOCK_SIZE} for the "
            f"cloud-style dustbin_block reference, got {args.length}."
        )

    problem = _make_problem(args.seed, args.length, args.dim)
    dustbin = _dustbin_reference(
        problem,
        n_iters=args.n_iters,
        refine_steps=args.refine_steps,
        epsilon=args.epsilon,
        band_width=args.band_width,
    )
    unbalanced = _unbalanced_reference(
        problem,
        n_iters=args.n_iters,
        refine_steps=args.refine_steps,
        epsilon=args.epsilon,
        band_width=None if args.band_width >= args.length else args.band_width,
        tau_q=args.tau_q,
        tau_k=args.tau_k,
    )

    active_rows = dustbin["base_active_rows"]
    output_diff = dustbin["output"] - unbalanced["output"]
    active_output_l2 = float(
        jnp.sqrt(jnp.sum(jnp.square(jnp.where(active_rows[:, None], output_diff, 0.0))))
    )
    active_output_max_abs = float(jnp.max(jnp.abs(jnp.where(active_rows[:, None], output_diff, 0.0))))

    report = {
        "active_output_l2": active_output_l2,
        "active_output_max_abs": active_output_max_abs,
        "dustbin_base_to_dustbin_mass": dustbin["base_to_dustbin_mass"],
        "dustbin_dustbin_to_base_mass": dustbin["dustbin_to_base_mass"],
        "dustbin_total_mass": float(jnp.sum(dustbin["plan"])),
        "unbalanced_total_mass": unbalanced["total_mass"],
        "unbalanced_row_gap_l1": unbalanced["row_gap_l1"],
        "unbalanced_col_gap_l1": unbalanced["col_gap_l1"],
        "alpha_q": unbalanced["alpha_q"],
        "alpha_k": unbalanced["alpha_k"],
        "dustbin_row_sum_min": float(jnp.min(jnp.where(active_rows, dustbin["row_sums_base"], jnp.inf))),
        "dustbin_row_sum_max": float(jnp.max(jnp.where(active_rows, dustbin["row_sums_base"], 0.0))),
        "unbalanced_row_sum_min": float(jnp.min(jnp.where(active_rows, unbalanced["row_sums"], jnp.inf))),
        "unbalanced_row_sum_max": float(jnp.max(jnp.where(active_rows, unbalanced["row_sums"], 0.0))),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
