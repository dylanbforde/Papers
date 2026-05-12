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

from src import unbalanced_ot


def _make_problem(seed: int, length: int, dim: int, band_width: int | None):
    key = jax.random.PRNGKey(seed)
    q_key, k_key, v_key, do_key = jax.random.split(key, 4)
    q = jax.random.normal(q_key, (length, dim), dtype=jnp.float32)
    k = jax.random.normal(k_key, (length, dim), dtype=jnp.float32)
    v = jax.random.normal(v_key, (length, dim), dtype=jnp.float32)
    dO = jax.random.normal(do_key, (length, dim), dtype=jnp.float32)

    q_valid = length - 2
    k_valid = length - 1
    q_mask = jnp.arange(length) < q_valid
    k_mask = jnp.arange(length) < k_valid

    support = q_mask[:, None] & k_mask[None, :]
    if band_width is not None and band_width < length:
        rows = jnp.arange(length, dtype=jnp.int32)[:, None]
        cols = jnp.arange(length, dtype=jnp.int32)[None, :]
        support = support & (jnp.abs(rows - cols) <= int(band_width))

    log_a, log_b = unbalanced_ot.make_uniform_log_masses(q_mask, k_mask)
    dO = jnp.where(q_mask[:, None], dO, 0.0)
    return {
        "Q": q,
        "K": k,
        "V": v,
        "dO": dO,
        "q_mask": q_mask,
        "k_mask": k_mask,
        "support": support,
        "log_a": log_a,
        "log_b": log_b,
    }
def _surrogate_loss(Q, K, V, *, dO, support, log_a, log_b, q_mask, k_mask, epsilon, n_iters, refine_steps, alpha_q, alpha_k):
    score = unbalanced_ot.score_from_qk(Q, K, epsilon)
    out = unbalanced_ot.surrogate_output_unbalanced_dense_from_score(
        score,
        V,
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
    return jnp.sum(out * dO)

def _closed_form_grads(problem, *, epsilon, n_iters, refine_steps, alpha_q, alpha_k):
    Q = problem["Q"]
    K = problem["K"]
    score = unbalanced_ot.score_from_qk(Q, K, epsilon)
    closed = unbalanced_ot.fixed_r_tail_unbalanced_grads_dense_from_score(
        score,
        problem["V"],
        problem["dO"],
        problem["support"],
        problem["log_a"],
        problem["log_b"],
        problem["q_mask"],
        problem["k_mask"],
        n_iters=n_iters,
        refine_steps=refine_steps,
        alpha_q=alpha_q,
        alpha_k=alpha_k,
    )
    dS = closed["dS"]
    scale = 1.0 / (jnp.sqrt(Q.shape[-1]) * epsilon)
    dQ = dS @ K * scale
    dK = dS.T @ Q * scale
    return {"dQ": dQ, "dK": dK, "dV": closed["dV"], "dS": dS}


def _factorization_report(problem, *, epsilon, n_iters, refine_steps, alpha_q, alpha_k):
    if refine_steps != 2:
        raise ValueError("factorization report currently assumes refine_steps=2")
    score = unbalanced_ot.score_from_qk(problem["Q"], problem["K"], epsilon)
    u_hist, v_hist = unbalanced_ot.fixed_tail_history_unbalanced_dense_from_score(
        score,
        problem["support"],
        problem["log_a"],
        problem["log_b"],
        problem["q_mask"],
        problem["k_mask"],
        n_iters=n_iters,
        refine_steps=refine_steps,
        alpha_q=alpha_q,
        alpha_k=alpha_k,
    )
    plans = {
        "P22": unbalanced_ot.plan_from_score_and_duals(score, u_hist[2], v_hist[2], problem["support"]),
        "P21": unbalanced_ot.plan_from_score_and_duals(score, u_hist[2], v_hist[1], problem["support"]),
        "P11": unbalanced_ot.plan_from_score_and_duals(score, u_hist[1], v_hist[1], problem["support"]),
        "P10": unbalanced_ot.plan_from_score_and_duals(score, u_hist[1], v_hist[0], problem["support"]),
    }
    p22 = plans["P22"]
    report = {}
    for name, (u_a, v_b, ref_plan) in {
        "P21": (u_hist[2], v_hist[1], plans["P21"]),
        "P11": (u_hist[1], v_hist[1], plans["P11"]),
        "P10": (u_hist[1], v_hist[0], plans["P10"]),
    }.items():
        recon = p22 * jnp.exp(u_a - u_hist[2])[:, None] * jnp.exp(v_b - v_hist[2])[None, :]
        report[f"{name}_max_abs"] = float(jnp.max(jnp.abs(ref_plan - recon)))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--band_width", type=int, default=16)
    parser.add_argument("--n_iters", type=int, default=5)
    parser.add_argument("--refine_steps", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--alpha_q", type=float, default=0.7)
    parser.add_argument("--alpha_k", type=float, default=0.6)
    parser.add_argument("--grad_tol", type=float, default=2e-5)
    parser.add_argument("--plan_tol", type=float, default=2e-6)
    args = parser.parse_args()

    band_width = None if args.band_width >= args.length else args.band_width
    problem = _make_problem(args.seed, args.length, args.dim, band_width)

    def loss_fn(q, k, v):
        return _surrogate_loss(
            q,
            k,
            v,
            dO=problem["dO"],
            support=problem["support"],
            log_a=problem["log_a"],
            log_b=problem["log_b"],
            q_mask=problem["q_mask"],
            k_mask=problem["k_mask"],
            epsilon=args.epsilon,
            n_iters=args.n_iters,
            refine_steps=args.refine_steps,
            alpha_q=args.alpha_q,
            alpha_k=args.alpha_k,
        )

    exact_loss, exact_grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(problem["Q"], problem["K"], problem["V"])
    exact_dS = jax.grad(
        lambda s: jnp.sum(
            unbalanced_ot.surrogate_output_unbalanced_dense_from_score(
                s,
                problem["V"],
                problem["support"],
                problem["log_a"],
                problem["log_b"],
                problem["q_mask"],
                problem["k_mask"],
                n_iters=args.n_iters,
                refine_steps=args.refine_steps,
                alpha_q=args.alpha_q,
                alpha_k=args.alpha_k,
            )
            * problem["dO"]
        ),
    )(unbalanced_ot.score_from_qk(problem["Q"], problem["K"], args.epsilon))
    closed = _closed_form_grads(
        problem,
        epsilon=args.epsilon,
        n_iters=args.n_iters,
        refine_steps=args.refine_steps,
        alpha_q=args.alpha_q,
        alpha_k=args.alpha_k,
    )
    factorization = _factorization_report(
        problem,
        epsilon=args.epsilon,
        n_iters=args.n_iters,
        refine_steps=args.refine_steps,
        alpha_q=args.alpha_q,
        alpha_k=args.alpha_k,
    )

    report = {
        "loss": float(exact_loss),
        "dQ_max_abs": float(jnp.max(jnp.abs(closed["dQ"] - exact_grads[0]))),
        "dK_max_abs": float(jnp.max(jnp.abs(closed["dK"] - exact_grads[1]))),
        "dV_max_abs": float(jnp.max(jnp.abs(closed["dV"] - exact_grads[2]))),
        "dS_max_abs": float(jnp.max(jnp.abs(closed["dS"] - exact_dS))),
        **factorization,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    failures = []
    for key in ("dQ_max_abs", "dK_max_abs", "dV_max_abs", "dS_max_abs"):
        if report[key] > args.grad_tol:
            failures.append(f"{key}={report[key]}")
    for key in ("P21_max_abs", "P11_max_abs", "P10_max_abs"):
        if report[key] > args.plan_tol:
            failures.append(f"{key}={report[key]}")
    if failures:
        raise SystemExit("validation failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
