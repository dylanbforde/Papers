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


def _parse_int_list(raw):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("expected at least one integer")
    return values


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


def _tail_output_from_base(
    q,
    k,
    v,
    u0,
    v0,
    q_mask,
    k_mask,
    *,
    refine_steps,
    band_width,
    epsilon,
):
    score_scale = ot_kernel._score_scale(q.shape[-1], epsilon)
    block = 128
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


def _base_duals(q, k, q_mask, k_mask, *, n_iters, band_width, epsilon):
    return ot_kernel.compute_global_uv(
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


def _loss_from_base(q, k, v, u0, v0, target, q_mask, k_mask, *, refine_steps, band_width, epsilon):
    out = _tail_output_from_base(
        q,
        k,
        v,
        u0,
        v0,
        q_mask,
        k_mask,
        refine_steps=refine_steps,
        band_width=band_width,
        epsilon=epsilon,
    )
    return jnp.sum(out * target) / out.size


def _tree_l2(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves)))


def _tree_max_abs(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return max(float(jnp.max(jnp.abs(x))) for x in leaves)


def _tree_sub(a, b):
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def _case(q, k, v, target, q_mask, k_mask, *, n_iters, band_width, refine_steps, epsilon):
    def full_loss(q_, k_, v_):
        u0, v0 = _base_duals(
            q_,
            k_,
            q_mask,
            k_mask,
            n_iters=n_iters,
            band_width=band_width,
            epsilon=epsilon,
        )
        return _loss_from_base(
            q_,
            k_,
            v_,
            u0,
            v0,
            target,
            q_mask,
            k_mask,
            refine_steps=refine_steps,
            band_width=band_width,
            epsilon=epsilon,
        )

    def stopped_loss(q_, k_, v_):
        u0, v0 = _base_duals(
            q_,
            k_,
            q_mask,
            k_mask,
            n_iters=n_iters,
            band_width=band_width,
            epsilon=epsilon,
        )
        return _loss_from_base(
            q_,
            k_,
            v_,
            jax.lax.stop_gradient(u0),
            jax.lax.stop_gradient(v0),
            target,
            q_mask,
            k_mask,
            refine_steps=refine_steps,
            band_width=band_width,
            epsilon=epsilon,
        )

    u0, v0 = _base_duals(
        q,
        k,
        q_mask,
        k_mask,
        n_iters=n_iters,
        band_width=band_width,
        epsilon=epsilon,
    )

    def base_seed_loss(u0_, v0_):
        return _loss_from_base(
            jax.lax.stop_gradient(q),
            jax.lax.stop_gradient(k),
            jax.lax.stop_gradient(v),
            u0_,
            v0_,
            target,
            q_mask,
            k_mask,
            refine_steps=refine_steps,
            band_width=band_width,
            epsilon=epsilon,
        )

    full_value, full_grads = jax.value_and_grad(full_loss, argnums=(0, 1, 2))(q, k, v)
    stopped_value, stopped_grads = jax.value_and_grad(stopped_loss, argnums=(0, 1, 2))(q, k, v)
    eta_u, eta_v = jax.grad(base_seed_loss, argnums=(0, 1))(u0, v0)
    eta_u = jax.lax.stop_gradient(eta_u)
    eta_v = jax.lax.stop_gradient(eta_v)

    def certificate_scalar(q_, k_):
        base_u, base_v = _base_duals(
            q_,
            k_,
            q_mask,
            k_mask,
            n_iters=n_iters,
            band_width=band_width,
            epsilon=epsilon,
        )
        return jnp.sum(base_u * eta_u) + jnp.sum(base_v * eta_v)

    cert_q, cert_k = jax.grad(certificate_scalar, argnums=(0, 1))(q, k)
    jax.block_until_ready((full_value, stopped_value, full_grads, stopped_grads, eta_u, eta_v, cert_q, cert_k))

    actual_bias = _tree_sub(full_grads, stopped_grads)
    certified_bias = (cert_q, cert_k, jnp.zeros_like(v))
    certificate_error = _tree_sub(actual_bias, certified_bias)
    return {
        "refine_steps": int(refine_steps),
        "full_loss": float(full_value),
        "stopped_loss": float(stopped_value),
        "eta_l2": _tree_l2((eta_u, eta_v)),
        "eta_max_abs": _tree_max_abs((eta_u, eta_v)),
        "bias_l2": _tree_l2(actual_bias),
        "bias_max_abs": _tree_max_abs(actual_bias),
        "bias_q_max_abs": float(jnp.max(jnp.abs(actual_bias[0]))),
        "bias_k_max_abs": float(jnp.max(jnp.abs(actual_bias[1]))),
        "bias_v_max_abs": float(jnp.max(jnp.abs(actual_bias[2]))),
        "certificate_error_l2": _tree_l2(certificate_error),
        "certificate_error_max_abs": _tree_max_abs(certificate_error),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--n_iters", type=int, default=15)
    parser.add_argument("--band_width", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--refine_steps", type=str, default="0,1,2,4")
    parser.add_argument("--cert_tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    seeds = _parse_int_list(args.seeds)
    refine_steps_list = _parse_int_list(args.refine_steps)
    rows = []
    for seed in seeds:
        q, k, v, target, q_mask, k_mask = _make_inputs(seed, args.length, args.dim)
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
        for previous, current in zip(cases, cases[1:]):
            denom = max(previous["eta_l2"], 1e-30)
            current["eta_l2_ratio_from_previous"] = current["eta_l2"] / denom
        rows.append({"seed": int(seed), "cases": cases})

    by_r = {}
    for refine_steps in refine_steps_list:
        matching = [case for row in rows for case in row["cases"] if case["refine_steps"] == refine_steps]
        by_r[str(refine_steps)] = {
            "eta_l2_max": max(case["eta_l2"] for case in matching),
            "bias_max_abs_max": max(case["bias_max_abs"] for case in matching),
            "certificate_error_max_abs_max": max(case["certificate_error_max_abs"] for case in matching),
        }

    selections = []
    for row in rows:
        selected = None
        for case in row["cases"]:
            if case["bias_max_abs"] <= args.cert_tolerance:
                selected = {
                    "refine_steps": case["refine_steps"],
                    "certified_bias_max_abs": case["bias_max_abs"],
                }
                break
        if selected is None:
            last = row["cases"][-1]
            selected = {
                "refine_steps": None,
                "certified_bias_max_abs": last["bias_max_abs"],
            }
        selections.append({"seed": row["seed"], **selected})

    report = {
        "settings": {
            "seeds": seeds,
            "length": args.length,
            "dim": args.dim,
            "n_iters": args.n_iters,
            "band_width": args.band_width,
            "epsilon": args.epsilon,
            "refine_steps": refine_steps_list,
            "cert_tolerance": args.cert_tolerance,
        },
        "summary_by_refine_steps": by_r,
        "certified_tail_depth_selection": selections,
        "rows": rows,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
