#!/usr/bin/env python3
# ruff: noqa: E402

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

np = None
jax = None
jnp = None
pfam = None


def _load_runtime_deps():
    global np, jax, jnp, pfam
    if np is not None and jax is not None and jnp is not None and pfam is not None:
        return
    import numpy as _np
    import jax as _jax
    import jax.numpy as _jnp

    try:
        from src.benchmarks import benchmark_pfam_training as _pfam
    except ModuleNotFoundError:
        import train_sinkhorn_pfam as _pfam

    if hasattr(_pfam, "_load_runtime_deps"):
        _pfam._load_runtime_deps()

    np = _np
    jax = _jax
    jnp = _jnp
    pfam = _pfam


METRIC_KEYS = (
    "reconstruction",
    "sparse_ce",
    "monotonicity",
    "barycenter_mae",
    "barycenter_within_1",
    "barycenter_within_5",
)

DIAGONAL_LOG_FLOOR = float(math.log(1e-12))


def _iter_shards(data_dir):
    shard_paths = sorted(Path(data_dir).glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.npz files found in {data_dir!r}.")
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            emb1 = shard["emb1"]
            emb2 = shard["emb2"]
            target = shard["target"]
        target_mask = np.any(target > 0, axis=-1)
        target_cols = target.argmax(axis=-1).astype(np.int32)
        target_cols = np.where(target_mask, target_cols, 0)
        q_mask = np.any(emb1 != 0, axis=-1)
        k_mask = np.any(emb2 != 0, axis=-1)
        target_mask = target_mask & q_mask
        yield {
            "emb1": emb1[:, None, :, :],
            "emb2": emb2[:, None, :, :],
            "target_cols": target_cols,
            "target_mask": target_mask,
            "q_mask": q_mask,
            "k_mask": k_mask,
            "name": shard_path.name,
        }


def _iter_minibatches(shard_batch, batch_size):
    total = int(shard_batch["emb1"].shape[0])
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        yield {
            "emb1": shard_batch["emb1"][start:stop],
            "emb2": shard_batch["emb2"][start:stop],
            "target_cols": shard_batch["target_cols"][start:stop],
            "target_mask": shard_batch["target_mask"][start:stop],
            "q_mask": shard_batch["q_mask"][start:stop],
            "k_mask": shard_batch["k_mask"][start:stop],
            "name": f"{shard_batch['name']}[{start}:{stop}]",
        }


def _load_params(ckpt_path):
    with open(ckpt_path, "rb") as f:
        payload = pickle.load(f)
    return payload["params"], int(payload["step"])


def _safe_div(num, denom):
    return num / jnp.maximum(denom, 1.0)


def _length_scaled_diagonal_cols(q_mask, k_mask):
    """Map active query ordinals to active key ordinals with a monotone linear rule."""
    q_mask = jnp.asarray(q_mask, dtype=bool)
    k_mask = jnp.asarray(k_mask, dtype=bool)
    q_count = jnp.sum(q_mask, axis=-1).astype(jnp.int32)
    k_count = jnp.sum(k_mask, axis=-1).astype(jnp.int32)
    q_rank = jnp.cumsum(q_mask.astype(jnp.int32), axis=-1) - 1

    q_denom = jnp.maximum(q_count - 1, 1)
    k_last = jnp.maximum(k_count - 1, 0)
    diagonal_scale = (k_last / q_denom).astype(jnp.float32)
    pred_ord = jnp.rint(q_rank.astype(jnp.float32) * diagonal_scale[:, None])
    pred_ord = pred_ord.astype(jnp.int32)
    pred_ord = jnp.clip(pred_ord, 0, k_last[:, None])

    key_positions = jnp.arange(k_mask.shape[-1], dtype=jnp.int32)[None, :]
    sentinel = jnp.full_like(key_positions, k_mask.shape[-1])
    sorted_key_positions = jnp.sort(jnp.where(k_mask, key_positions, sentinel), axis=-1)
    pred_cols = jnp.take_along_axis(sorted_key_positions, pred_ord, axis=-1)
    return jnp.where((q_mask & (k_count[:, None] > 0)), pred_cols, 0)


def _length_band_mask(length, band_width):
    if band_width is None or int(band_width) >= length:
        return None
    row_idx = jnp.arange(length, dtype=jnp.int32)[:, None]
    col_idx = jnp.arange(length, dtype=jnp.int32)[None, :]
    return jnp.abs(row_idx - col_idx) <= int(band_width)


def _flash_attention_outputs(Q, K, V, q_mask, k_mask, band_width=None):
    scale = 1.0 / jnp.sqrt(Q.shape[-1])
    logits = jnp.einsum("bhmd,bhnd->bhmn", Q, K) * scale
    logits = jnp.where(k_mask[:, None, None, :], logits, -jnp.inf)
    band = _length_band_mask(Q.shape[-2], band_width)
    if band is not None:
        logits = jnp.where(band[None, None, :, :], logits, -jnp.inf)
    probs = jax.nn.softmax(logits, axis=-1)
    outputs = jnp.einsum("bhmn,bhnd->bhmd", probs, V)
    return jnp.where(q_mask[:, None, :, None], outputs, 0.0)


def _flash_alignment_barycenters(Q, K, q_mask, k_mask, band_width):
    scale = 1.0 / jnp.sqrt(Q.shape[-1])
    logits = jnp.einsum("bhmd,bhnd->bhmn", Q, K) * scale
    logits = jnp.where(k_mask[:, None, None, :], logits, -jnp.inf)
    band = _length_band_mask(Q.shape[-2], band_width)
    if band is not None:
        logits = jnp.where(band[None, None, :, :], logits, -jnp.inf)
    probs = jax.nn.softmax(logits, axis=-1)
    col_idx = jnp.arange(K.shape[-2], dtype=Q.dtype)
    bary = jnp.sum(probs * col_idx[None, None, None, :], axis=-1)
    return jnp.where(q_mask[:, None, :], bary, 0.0)


def _metric_values(parts):
    return {key: float(_safe_div(parts[key]["num"], parts[key]["denom"])) for key in METRIC_KEYS}


def _empty_parts():
    return {key: {"num": 0.0, "denom": 0.0} for key in METRIC_KEYS}


def _add_parts(total, batch_parts):
    for key in METRIC_KEYS:
        total[key]["num"] += float(batch_parts[key]["num"])
        total[key]["denom"] += float(batch_parts[key]["denom"])


def _bootstrap_ci(example_parts, *, samples, seed):
    if samples <= 0 or not example_parts:
        return None
    rng = np.random.default_rng(seed)
    count = len(example_parts)
    intervals = {}
    for key in METRIC_KEYS:
        values = []
        nums = np.asarray([item[key]["num"] for item in example_parts], dtype=np.float64)
        denoms = np.asarray([item[key]["denom"] for item in example_parts], dtype=np.float64)
        for _ in range(samples):
            idx = rng.integers(0, count, size=count)
            denom = denoms[idx].sum()
            value = nums[idx].sum() / max(denom, 1.0)
            values.append(value)
        lo, hi = np.percentile(values, [2.5, 97.5])
        intervals[key] = {
            "lower": float(lo),
            "upper": float(hi),
            "samples": int(samples),
            "unit": "example",
        }
    return intervals


def _metric_parts(
    method,
    Q,
    K,
    V,
    target_cols,
    target_mask,
    q_mask,
    k_mask,
    *,
    n_iters,
    band_width,
    sinkhorn_gradient_mode,
    sinkhorn_refine_steps,
    sinkhorn_refine_tolerance,
    sinkhorn_epsilon,
    sinkhorn_epsilon_scaling_steps,
    sinkhorn_epsilon_scaling_factor,
    support_reconstruction,
    per_example=False,
):
    safe_cols, active_mask = pfam._prepare_sparse_supervision(target_cols, target_mask, q_mask, k_mask)
    target_v = pfam._batch_take_rows(V, safe_cols)

    if method == "linear_diagonal":
        pred_cols = _length_scaled_diagonal_cols(q_mask, k_mask)
        outputs = pfam._batch_take_rows(V, pred_cols)
        matches = safe_cols[:, None, :] == pred_cols[:, None, :]
        log_pos = jnp.where(matches, 0.0, DIAGONAL_LOG_FLOOR)
        barycenters = pred_cols[:, None, :].astype(Q.dtype)
    elif method == "sinkhorn_ot":
        outputs = pfam.sinkhorn_outputs(
            Q,
            K,
            V,
            q_mask=q_mask,
            k_mask=k_mask,
            n_iters=n_iters,
            band_width=band_width,
            gradient_mode=sinkhorn_gradient_mode,
            refine_steps=sinkhorn_refine_steps,
            refine_tolerance=sinkhorn_refine_tolerance,
            epsilon=sinkhorn_epsilon,
            epsilon_scaling_steps=sinkhorn_epsilon_scaling_steps,
            epsilon_scaling_factor=sinkhorn_epsilon_scaling_factor,
            support_reconstruction=support_reconstruction,
        )
        u, v = pfam.ot_kernel.compute_global_uv(
            Q,
            K,
            n_iters=n_iters,
            band_width=band_width,
            q_mask=q_mask,
            k_mask=k_mask,
            epsilon=sinkhorn_epsilon,
            epsilon_scaling_steps=sinkhorn_epsilon_scaling_steps,
            epsilon_scaling_factor=sinkhorn_epsilon_scaling_factor,
            support_reconstruction=support_reconstruction,
        )
        scale = 1.0 / (jnp.sqrt(Q.shape[-1]) * sinkhorn_epsilon)
        matched_k = pfam._batch_take_rows(K, safe_cols)
        matched_v = pfam._batch_take_last(v, safe_cols)
        log_pos = jnp.sum(Q * matched_k, axis=-1) * scale + u + matched_v
        if band_width is not None:
            band = pfam.ot_kernel.materialize_support_mask(
                Q,
                K,
                band_width=band_width,
                q_mask=q_mask,
                k_mask=k_mask,
                support_reconstruction=support_reconstruction,
            )
            in_band = jnp.take_along_axis(
                jnp.broadcast_to(band[None, :, :], (Q.shape[0],) + band.shape),
                safe_cols[..., None],
                axis=-1,
            )[..., 0]
            log_pos = jnp.where(in_band[:, None, :], log_pos, -1e9)
        barycenters = pfam.sinkhorn_alignment_barycenters(
            Q,
            K,
            q_mask,
            k_mask,
            n_iters,
            band_width,
            sinkhorn_epsilon,
            sinkhorn_epsilon_scaling_steps,
            sinkhorn_epsilon_scaling_factor,
            support_reconstruction,
        )
    else:
        outputs = _flash_attention_outputs(Q, K, V, q_mask, k_mask, band_width=band_width)
        scale = 1.0 / jnp.sqrt(Q.shape[-1])
        logits = jnp.einsum("bhmd,bhnd->bhmn", Q, K) * scale
        logits = jnp.where(k_mask[:, None, None, :], logits, -jnp.inf)
        band = _length_band_mask(Q.shape[-2], band_width)
        if band is not None:
            logits = jnp.where(band[None, None, :, :], logits, -jnp.inf)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        log_pos = pfam._rowwise_take_cols(log_probs, safe_cols)
        barycenters = _flash_alignment_barycenters(Q, K, q_mask, k_mask, band_width)

    sq_error = jnp.sum(jnp.square(outputs - target_v), axis=-1)
    active = active_mask[:, None, :]
    rec_num_examples = jnp.sum(jnp.where(active, sq_error, 0.0), axis=(1, 2))
    ce_num_examples = -jnp.sum(jnp.where(active, log_pos, 0.0), axis=(1, 2))
    active_denom_examples = jnp.maximum(jnp.sum(active_mask, axis=-1) * Q.shape[1], 1)

    pair_mask = target_mask[:, 1:] & target_mask[:, :-1]
    deltas = barycenters[..., 1:] - barycenters[..., :-1]
    violation = jnp.maximum(-deltas, 0.0)
    pair = pair_mask[:, None, :]
    mono_num_examples = jnp.sum(jnp.where(pair, violation, 0.0), axis=(1, 2))
    mono_denom_examples = jnp.maximum(jnp.sum(pair_mask, axis=-1) * barycenters.shape[1], 1)

    target_cols_f = safe_cols.astype(barycenters.dtype)
    bary_err = jnp.abs(barycenters - target_cols_f[:, None, :])
    bary_num_examples = jnp.sum(jnp.where(active, bary_err, 0.0), axis=(1, 2))
    within_1_examples = jnp.sum(jnp.where(active & (bary_err <= 1.0), 1.0, 0.0), axis=(1, 2))
    within_5_examples = jnp.sum(jnp.where(active & (bary_err <= 5.0), 1.0, 0.0), axis=(1, 2))

    example_arrays = {
        "reconstruction": {"num": rec_num_examples, "denom": active_denom_examples},
        "sparse_ce": {"num": ce_num_examples, "denom": active_denom_examples},
        "monotonicity": {"num": mono_num_examples, "denom": mono_denom_examples},
        "barycenter_mae": {"num": bary_num_examples, "denom": active_denom_examples},
        "barycenter_within_1": {"num": within_1_examples, "denom": active_denom_examples},
        "barycenter_within_5": {"num": within_5_examples, "denom": active_denom_examples},
    }
    parts = {
        key: {
            "num": jnp.sum(value["num"]),
            "denom": jnp.sum(value["denom"]),
        }
        for key, value in example_arrays.items()
    }
    if not per_example:
        return parts
    batch_size = int(Q.shape[0])
    examples = []
    for idx in range(batch_size):
        examples.append(
            {
                key: {
                    "num": float(example_arrays[key]["num"][idx]),
                    "denom": float(example_arrays[key]["denom"][idx]),
                }
                for key in METRIC_KEYS
            }
        )
    return parts, examples


def _evaluate_batch(
    params,
    batch,
    *,
    method,
    n_iters,
    band_width,
    sinkhorn_gradient_mode,
    sinkhorn_refine_steps,
    sinkhorn_refine_tolerance,
    sinkhorn_epsilon,
    sinkhorn_epsilon_scaling_steps,
    sinkhorn_epsilon_scaling_factor,
    support_reconstruction,
    gap_mode,
    collect_examples=False,
):
    emb_q = jnp.asarray(batch["emb1"], dtype=jnp.float32)
    emb_k = jnp.asarray(batch["emb2"], dtype=jnp.float32)
    target_cols = jnp.asarray(batch["target_cols"], dtype=jnp.int32)
    target_mask = jnp.asarray(batch["target_mask"], dtype=bool)
    q_mask = jnp.asarray(batch["q_mask"], dtype=bool)
    k_mask = jnp.asarray(batch["k_mask"], dtype=bool)

    Q, K, V = pfam._project_qkv(params, emb_q, emb_k)
    Q, K, V, target_cols, target_mask, q_mask, k_mask = pfam._apply_gap_mode(
        params,
        Q,
        K,
        V,
        target_cols,
        target_mask,
        q_mask,
        k_mask,
        gap_mode,
    )
    band_width_eff = pfam._effective_band_width(
        band_width, gap_mode, int(target_cols.shape[-1] - pfam.DUSTBIN_BLOCK_SIZE)
    )
    parts = _metric_parts(
        method,
        Q,
        K,
        V,
        target_cols,
        target_mask,
        q_mask,
        k_mask,
        n_iters=n_iters,
        band_width=band_width_eff,
        sinkhorn_gradient_mode=sinkhorn_gradient_mode,
        sinkhorn_refine_steps=sinkhorn_refine_steps,
        sinkhorn_refine_tolerance=sinkhorn_refine_tolerance,
        sinkhorn_epsilon=sinkhorn_epsilon,
        sinkhorn_epsilon_scaling_steps=sinkhorn_epsilon_scaling_steps,
        sinkhorn_epsilon_scaling_factor=sinkhorn_epsilon_scaling_factor,
        support_reconstruction=support_reconstruction,
        per_example=collect_examples,
    )
    jax.block_until_ready(parts)
    if collect_examples:
        parts, examples = parts
    else:
        examples = []
    return {
        "parts": {
            key: {
                "num": float(value["num"]),
                "denom": float(value["denom"]),
            }
            for key, value in parts.items()
        },
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["sinkhorn_ot", "flash_attention", "linear_diagonal"], required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--data_dir", default="data/pfam_shards_test")
    parser.add_argument("--band_width", type=int, default=1024)
    parser.add_argument("--n_iters", type=int, default=15)
    parser.add_argument("--sinkhorn_gradient_mode", default="tail_refinement")
    parser.add_argument("--sinkhorn_refine_steps", type=int, default=2)
    parser.add_argument("--sinkhorn_refine_tolerance", type=float, default=None)
    parser.add_argument("--sinkhorn_epsilon", type=float, default=1.0)
    parser.add_argument("--sinkhorn_epsilon_scaling_steps", type=int, default=0)
    parser.add_argument("--sinkhorn_epsilon_scaling_factor", type=float, default=2.0)
    parser.add_argument("--support_reconstruction", default="none")
    parser.add_argument("--gap_mode", choices=["none", "dustbin_block"], default="none")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--bootstrap_samples", type=int, default=0)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, help="Optional training/evaluation seed metadata to include in the JSON.")
    parser.add_argument("--run_label", default=None, help="Optional human-readable run label metadata to include in the JSON.")
    args = parser.parse_args()

    _load_runtime_deps()
    params, step = _load_params(args.ckpt_path)
    shard_results = []
    example_parts = []
    start = time.perf_counter()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    for shard in _iter_shards(args.data_dir):
        shard_parts = _empty_parts()
        examples = 0
        for batch in _iter_minibatches(shard, args.batch_size):
            batch_parts = _evaluate_batch(
                params,
                batch,
                method=args.method,
                n_iters=args.n_iters,
                band_width=args.band_width,
                sinkhorn_gradient_mode=args.sinkhorn_gradient_mode,
                sinkhorn_refine_steps=args.sinkhorn_refine_steps,
                sinkhorn_refine_tolerance=args.sinkhorn_refine_tolerance,
                sinkhorn_epsilon=args.sinkhorn_epsilon,
                sinkhorn_epsilon_scaling_steps=args.sinkhorn_epsilon_scaling_steps,
                sinkhorn_epsilon_scaling_factor=args.sinkhorn_epsilon_scaling_factor,
                support_reconstruction=args.support_reconstruction,
                gap_mode=args.gap_mode,
                collect_examples=args.bootstrap_samples > 0,
            )
            examples += int(batch["emb1"].shape[0])
            _add_parts(shard_parts, batch_parts["parts"])
            example_parts.extend(batch_parts["examples"])
        shard_results.append(
            {
                "shard": shard["name"],
                "examples": examples,
                **_metric_values(shard_parts),
                "metric_denominators": {
                    key: shard_parts[key]["denom"]
                    for key in METRIC_KEYS
                },
            }
        )

    total_parts = _empty_parts()
    for shard_result in shard_results:
        for key in METRIC_KEYS:
            total_parts[key]["num"] += shard_result[key] * shard_result["metric_denominators"][key]
            total_parts[key]["denom"] += shard_result["metric_denominators"][key]
    totals = {
        "examples": sum(item["examples"] for item in shard_results),
        **_metric_values(total_parts),
        "metric_denominators": {
            key: total_parts[key]["denom"]
            for key in METRIC_KEYS
        },
        "elapsed_sec": float(time.perf_counter() - start),
    }
    bootstrap = _bootstrap_ci(example_parts, samples=args.bootstrap_samples, seed=args.bootstrap_seed)
    if bootstrap is not None:
        totals["bootstrap_95ci"] = bootstrap
    result = {
        "method": args.method,
        "ckpt_path": args.ckpt_path,
        "checkpoint_step": step,
        "seed": args.seed,
        "run_label": args.run_label,
        "settings": {
            "data_dir": args.data_dir,
            "band_width": args.band_width,
            "n_iters": args.n_iters,
            "sinkhorn_gradient_mode": args.sinkhorn_gradient_mode,
            "sinkhorn_refine_steps": args.sinkhorn_refine_steps,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "support_reconstruction": args.support_reconstruction,
            "gap_mode": args.gap_mode,
            "batch_size": args.batch_size,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "seed": args.seed,
            "run_label": args.run_label,
            "linear_diagonal_log_floor": DIAGONAL_LOG_FLOOR,
        },
        "aggregate": totals,
        "shards": shard_results,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
