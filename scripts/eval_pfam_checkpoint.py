#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks import benchmark_pfam_training as pfam


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
    terms = pfam._alignment_loss_terms(
        method,
        Q,
        K,
        V,
        target_cols,
        target_mask,
        q_mask,
        k_mask,
        n_iters,
        band_width_eff,
        sinkhorn_gradient_mode,
        sinkhorn_refine_steps,
        sinkhorn_refine_tolerance,
        sinkhorn_epsilon,
        sinkhorn_epsilon_scaling_steps,
        sinkhorn_epsilon_scaling_factor,
        support_reconstruction,
    )
    jax.block_until_ready(terms)
    return {k: float(v) for k, v in terms.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["sinkhorn_ot", "flash_attention"], required=True)
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
    args = parser.parse_args()

    params, step = _load_params(args.ckpt_path)
    shard_results = []
    start = time.perf_counter()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    for shard in _iter_shards(args.data_dir):
        minibatch_results = []
        for batch in _iter_minibatches(shard, args.batch_size):
            metrics = _evaluate_batch(
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
            )
            minibatch_results.append((int(batch["emb1"].shape[0]), metrics))
        examples = sum(size for size, _ in minibatch_results)
        shard_results.append(
            {
                "shard": shard["name"],
                "examples": examples,
                "reconstruction": float(
                    sum(size * metrics["reconstruction"] for size, metrics in minibatch_results) / max(examples, 1)
                ),
                "sparse_ce": float(
                    sum(size * metrics["sparse_ce"] for size, metrics in minibatch_results) / max(examples, 1)
                ),
                "monotonicity": float(
                    sum(size * metrics["monotonicity"] for size, metrics in minibatch_results) / max(examples, 1)
                ),
            }
        )

    totals = {
        "examples": sum(item["examples"] for item in shard_results),
        "reconstruction": float(np.mean([item["reconstruction"] for item in shard_results])),
        "sparse_ce": float(np.mean([item["sparse_ce"] for item in shard_results])),
        "monotonicity": float(np.mean([item["monotonicity"] for item in shard_results])),
        "elapsed_sec": float(time.perf_counter() - start),
    }
    result = {
        "method": args.method,
        "ckpt_path": args.ckpt_path,
        "checkpoint_step": step,
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
        },
        "aggregate": totals,
        "shards": shard_results,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
