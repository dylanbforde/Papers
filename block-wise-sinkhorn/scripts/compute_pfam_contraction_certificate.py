#!/usr/bin/env python3

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _iter_examples(data_dir, max_examples):
    count = 0
    for shard_path in sorted(Path(data_dir).glob("shard_*.npz")):
        with np.load(shard_path, allow_pickle=False) as shard:
            emb1 = shard["emb1"].astype(np.float32)
            emb2 = shard["emb2"].astype(np.float32)
        for idx in range(emb1.shape[0]):
            yield shard_path.name, idx, emb1[idx], emb2[idx]
            count += 1
            if max_examples is not None and count >= max_examples:
                return


def _project(params, emb1, emb2):
    q = emb1 @ params["W_q"].astype(np.float32)
    k = emb2 @ params["W_k"].astype(np.float32)
    return q, k


def _projective_delta(score_block):
    rows = score_block.shape[0]
    if rows <= 1 or score_block.shape[1] <= 1:
        return 0.0
    delta = 0.0
    for row in range(rows):
        diff = score_block[row][None, :] - score_block
        ranges = diff.max(axis=1) - diff.min(axis=1)
        delta = max(delta, float(ranges.max()))
    return delta


def _block_ranges(q, k, q_mask, k_mask, *, block_size, band_width, epsilon):
    length = q.shape[0]
    raw = (q @ k.T) / np.sqrt(q.shape[-1]).astype(np.float32)
    for start in range(0, length, block_size):
        stop = min(start + block_size, length)
        row_idx = np.where(q_mask[start:stop])[0] + start
        if row_idx.size == 0:
            continue
        if band_width is None or int(band_width) >= length:
            col_lo, col_hi = 0, length
        else:
            col_lo = max(0, start - int(band_width))
            col_hi = min(length, stop + int(band_width))
        col_candidates = np.arange(col_lo, col_hi)
        col_idx = col_candidates[k_mask[col_candidates]]
        if col_idx.size == 0:
            continue
        score_block = raw[np.ix_(row_idx, col_idx)] / float(epsilon)
        omega = float(score_block.max() - score_block.min())
        delta = _projective_delta(score_block)
        yield {
            "rows": int(row_idx.size),
            "cols": int(col_idx.size),
            "omega_scaled": omega,
            "delta_projective": delta,
            "rho_h": float(np.tanh(delta / 4.0) ** 2),
            "rho_h_range_bound": float(np.tanh(omega / 2.0) ** 2),
        }


def _summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _checkpoint_report(path, args):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    params = payload["params"]
    block_reports = []
    examples = 0
    for _shard, _idx, emb1, emb2 in _iter_examples(args.data_dir, args.max_examples):
        q, k = _project(params, emb1, emb2)
        q_mask = np.any(emb1 != 0, axis=-1)
        k_mask = np.any(emb2 != 0, axis=-1)
        block_reports.extend(
            _block_ranges(
                q,
                k,
                q_mask,
                k_mask,
                block_size=args.block_size,
                band_width=args.band_width,
                epsilon=args.epsilon,
            )
        )
        examples += 1
    if not block_reports:
        raise RuntimeError(f"No active blocks found for {path}.")
    return {
        "checkpoint": str(path),
        "step": int(payload["step"]),
        "examples": examples,
        "active_blocks": len(block_reports),
        "rho_h": _summarize([item["rho_h"] for item in block_reports]),
        "rho_h_range_bound": _summarize([item["rho_h_range_bound"] for item in block_reports]),
        "omega_scaled": _summarize([item["omega_scaled"] for item in block_reports]),
        "delta_projective": _summarize([item["delta_projective"] for item in block_reports]),
        "rows_per_block": _summarize([item["rows"] for item in block_reports]),
        "cols_per_block": _summarize([item["cols"] for item in block_reports]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", action="append", required=True)
    parser.add_argument("--data_dir", default="data/pfam_shards_test")
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--band_width", type=int, default=1024)
    parser.add_argument("--epsilon", type=float, default=1.0)
    args = parser.parse_args()
    reports = [_checkpoint_report(Path(path), args) for path in args.ckpt]
    print(json.dumps({"settings": vars(args), "reports": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
