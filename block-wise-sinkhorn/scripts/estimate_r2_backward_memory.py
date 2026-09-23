#!/usr/bin/env python3
"""Analytic memory ledger for the R=2 one-reference backward schedule."""

import argparse
import json


def _band_entries(length, half_band_width):
    if half_band_width is None or half_band_width >= length:
        return length * length
    width = int(half_band_width)
    return length * (2 * width + 1) - width * (width + 1)


def _mib(num_bytes):
    return num_bytes / (1024.0 * 1024.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument("--band_width", type=int, default=1024)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--base_steps", type=int, default=15)
    parser.add_argument("--refine_steps", type=int, default=2)
    parser.add_argument("--dtype_bytes", type=int, default=4)
    args = parser.parse_args()

    active_entries = _band_entries(args.length, args.band_width)
    plan_bytes = active_entries * args.dtype_bytes
    tile_bytes = args.block_size * args.block_size * args.dtype_bytes
    tail_vectors = 2 * (args.refine_steps + 1) * args.length * args.dtype_bytes
    base_tail_history = 2 * (args.base_steps + args.refine_steps + 1) * args.length * args.dtype_bytes
    qkv_bytes = 3 * args.length * args.head_dim * args.dtype_bytes
    qkv_dout_bytes = 4 * args.length * args.head_dim * args.dtype_bytes

    result = {
        "settings": vars(args),
        "active_support_entries_per_head": active_entries,
        "logical_materialized_plan_factors_mib": {
            "direct_four_plan": _mib(4 * plan_bytes),
            "one_reference": _mib(plan_bytes),
            "saved_by_one_reference": _mib(3 * plan_bytes),
        },
        "resident_plan_tile_mib": {
            "direct_four_plan": _mib(4 * tile_bytes),
            "one_reference": _mib(tile_bytes),
            "saved_by_one_reference": _mib(3 * tile_bytes),
        },
        "vector_state_mib": {
            "r2_tail_uv_only": _mib(tail_vectors),
            "base_plus_tail_uv_history_upper_ledger": _mib(base_tail_history),
        },
        "activation_storage_mib": {
            "qkv_inputs": _mib(qkv_bytes),
            "qkv_plus_output_cotangent": _mib(qkv_dout_bytes),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
