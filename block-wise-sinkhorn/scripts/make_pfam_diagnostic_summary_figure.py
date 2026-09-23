#!/usr/bin/env python3
"""Render compact Pfam screen and contraction diagnostic summaries."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


SCREEN_ROWS = [
    {"config": "k2_eps1", "loss": 3.5778, "ce": 5.3223, "mono": 0.3353, "throughput": 8.46},
    {"config": "k4_eps1", "loss": 3.5779, "ce": 5.3223, "mono": 0.3353, "throughput": 8.38},
    {"config": "dustbin_mono", "loss": 3.9030, "ce": 5.3258, "mono": 0.4155, "throughput": 8.41},
    {"config": "dustbin_hybrid", "loss": 4.6250, "ce": 5.3244, "mono": 0.4162, "throughput": 8.45},
]


def _iter_examples(data_dir: Path, max_examples: int | None):
    count = 0
    for shard_path in sorted(data_dir.glob("shard_*.npz")):
        with np.load(shard_path, allow_pickle=False) as shard:
            emb1 = shard["emb1"].astype(np.float32)
            emb2 = shard["emb2"].astype(np.float32)
        for idx in range(emb1.shape[0]):
            yield shard_path.name, idx, emb1[idx], emb2[idx]
            count += 1
            if max_examples is not None and count >= max_examples:
                return


def _load_checkpoint(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    return payload["params"], int(payload["step"])


def _project(params: dict, emb1: np.ndarray, emb2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = emb1 @ params["W_q"].astype(np.float32)
    k = emb2 @ params["W_k"].astype(np.float32)
    return q, k


def _projective_delta(score_block: np.ndarray) -> float:
    if score_block.shape[0] <= 1 or score_block.shape[1] <= 1:
        return 0.0
    delta = 0.0
    for row in range(score_block.shape[0]):
        diff = score_block[row][None, :] - score_block
        ranges = diff.max(axis=1) - diff.min(axis=1)
        delta = max(delta, float(ranges.max()))
    return delta


def _block_reports(
    q: np.ndarray,
    k: np.ndarray,
    q_mask: np.ndarray,
    k_mask: np.ndarray,
    *,
    block_size: int,
    band_width: int,
    epsilon: float,
) -> list[dict]:
    reports = []
    length_q = q.shape[0]
    length_k = k.shape[0]
    raw = (q @ k.T) / np.float32(np.sqrt(q.shape[-1]))
    for start in range(0, length_q, block_size):
        stop = min(start + block_size, length_q)
        row_idx = np.where(q_mask[start:stop])[0] + start
        if row_idx.size == 0:
            continue
        if band_width >= max(length_q, length_k):
            col_lo, col_hi = 0, length_k
        else:
            col_lo = max(0, start - band_width)
            col_hi = min(length_k, stop + band_width)
        col_candidates = np.arange(col_lo, col_hi)
        col_idx = col_candidates[k_mask[col_candidates]]
        if col_idx.size == 0:
            continue
        score_block = raw[np.ix_(row_idx, col_idx)] / float(epsilon)
        omega = float(score_block.max() - score_block.min())
        delta = _projective_delta(score_block)
        reports.append(
            {
                "rows": int(row_idx.size),
                "cols": int(col_idx.size),
                "omega_scaled": omega,
                "delta_projective": delta,
                "rho_h": float(np.tanh(delta / 4.0) ** 2),
                "rho_h_range_bound": float(np.tanh(omega / 2.0) ** 2),
            }
        )
    return reports


def _summarize(values: np.ndarray) -> dict:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _checkpoint_contraction(path: Path, args: argparse.Namespace) -> dict:
    params, step = _load_checkpoint(path)
    reports = []
    examples = 0
    for _shard, _idx, emb1, emb2 in _iter_examples(args.data_dir, args.max_examples):
        q, k = _project(params, emb1, emb2)
        reports.extend(
            _block_reports(
                q,
                k,
                np.any(emb1 != 0, axis=-1),
                np.any(emb2 != 0, axis=-1),
                block_size=args.block_size,
                band_width=args.band_width,
                epsilon=args.epsilon,
            )
        )
        examples += 1
    if not reports:
        raise RuntimeError(f"No active contraction blocks found for {path}.")
    rho_h = np.asarray([item["rho_h"] for item in reports], dtype=np.float64)
    rho_range = np.asarray([item["rho_h_range_bound"] for item in reports], dtype=np.float64)
    return {
        "checkpoint": str(path),
        "step": step,
        "examples": examples,
        "active_blocks": len(reports),
        "rho_h_values": rho_h,
        "rho_h_range_values": rho_range,
        "rho_h": _summarize(rho_h),
        "rho_h_range_bound": _summarize(rho_range),
    }


def _screen_score_matrix() -> tuple[np.ndarray, list[str], list[str]]:
    metrics = ["loss", "ce", "mono", "throughput"]
    labels = [row["config"] for row in SCREEN_ROWS]
    values = np.asarray([[row[metric] for metric in metrics] for row in SCREEN_ROWS], dtype=np.float64)
    scores = np.empty_like(values)
    for col, metric in enumerate(metrics):
        column = values[:, col]
        if metric == "throughput":
            best = column.max()
            worst = column.min()
            scores[:, col] = (best - column) / max(best - worst, 1e-12)
        else:
            best = column.min()
            worst = column.max()
            scores[:, col] = (column - best) / max(worst - best, 1e-12)
    return scores, labels, metrics


def _render(reports: list[dict], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.65), constrained_layout=True)

    scores, labels, metrics = _screen_score_matrix()
    ax = axes[0]
    im = ax.imshow(scores, aspect="auto", cmap="viridis_r", vmin=0.0, vmax=1.0)
    ax.set_title("Step-50 screen snapshot")
    ax.set_xticks(np.arange(len(metrics)), ["loss", "CE", "mono", "ex/s"])
    ax.set_yticks(np.arange(len(labels)), labels)
    for row, screen_row in enumerate(SCREEN_ROWS):
        values = [screen_row["loss"], screen_row["ce"], screen_row["mono"], screen_row["throughput"]]
        for col, value in enumerate(values):
            ax.text(col, row, f"{value:.3g}", ha="center", va="center", color="white" if scores[row, col] > 0.45 else "black", fontsize=6.4)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("relative column rank")

    ax = axes[1]
    max_rho = max(float(np.max(report["rho_h_values"])) for report in reports)
    bins = np.linspace(0.0, max(0.5, max_rho), 28)
    colors = ["#6b7280", "#0f766e"]
    for report, color in zip(reports, colors):
        ax.hist(
            report["rho_h_values"],
            bins=bins,
            histtype="step",
            linewidth=1.6,
            color=color,
            label=f"step {report['step']}",
        )
        ax.axvline(report["rho_h"]["median"], color=color, linestyle="--", linewidth=1.0)
    ax.set_title(r"Projective contraction distribution")
    ax.set_xlabel(r"$\rho_H$ over active blocks")
    ax.set_ylabel("block count")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data/pfam_shards_test"))
    parser.add_argument("--initial_ckpt", type=Path, default=Path("checkpoints/sinkhorn_ot/ckpt.pkl"))
    parser.add_argument("--final_ckpt", type=Path, default=Path("checkpoints/sinkhorn_ot/promoted_step1437_ckpt.pkl"))
    parser.add_argument("--output", type=Path, default=Path("paper_neurips/figures/pfam_diagnostic_summary.pdf"))
    parser.add_argument("--metadata", type=Path, default=Path("paper_neurips/figures/pfam_diagnostic_summary.json"))
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--band_width", type=int, default=1024)
    parser.add_argument("--epsilon", type=float, default=1.0)
    args = parser.parse_args()

    global plt, np  # noqa: PLW0603
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    reports = [_checkpoint_contraction(args.initial_ckpt, args), _checkpoint_contraction(args.final_ckpt, args)]
    _render(reports, args.output)

    serializable_reports = []
    for report in reports:
        item = {key: value for key, value in report.items() if not key.endswith("_values")}
        item["rho_h_histogram_values"] = [float(value) for value in report["rho_h_values"]]
        serializable_reports.append(item)
    payload = {
        "figure": str(args.output),
        "settings": {
            "data_dir": str(args.data_dir),
            "max_examples": args.max_examples,
            "block_size": args.block_size,
            "band_width": args.band_width,
            "epsilon": args.epsilon,
        },
        "screen_rows": SCREEN_ROWS,
        "contraction_reports": serializable_reports,
        "notes": [
            "The screen heatmap normalizes each metric column so the best row has score 0 and the worst has score 1.",
            "The contraction histogram uses exact projective rho_H values over active blocks from the local Pfam test shards.",
        ],
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
