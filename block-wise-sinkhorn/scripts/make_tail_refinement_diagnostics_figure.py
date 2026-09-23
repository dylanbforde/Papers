#!/usr/bin/env python3
"""Render the synthetic tail-refinement diagnostic summary figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BIAS_CERTIFICATE_ROWS = [
    {"R": 0, "eta_l2": 5.89e-3, "gradient_gap": 6.29e-4, "residual": 2.47e-10},
    {"R": 1, "eta_l2": 5.40e-4, "gradient_gap": 5.93e-5, "residual": 3.64e-11},
    {"R": 2, "eta_l2": 5.47e-5, "gradient_gap": 5.33e-6, "residual": 2.84e-11},
    {"R": 4, "eta_l2": 9.03e-7, "gradient_gap": 9.99e-8, "residual": 2.74e-11},
]


def _load_exactness(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sorted(payload["table_rows"], key=lambda row: row["length"])


def _render_figure(exactness_rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    lengths = np.asarray([row["length"] for row in exactness_rows], dtype=np.float64)
    d_q = np.asarray([row["dQ_max_abs"] for row in exactness_rows], dtype=np.float64)
    d_k = np.asarray([row["dK_max_abs"] for row in exactness_rows], dtype=np.float64)
    d_v = np.asarray([row["dV_max_abs"] for row in exactness_rows], dtype=np.float64)
    grad_l2 = np.asarray([row["max_gradient_l2_abs"] for row in exactness_rows], dtype=np.float64)
    out_rel = np.asarray([row["output_l2_rel"] for row in exactness_rows], dtype=np.float64)

    r_values = np.asarray([row["R"] for row in BIAS_CERTIFICATE_ROWS], dtype=np.float64)
    eta_l2 = np.asarray([row["eta_l2"] for row in BIAS_CERTIFICATE_ROWS], dtype=np.float64)
    gap = np.asarray([row["gradient_gap"] for row in BIAS_CERTIFICATE_ROWS], dtype=np.float64)
    residual = np.asarray([row["residual"] for row in BIAS_CERTIFICATE_ROWS], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.55), constrained_layout=True)
    ax = axes[0]
    ax.loglog(lengths, d_q, marker="o", linewidth=1.3, label="max |Delta dQ|")
    ax.loglog(lengths, d_k, marker="s", linewidth=1.3, label="max |Delta dK|")
    ax.loglog(lengths, d_v, marker="^", linewidth=1.3, label="max |Delta dV|")
    ax.loglog(lengths, grad_l2, marker="D", linewidth=1.3, label="max grad L2 abs")
    ax.loglog(lengths, out_rel, marker="x", linewidth=1.1, label="output L2 rel")
    ax.set_title("Exactness vs. sequence length")
    ax.set_xlabel("sequence length L")
    ax.set_ylabel("error")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, loc="best")

    ax = axes[1]
    ax.semilogy(r_values, eta_l2, marker="o", linewidth=1.4, label="base cotangent norm")
    ax.semilogy(r_values, gap, marker="s", linewidth=1.4, label="certified grad gap")
    ax.semilogy(r_values, residual, marker="^", linewidth=1.4, label="residual")
    ax.axhline(1e-5, color="0.35", linestyle="--", linewidth=0.9, label="selector tol.")
    ax.set_xticks(r_values)
    ax.set_title("Bias certificate decay")
    ax.set_xlabel("tail depth R")
    ax.set_ylabel("value")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False, loc="best")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exactness_json",
        type=Path,
        default=Path("paper_neurips/figures/tail_refinement_exactness.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_neurips/figures/tail_refinement_diagnostics.pdf"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("paper_neurips/figures/tail_refinement_diagnostics.json"),
    )
    args = parser.parse_args()

    exactness_rows = _load_exactness(args.exactness_json)
    _render_figure(exactness_rows, args.output)

    metadata = {
        "figure": str(args.output),
        "exactness_json": str(args.exactness_json),
        "exactness_rows": exactness_rows,
        "bias_certificate_rows": BIAS_CERTIFICATE_ROWS,
        "notes": [
            "Exactness rows are loaded from tail_refinement_exactness.json.",
            "Bias-certificate rows match the manuscript bias-certificate table.",
        ],
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
