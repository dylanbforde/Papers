#!/usr/bin/env python3
"""Render Pfam training curves from benchmark logs and evaluator JSONs."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


HEADER_PATTERNS = {
    "method": re.compile(r"^Pfam training \((?P<value>[^)]+)\)$"),
    "per_device_batch": re.compile(r"^\s+per-device batch: (?P<value>\d+)$"),
    "sinkhorn_refine_steps": re.compile(r"^\s+sinkhorn refine steps: (?P<value>\d+)$"),
    "sinkhorn_epsilon": re.compile(r"^\s+sinkhorn epsilon: (?P<value>[-+]?\d+(?:\.\d+)?)$"),
    "band_width": re.compile(r"^\s+band width: (?P<value>\d+)$"),
    "train_reconstruction_weight": re.compile(
        r"^\s+train reconstruction weight: (?P<value>[-+]?\d+(?:\.\d+)?)$"
    ),
    "train_sparse_ce_weight": re.compile(r"^\s+train sparse CE weight: (?P<value>[-+]?\d+(?:\.\d+)?)$"),
    "train_monotonicity_weight": re.compile(
        r"^\s+train monotonicity weight: (?P<value>[-+]?\d+(?:\.\d+)?)$"
    ),
    "gap_mode": re.compile(r"^\s+gap mode: (?P<value>\S+)$"),
    "support_reconstruction": re.compile(r"^\s+support reconstruction: (?P<value>\S+)$"),
}

STEP_PATTERN = re.compile(
    r"^Step\s+(?P<step>\d+)\s+\|\s+Loss:\s+(?P<loss>[-+]?\d+(?:\.\d+)?)\s+\|\s+"
    r"CE:\s+(?P<ce>off|[-+]?\d+(?:\.\d+)?)\s+\|\s+"
    r"Mono:\s+(?P<mono>off|[-+]?\d+(?:\.\d+)?)\s+\|\s+"
    r"Time:\s+(?P<seconds>\d+)s\s+\|\s+Ex/s:\s+(?P<ex_s>[-+]?\d+(?:\.\d+)?)$"
)


def _optional_float(raw: str) -> float | None:
    return None if raw == "off" else float(raw)


def _expand_paths(raw_paths: list[str], *, default_pattern: str = "*.log") -> list[Path]:
    paths: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.glob(default_pattern)))
        elif any(char in raw for char in "*?[]"):
            paths.extend(Path(match) for match in sorted(glob.glob(raw)))
        else:
            paths.append(path)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def _parse_log(path: Path) -> dict:
    header: dict[str, str] = {}
    steps = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for key, pattern in HEADER_PATTERNS.items():
            match = pattern.match(line)
            if match:
                header[key] = match.group("value")
                break
        else:
            match = STEP_PATTERN.match(line)
            if match:
                steps.append(
                    {
                        "step": int(match.group("step")),
                        "loss": float(match.group("loss")),
                        "ce": _optional_float(match.group("ce")),
                        "mono": _optional_float(match.group("mono")),
                        "seconds": int(match.group("seconds")),
                        "ex_s": float(match.group("ex_s")),
                    }
                )

    if not steps:
        raise ValueError(f"No benchmark step lines found in {path}")

    return {
        "path": str(path),
        "run": path.stem,
        "method": header.get("method", "unknown"),
        "header": header,
        "steps": steps,
    }


def _load_eval_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("aggregate", payload)
    if "checkpoint_step" not in payload or "reconstruction" not in metrics:
        raise ValueError(f"{path} is missing checkpoint_step or reconstruction")
    return {
        "path": str(path),
        "run": path.stem,
        "method": payload.get("method", "unknown"),
        "checkpoint_step": int(payload["checkpoint_step"]),
        "reconstruction": float(metrics["reconstruction"]),
        "sparse_ce": float(metrics["sparse_ce"]) if "sparse_ce" in metrics else None,
        "run_label": payload.get("run_label"),
        "seed": payload.get("seed"),
    }


def _parse_checkpoint_steps(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _style_label(run: dict) -> str:
    pieces = [run["run"]]
    if run.get("method") and run["method"] != "unknown":
        pieces.append(f"({run['method']})")
    return " ".join(pieces)


def _render(runs: list[dict], evals: list[dict], checkpoint_steps: list[int], output: Path) -> None:
    import matplotlib.pyplot as plt

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

    fig, ax_loss = plt.subplots(figsize=(6.7, 2.8), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, run in enumerate(runs):
        steps = [item["step"] for item in run["steps"]]
        losses = [item["loss"] for item in run["steps"]]
        ax_loss.plot(
            steps,
            losses,
            marker="o",
            markersize=2.8,
            linewidth=1.2,
            color=colors[idx % len(colors)],
            label=_style_label(run),
        )

    marker_steps = set(checkpoint_steps)
    marker_steps.update(item["checkpoint_step"] for item in evals)
    for step in sorted(marker_steps):
        ax_loss.axvline(step, color="0.7", linewidth=0.7, linestyle="--", zorder=0)

    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("minibatch loss")
    ax_loss.set_title("Pfam training trajectory")
    ax_loss.grid(True, alpha=0.22)
    ax_loss.spines[["top", "right"]].set_visible(False)

    handles, labels = ax_loss.get_legend_handles_labels()
    if evals:
        ax_eval = ax_loss.twinx()
        for idx, item in enumerate(evals):
            label = item.get("run_label") or item["run"]
            ax_eval.scatter(
                [item["checkpoint_step"]],
                [item["reconstruction"]],
                marker="D",
                s=32,
                color=colors[(idx + len(runs)) % len(colors)],
                label=f"held-out recon: {label}",
                zorder=3,
            )
        ax_eval.set_ylabel("held-out reconstruction MSE")
        ax_eval.spines["top"].set_visible(False)
        eval_handles, eval_labels = ax_eval.get_legend_handles_labels()
        handles.extend(eval_handles)
        labels.extend(eval_labels)

    if handles:
        ax_loss.legend(handles, labels, frameon=False, loc="best")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="Benchmark log files, globs, or directories.")
    parser.add_argument(
        "--eval_json",
        nargs="*",
        default=[],
        help="Optional held-out evaluator JSONs to overlay at checkpoint_step.",
    )
    parser.add_argument(
        "--checkpoint_steps",
        default="",
        help="Optional comma-separated checkpoint steps to mark, for example 0,250,500,750.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_neurips/figures/pfam_training_curve.pdf"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("paper_neurips/figures/pfam_training_curve.json"),
    )
    args = parser.parse_args()

    log_paths = _expand_paths(args.logs)
    if not log_paths:
        raise SystemExit("No log files found.")
    runs = [_parse_log(path) for path in log_paths]

    eval_paths = _expand_paths(args.eval_json, default_pattern="*.json")
    evals = [_load_eval_json(path) for path in eval_paths]
    checkpoint_steps = _parse_checkpoint_steps(args.checkpoint_steps)

    _render(runs, evals, checkpoint_steps, args.output)
    metadata = {
        "figure": str(args.output),
        "logs": [str(path) for path in log_paths],
        "eval_json": [str(path) for path in eval_paths],
        "checkpoint_steps": checkpoint_steps,
        "runs": runs,
        "heldout_evaluations": evals,
        "notes": [
            "Training curves are rendered from benchmark log lines emitted by benchmark_pfam_training.py.",
            "Held-out reconstruction markers require evaluator JSONs from eval_pfam_checkpoint.py.",
        ],
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
