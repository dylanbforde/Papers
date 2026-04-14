#!/usr/bin/env python3
"""Summarize benchmark logs into a compact decision table."""

from __future__ import annotations

import argparse
import glob
import math
import re
from pathlib import Path


HEADER_PATTERNS = {
    "method": re.compile(r"^Pfam training \((?P<value>[^)]+)\)$"),
    "per_device_batch": re.compile(r"^\s+per-device batch: (?P<value>\d+)$"),
    "sinkhorn_refine_steps": re.compile(r"^\s+sinkhorn refine steps: (?P<value>\d+)$"),
    "sinkhorn_epsilon": re.compile(r"^\s+sinkhorn epsilon: (?P<value>[-+]?\d+(?:\.\d+)?)$"),
    "sinkhorn_epsilon_scaling_steps": re.compile(r"^\s+sinkhorn epsilon scaling steps: (?P<value>\d+)$"),
    "sinkhorn_epsilon_scaling_factor": re.compile(
        r"^\s+sinkhorn epsilon scaling factor: (?P<value>[-+]?\d+(?:\.\d+)?)$"
    ),
    "train_reconstruction_weight": re.compile(
        r"^\s+train reconstruction weight: (?P<value>[-+]?\d+(?:\.\d+)?)$"
    ),
    "train_sparse_ce_weight": re.compile(r"^\s+train sparse CE weight: (?P<value>[-+]?\d+(?:\.\d+)?)$"),
    "train_monotonicity_weight": re.compile(
        r"^\s+train monotonicity weight: (?P<value>[-+]?\d+(?:\.\d+)?)$"
    ),
    "gap_mode": re.compile(r"^\s+gap mode: (?P<value>\S+)$"),
}

STEP_PATTERN = re.compile(
    r"^Step\s+(?P<step>\d+)\s+\|\s+Loss:\s+(?P<loss>[-+]?\d+(?:\.\d+)?)\s+\|\s+"
    r"CE:\s+(?P<ce>off|[-+]?\d+(?:\.\d+)?)\s+\|\s+"
    r"Mono:\s+(?P<mono>off|[-+]?\d+(?:\.\d+)?)\s+\|\s+"
    r"Time:\s+(?P<seconds>\d+)s\s+\|\s+Ex/s:\s+(?P<ex_s>[-+]?\d+(?:\.\d+)?)$"
)


def _parse_optional_float(raw: str) -> float | None:
    if raw == "off":
        return None
    return float(raw)


def parse_log(path: Path) -> dict:
    header = {}
    steps = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        matched_header = False
        for key, pattern in HEADER_PATTERNS.items():
            match = pattern.match(line)
            if match:
                header[key] = match.group("value")
                matched_header = True
                break
        if matched_header:
            continue
        match = STEP_PATTERN.match(line)
        if match:
            steps.append(
                {
                    "step": int(match.group("step")),
                    "loss": float(match.group("loss")),
                    "ce": _parse_optional_float(match.group("ce")),
                    "mono": _parse_optional_float(match.group("mono")),
                    "seconds": int(match.group("seconds")),
                    "ex_s": float(match.group("ex_s")),
                }
            )

    if not steps:
        raise ValueError(f"No benchmark step lines found in {path}")

    last = steps[-1]
    ce_values = [entry["ce"] for entry in steps if entry["ce"] is not None]
    mono_values = [entry["mono"] for entry in steps if entry["mono"] is not None]
    result = {
        "path": str(path),
        "run": path.stem,
        "method": header.get("method", "unknown"),
        "per_device_batch": int(header["per_device_batch"]) if "per_device_batch" in header else None,
        "sinkhorn_refine_steps": int(header["sinkhorn_refine_steps"]) if "sinkhorn_refine_steps" in header else None,
        "sinkhorn_epsilon": float(header["sinkhorn_epsilon"]) if "sinkhorn_epsilon" in header else None,
        "sinkhorn_epsilon_scaling_steps": (
            int(header["sinkhorn_epsilon_scaling_steps"]) if "sinkhorn_epsilon_scaling_steps" in header else None
        ),
        "sinkhorn_epsilon_scaling_factor": (
            float(header["sinkhorn_epsilon_scaling_factor"])
            if "sinkhorn_epsilon_scaling_factor" in header
            else None
        ),
        "train_reconstruction_weight": (
            float(header["train_reconstruction_weight"]) if "train_reconstruction_weight" in header else None
        ),
        "train_sparse_ce_weight": (
            float(header["train_sparse_ce_weight"]) if "train_sparse_ce_weight" in header else None
        ),
        "train_monotonicity_weight": (
            float(header["train_monotonicity_weight"]) if "train_monotonicity_weight" in header else None
        ),
        "gap_mode": header.get("gap_mode"),
        "last_step": last["step"],
        "last_loss": last["loss"],
        "last_ce": last["ce"],
        "best_ce": min(ce_values) if ce_values else None,
        "last_mono": last["mono"],
        "best_mono": min(mono_values) if mono_values else None,
        "last_ex_s": last["ex_s"],
        "last_seconds": last["seconds"],
    }
    return result


def _rank_key(entry: dict) -> tuple:
    inf = math.inf
    return (
        entry["last_ce"] if entry["last_ce"] is not None else inf,
        entry["last_mono"] if entry["last_mono"] is not None else inf,
        -entry["last_ex_s"],
        entry["last_loss"],
    )


def _format_optional(value, fmt: str) -> str:
    if value is None:
        return "n/a"
    return format(value, fmt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Log files or directories to summarize.")
    args = parser.parse_args()

    log_paths = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            log_paths.extend(sorted(path.glob("*.log")))
        else:
            log_paths.extend(
                [Path(match) for match in sorted(glob.glob(raw))] if any(ch in raw for ch in "*?[]") else [path]
            )

    unique_paths = []
    seen = set()
    for path in log_paths:
        resolved = path.resolve()
        if resolved not in seen and path.exists():
            seen.add(resolved)
            unique_paths.append(path)

    if not unique_paths:
        raise SystemExit("No log files found.")

    rows = sorted((parse_log(path) for path in unique_paths), key=_rank_key)
    header = (
        f"{'Run':<20} {'K':>2} {'eps':>5} {'scale':>5} {'fac':>4} {'gap':<13} "
        f"{'r':>4} {'ce':>4} {'mono':>5} {'bs':>3} {'step':>6} {'loss':>8} {'CE':>8} {'Mono':>8} {'Ex/s':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['run']:<20} "
            f"{_format_optional(row['sinkhorn_refine_steps'], 'd'):>2} "
            f"{_format_optional(row['sinkhorn_epsilon'], '.2f'):>5} "
            f"{_format_optional(row['sinkhorn_epsilon_scaling_steps'], 'd'):>5} "
            f"{_format_optional(row['sinkhorn_epsilon_scaling_factor'], '.2f'):>4} "
            f"{(row['gap_mode'] or 'n/a'):<13} "
            f"{_format_optional(row['train_reconstruction_weight'], '.2f'):>4} "
            f"{_format_optional(row['train_sparse_ce_weight'], '.2f'):>4} "
            f"{_format_optional(row['train_monotonicity_weight'], '.2f'):>5} "
            f"{_format_optional(row['per_device_batch'], 'd'):>3} "
            f"{row['last_step']:>6} "
            f"{row['last_loss']:>8.4f} "
            f"{_format_optional(row['last_ce'], '.4f'):>8} "
            f"{_format_optional(row['last_mono'], '.4f'):>8} "
            f"{row['last_ex_s']:>8.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
