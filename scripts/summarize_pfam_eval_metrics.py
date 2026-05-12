#!/usr/bin/env python3
"""Summarize held-out Pfam evaluator JSON files across runs or seeds."""

import argparse
import json
import re
import statistics
from pathlib import Path


DEFAULT_METRICS = (
    "reconstruction",
    "sparse_ce",
    "barycenter_mae",
    "barycenter_within_5",
    "monotonicity",
)


def _seed_from_path(path):
    match = re.search(r"(?:seed[_-]?|s)(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return None


def _load_run(path, metrics):
    payload = json.loads(path.read_text())
    aggregate = payload["aggregate"]
    settings = payload.get("settings", {})
    return {
        "path": str(path),
        "method": payload.get("method"),
        "checkpoint_step": payload.get("checkpoint_step"),
        "ckpt_path": payload.get("ckpt_path"),
        "seed": payload.get("seed", settings.get("seed", _seed_from_path(path))),
        "run_label": payload.get("run_label", settings.get("run_label")),
        "examples": aggregate.get("examples"),
        "metrics": {
            metric: float(aggregate[metric])
            for metric in metrics
            if metric in aggregate
        },
    }


def _group_key(run, group_by):
    values = []
    for field in group_by:
        values.append(str(run.get(field)))
    return tuple(values)


def _summarize(values):
    count = len(values)
    mean = statistics.fmean(values)
    if count > 1:
        std = statistics.stdev(values)
        sem = std / (count ** 0.5)
    else:
        std = 0.0
        sem = 0.0
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "sem": sem,
        "min": min(values),
        "max": max(values),
    }


def _summaries(runs, metrics, group_by):
    groups = {}
    for run in runs:
        groups.setdefault(_group_key(run, group_by), []).append(run)

    result = []
    for key, group_runs in sorted(groups.items()):
        row = {
            "group": dict(zip(group_by, key, strict=True)),
            "runs": group_runs,
            "metrics": {},
        }
        for metric in metrics:
            values = [run["metrics"][metric] for run in group_runs if metric in run["metrics"]]
            if values:
                row["metrics"][metric] = _summarize(values)
        result.append(row)
    return result


def _format_float(value):
    return f"{value:.4f}"


def _print_markdown(summary, metrics, group_by):
    columns = list(group_by)
    for metric in metrics:
        columns.extend([f"{metric} mean", f"{metric} std", f"{metric} n"])
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for group in summary:
        values = [group["group"].get(field, "") for field in group_by]
        for metric in metrics:
            item = group["metrics"].get(metric)
            if item is None:
                values.extend(["", "", "0"])
            else:
                values.extend([_format_float(item["mean"]), _format_float(item["std"]), str(item["n"])])
        print("| " + " | ".join(values) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+", type=Path)
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated aggregate metric names to summarize.",
    )
    parser.add_argument(
        "--group_by",
        default="method",
        help="Comma-separated run fields to group by, for example method or method,checkpoint_step.",
    )
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    args = parser.parse_args()

    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    group_by = [item.strip() for item in args.group_by.split(",") if item.strip()]
    if not group_by:
        raise ValueError("--group_by must include at least one field.")

    runs = [_load_run(path, metrics) for path in args.json_files]
    summary = _summaries(runs, metrics, group_by)
    if args.format == "json":
        print(json.dumps({"metrics": metrics, "group_by": group_by, "summary": summary}, indent=2, sort_keys=True))
    else:
        _print_markdown(summary, metrics, group_by)


if __name__ == "__main__":
    main()
