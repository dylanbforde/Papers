#!/usr/bin/env python3
# ruff: noqa: E402

import argparse
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import ot_kernel
from src.benchmarks import benchmark_pfam_training as pfam


def _load_params(path: Path):
    with path.open("rb") as f:
        payload = pickle.load(f)
    return payload["params"], int(payload["step"])


def _iter_examples(data_dir: Path):
    shard_paths = sorted(data_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.npz files found in {data_dir}.")
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
        for idx in range(emb1.shape[0]):
            yield {
                "shard": shard_path.name,
                "index": idx,
                "emb1": emb1[idx],
                "emb2": emb2[idx],
                "target_cols": target_cols[idx],
                "target_mask": target_mask[idx],
                "q_mask": q_mask[idx],
                "k_mask": k_mask[idx],
            }


def _batch_examples(examples):
    return {
        "emb1": np.stack([item["emb1"] for item in examples])[:, None, :, :],
        "emb2": np.stack([item["emb2"] for item in examples])[:, None, :, :],
        "target_cols": np.stack([item["target_cols"] for item in examples]),
        "target_mask": np.stack([item["target_mask"] for item in examples]),
        "q_mask": np.stack([item["q_mask"] for item in examples]),
        "k_mask": np.stack([item["k_mask"] for item in examples]),
    }


def _project_batch(params, batch):
    emb_q = jnp.asarray(batch["emb1"], dtype=jnp.float32)
    emb_k = jnp.asarray(batch["emb2"], dtype=jnp.float32)
    return pfam._project_qkv(params, emb_q, emb_k)


def _outputs_and_metrics(params, batch, args):
    target_cols = jnp.asarray(batch["target_cols"], dtype=jnp.int32)
    target_mask = jnp.asarray(batch["target_mask"], dtype=bool)
    q_mask = jnp.asarray(batch["q_mask"], dtype=bool)
    k_mask = jnp.asarray(batch["k_mask"], dtype=bool)
    Q, K, V = _project_batch(params, batch)
    outputs = pfam.sinkhorn_outputs(
        Q,
        K,
        V,
        q_mask=q_mask,
        k_mask=k_mask,
        n_iters=args.n_iters,
        band_width=args.band_width,
        gradient_mode="exact_autodiff",
        refine_steps=args.refine_steps,
        refine_tolerance=None,
        epsilon=args.epsilon,
        epsilon_scaling_steps=0,
        epsilon_scaling_factor=2.0,
        support_reconstruction="none",
    )
    safe_cols, active_mask = pfam._prepare_sparse_supervision(target_cols, target_mask, q_mask, k_mask)
    target_v = pfam._batch_take_rows(V, safe_cols)
    sq_error = jnp.sum(jnp.square(outputs - target_v), axis=-1)[:, 0, :]
    rec_num = jnp.sum(jnp.where(active_mask, sq_error, 0.0), axis=-1)
    rec_den = jnp.maximum(jnp.sum(active_mask, axis=-1), 1)
    reconstruction = rec_num / rec_den

    dot = jnp.sum(outputs[:, 0, :, :] * target_v[:, 0, :, :], axis=-1)
    out_norm = jnp.linalg.norm(outputs[:, 0, :, :], axis=-1)
    target_norm = jnp.linalg.norm(target_v[:, 0, :, :], axis=-1)
    cosine = dot / jnp.maximum(out_norm * target_norm, 1e-8)
    cosine = jnp.sum(jnp.where(active_mask, cosine, 0.0), axis=-1) / rec_den

    u, v = ot_kernel.compute_global_uv(
        Q,
        K,
        n_iters=args.n_iters,
        band_width=args.band_width,
        q_mask=q_mask,
        k_mask=k_mask,
        epsilon=args.epsilon,
        epsilon_scaling_steps=0,
        epsilon_scaling_factor=2.0,
        support_reconstruction="none",
    )
    scale = 1.0 / (jnp.sqrt(Q.shape[-1]) * args.epsilon)
    matched_k = pfam._batch_take_rows(K, safe_cols)
    matched_v = pfam._batch_take_last(v, safe_cols)
    log_pos = jnp.sum(Q * matched_k, axis=-1)[:, 0, :] * scale + u[:, 0, :] + matched_v[:, 0, :]
    sparse_ce = -jnp.sum(jnp.where(active_mask, log_pos, 0.0), axis=-1) / rec_den

    jax.block_until_ready((reconstruction, cosine, sparse_ce, sq_error))
    return {
        "reconstruction": np.asarray(reconstruction),
        "cosine": np.asarray(cosine),
        "sparse_ce": np.asarray(sparse_ce),
        "sq_error": np.asarray(sq_error),
    }


def _score_candidates(params0, params1, examples, args):
    scored = []
    for start in range(0, len(examples), args.batch_size):
        chunk = examples[start : start + args.batch_size]
        batch = _batch_examples(chunk)
        initial = _outputs_and_metrics(params0, batch, args)
        final = _outputs_and_metrics(params1, batch, args)
        for offset, item in enumerate(chunk):
            q_len = int(item["q_mask"].sum())
            k_len = int(item["k_mask"].sum())
            active = int(item["target_mask"].sum())
            if q_len > args.max_visual_length or k_len > args.max_visual_length or active < args.min_active_targets:
                continue
            rec0 = float(initial["reconstruction"][offset])
            rec1 = float(final["reconstruction"][offset])
            scored.append(
                {
                    "shard": item["shard"],
                    "index": item["index"],
                    "q_len": q_len,
                    "k_len": k_len,
                    "active_targets": active,
                    "reconstruction_initial": rec0,
                    "reconstruction_final": rec1,
                    "reconstruction_improvement": rec0 - rec1,
                    "cosine_initial": float(initial["cosine"][offset]),
                    "cosine_final": float(final["cosine"][offset]),
                    "sparse_ce_initial": float(initial["sparse_ce"][offset]),
                    "sparse_ce_final": float(final["sparse_ce"][offset]),
                    "global_index": start + offset,
                }
            )
    if not scored:
        raise RuntimeError("No candidate examples met the visualization length/target constraints.")
    return sorted(scored, key=lambda item: (item["reconstruction_improvement"], -item["q_len"]), reverse=True)


def _single_example_arrays(params, example, args):
    batch = _batch_examples([example])
    target_cols = jnp.asarray(batch["target_cols"], dtype=jnp.int32)
    target_mask = jnp.asarray(batch["target_mask"], dtype=bool)
    q_mask = jnp.asarray(batch["q_mask"], dtype=bool)
    k_mask = jnp.asarray(batch["k_mask"], dtype=bool)
    Q, K, V = _project_batch(params, batch)
    outputs = pfam.sinkhorn_outputs(
        Q,
        K,
        V,
        q_mask=q_mask,
        k_mask=k_mask,
        n_iters=args.n_iters,
        band_width=args.band_width,
        gradient_mode="exact_autodiff",
        refine_steps=args.refine_steps,
        refine_tolerance=None,
        epsilon=args.epsilon,
        epsilon_scaling_steps=0,
        epsilon_scaling_factor=2.0,
        support_reconstruction="none",
    )
    u0, v0 = ot_kernel.compute_global_uv(
        Q,
        K,
        n_iters=args.n_iters,
        band_width=args.band_width,
        q_mask=q_mask,
        k_mask=k_mask,
        epsilon=args.epsilon,
        epsilon_scaling_steps=0,
        epsilon_scaling_factor=2.0,
        support_reconstruction="none",
    )
    scale = 1.0 / (jnp.sqrt(Q.shape[-1]) * args.epsilon)
    u_tail, v_tail = ot_kernel._tail_refinement_schedule(
        Q,
        K,
        u0,
        v0,
        refine_steps=args.refine_steps,
        refine_tolerance=None,
        band_width=args.band_width,
        q_mask=q_mask,
        k_mask=k_mask,
        score_scale=scale,
        support_reconstruction="none",
    )
    safe_cols, active_mask = pfam._prepare_sparse_supervision(target_cols, target_mask, q_mask, k_mask)
    target_v = pfam._batch_take_rows(V, safe_cols)
    sq_error = jnp.sum(jnp.square(outputs - target_v), axis=-1)[0, 0, :]
    dot = jnp.sum(outputs[0, 0, :, :] * target_v[0, 0, :, :], axis=-1)
    cosine = dot / jnp.maximum(
        jnp.linalg.norm(outputs[0, 0, :, :], axis=-1) * jnp.linalg.norm(target_v[0, 0, :, :], axis=-1),
        1e-8,
    )

    q_len = int(np.asarray(batch["q_mask"][0]).sum())
    k_len = int(np.asarray(batch["k_mask"][0]).sum())
    logits = (
        jnp.einsum("md,nd->mn", Q[0, 0, :q_len, :], K[0, 0, :k_len, :]) * scale
        + u_tail[0, 0, :q_len, None]
        + v_tail[0, 0, None, :k_len]
    )
    plan = jnp.exp(logits)
    row_sum = jnp.maximum(jnp.sum(plan, axis=-1), 1e-8)
    cols = jnp.arange(k_len, dtype=plan.dtype)
    barycenter = jnp.sum(plan * cols[None, :], axis=-1) / row_sum
    jax.block_until_ready((plan, sq_error, cosine, barycenter))
    return {
        "plan": np.asarray(plan),
        "sq_error": np.asarray(sq_error)[:q_len],
        "cosine": np.asarray(cosine)[:q_len],
        "barycenter": np.asarray(barycenter),
        "target_cols": np.asarray(batch["target_cols"][0])[:q_len],
        "target_mask": np.asarray(batch["target_mask"][0])[:q_len],
        "q_len": q_len,
        "k_len": k_len,
    }


def _rolling_mean(values, window):
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def _make_example_figure(initial, final, selected, output_path: Path):
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
    active = final["target_mask"].astype(bool)
    rows = np.flatnonzero(active)
    target_cols = final["target_cols"][active]

    fig = plt.figure(figsize=(6.7, 4.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.14, 1.0], height_ratios=[1.0, 0.85])
    ax_plan = fig.add_subplot(gs[:, 0])
    ax_err = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, 1])

    log_plan = np.log10(final["plan"] + 1e-12)
    im = ax_plan.imshow(
        log_plan,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=np.percentile(log_plan, 2),
        vmax=np.percentile(log_plan, 99.8),
    )
    ax_plan.scatter(target_cols, rows, s=3, c="#2bd3ff", linewidths=0, label="target")
    ax_plan.plot(final["barycenter"], np.arange(final["q_len"]), color="white", linewidth=0.8, alpha=0.9, label="plan mean")
    ax_plan.set_title("Final checkpoint transport")
    ax_plan.set_xlabel("key position")
    ax_plan.set_ylabel("query position")
    ax_plan.set_xlim(0, final["k_len"] - 1)
    ax_plan.set_ylim(0, final["q_len"] - 1)
    ax_plan.legend(loc="upper left", frameon=True)
    cbar = fig.colorbar(im, ax=ax_plan, fraction=0.046, pad=0.02)
    cbar.set_label("log10 plan mass")

    win = max(7, int(round(final["q_len"] / 28)))
    x = np.arange(final["q_len"])
    ax_err.plot(x, _rolling_mean(initial["sq_error"], win), color="#6b7280", linewidth=1.4, label="step 0")
    ax_err.plot(x, _rolling_mean(final["sq_error"], win), color="#0f766e", linewidth=1.6, label="step 1437")
    ax_err.fill_between(x, 0, np.minimum(initial["sq_error"], np.percentile(initial["sq_error"], 95)), color="#6b7280", alpha=0.12)
    ax_err.fill_between(x, 0, np.minimum(final["sq_error"], np.percentile(initial["sq_error"], 95)), color="#0f766e", alpha=0.14)
    ax_err.set_title("Output reconstruction error")
    ax_err.set_xlabel("query position")
    ax_err.set_ylabel("squared L2")
    ax_err.legend(frameon=False)

    labels = ["rec. MSE", "sparse CE"]
    before = [
        selected["reconstruction_initial"],
        selected["sparse_ce_initial"],
    ]
    after = [
        selected["reconstruction_final"],
        selected["sparse_ce_final"],
    ]
    idx = np.arange(len(labels))
    width = 0.36
    ax_bar.bar(idx - width / 2, before, width, color="#9ca3af", label="step 0")
    ax_bar.bar(idx + width / 2, after, width, color="#14b8a6", label="step 1437")
    ax_bar.set_xticks(idx, labels)
    ax_bar.set_title("Example metrics (lower is better)")
    ax_bar.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_err.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Real Pfam/ESM-2 example: {selected['shard']} #{selected['index']} "
        f"({selected['q_len']}x{selected['k_len']}, {selected['active_targets']} targets)",
        y=1.02,
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _make_grid_figure(items, output_path: Path):
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    n_items = len(items)
    n_cols = 2
    n_rows = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.7, 2.7 * n_rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, item in zip(axes, items):
        selected = item["selected"]
        arrays = item["final"]
        active = arrays["target_mask"].astype(bool)
        rows = np.flatnonzero(active)
        target_cols = arrays["target_cols"][active]
        log_plan = np.log10(arrays["plan"] + 1e-12)
        ax.imshow(
            log_plan,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            vmin=np.percentile(log_plan, 2),
            vmax=np.percentile(log_plan, 99.8),
        )
        ax.scatter(target_cols, rows, s=2.2, c="#2bd3ff", linewidths=0)
        ax.plot(arrays["barycenter"], np.arange(arrays["q_len"]), color="white", linewidth=0.65, alpha=0.9)
        ax.set_xlim(0, arrays["k_len"] - 1)
        ax.set_ylim(0, arrays["q_len"] - 1)
        ax.set_xlabel("key")
        ax.set_ylabel("query")
        ax.set_title(
            f"{selected['shard']} #{selected['index']} "
            f"({selected['q_len']}x{selected['k_len']}); "
            f"rec. {selected['reconstruction_initial']:.2f}->{selected['reconstruction_final']:.2f}"
        )
    for ax in axes[n_items:]:
        ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/pfam_shards_test"))
    parser.add_argument("--initial_ckpt", type=Path, default=Path("checkpoints/sinkhorn_ot/ckpt.pkl"))
    parser.add_argument(
        "--final_ckpt",
        type=Path,
        default=Path("checkpoints/sinkhorn_ot/promoted_step1437_ckpt.pkl"),
    )
    parser.add_argument("--output", type=Path, default=Path("paper_neurips/figures/pfam_checkpoint_example.pdf"))
    parser.add_argument("--metadata", type=Path, default=Path("paper_neurips/figures/pfam_checkpoint_example.json"))
    parser.add_argument("--grid_output", type=Path, default=Path("paper_neurips/figures/pfam_checkpoint_grid.pdf"))
    parser.add_argument("--grid_metadata", type=Path, default=Path("paper_neurips/figures/pfam_checkpoint_grid.json"))
    parser.add_argument("--grid_examples", type=int, default=4)
    parser.add_argument("--n_iters", type=int, default=15)
    parser.add_argument("--band_width", type=int, default=1024)
    parser.add_argument("--refine_steps", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--max_examples", type=int, default=36)
    parser.add_argument("--max_visual_length", type=int, default=320)
    parser.add_argument("--min_active_targets", type=int, default=180)
    args = parser.parse_args()

    params0, step0 = _load_params(args.initial_ckpt)
    params1, step1 = _load_params(args.final_ckpt)
    examples = list(_iter_examples(args.data_dir))[: args.max_examples]
    candidates = _score_candidates(params0, params1, examples, args)
    selected = candidates[0]
    example = examples[selected["global_index"]]

    initial_arrays = _single_example_arrays(params0, example, args)
    final_arrays = _single_example_arrays(params1, example, args)
    _make_example_figure(initial_arrays, final_arrays, selected, args.output)

    grid_candidates = candidates[: max(0, args.grid_examples)]
    grid_items = []
    for candidate in grid_candidates:
        grid_example = examples[candidate["global_index"]]
        grid_items.append(
            {
                "selected": candidate,
                "final": _single_example_arrays(params1, grid_example, args),
            }
        )
    if grid_items:
        _make_grid_figure(grid_items, args.grid_output)

    report = {
        "figure": str(args.output),
        "grid_figure": str(args.grid_output) if grid_items else None,
        "data_dir": str(args.data_dir),
        "initial_checkpoint": str(args.initial_ckpt),
        "final_checkpoint": str(args.final_ckpt),
        "initial_step": step0,
        "final_step": step1,
        "settings": {
            "n_iters": args.n_iters,
            "band_width": args.band_width,
            "refine_steps": args.refine_steps,
            "epsilon": args.epsilon,
            "max_examples_scanned": args.max_examples,
            "max_visual_length": args.max_visual_length,
            "min_active_targets": args.min_active_targets,
            "grid_examples": args.grid_examples,
        },
        "selected": selected,
        "top_candidates": candidates[:8],
        "grid_selected": grid_candidates,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.grid_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.grid_metadata.write_text(
        json.dumps(
            {
                "figure": str(args.grid_output),
                "data_dir": str(args.data_dir),
                "initial_checkpoint": str(args.initial_ckpt),
                "final_checkpoint": str(args.final_ckpt),
                "initial_step": step0,
                "final_step": step1,
                "settings": report["settings"],
                "selected": grid_candidates,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
