#!/usr/bin/env python3
"""Benchmark production Sinkhorn backward modes on one synthetic banded shape.

This is the TPU-facing harness for the remaining production backward evidence
gap. It measures the optimized tail-refinement custom VJP and, when requested,
the exact-autodiff reference mode exposed by the same public
`sinkhorn_attention` API. The reference mode is mainly for numerical agreement
at sizes where it compiles; long-context TPU runs can use `--skip_reference` to
collect optimized wall-clock and XLA memory-analysis fields only.

With `--include_score_adjoint_comparison`, the harness also isolates the R=2
score-adjoint stage on the same block-sparse support and compares explicit
stored-coordinate four-plan per-block materialization against the
one-reference formula used by the centered implementation. That stage
comparison is not the complete production custom VJP, nor an ungauged
gauge-recovered formula, but it is the closest direct-four-plan systems
diagnostic exposed by this script.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


MEMORY_FIELDS = (
    "argument_size_in_bytes",
    "output_size_in_bytes",
    "alias_size_in_bytes",
    "temp_size_in_bytes",
    "generated_code_size_in_bytes",
    "host_argument_size_in_bytes",
    "host_output_size_in_bytes",
    "host_alias_size_in_bytes",
    "host_temp_size_in_bytes",
    "host_generated_code_size_in_bytes",
)


def _memory_analysis(compiled):
    analysis = compiled.memory_analysis()
    if analysis is None:
        return None
    result = {}
    for field in MEMORY_FIELDS:
        if hasattr(analysis, field):
            value = getattr(analysis, field)
            if value is not None:
                result[field] = int(value)
    return result


def _make_inputs(args):
    key = jax.random.PRNGKey(args.seed)
    q_key, k_key, v_key, do_key = jax.random.split(key, 4)
    shape = (args.batch_size, args.heads, args.length, args.dim)
    q = jax.random.normal(q_key, shape, dtype=jnp.float32)
    k = jax.random.normal(k_key, shape, dtype=jnp.float32)
    v = jax.random.normal(v_key, shape, dtype=jnp.float32)
    d_out = jax.random.normal(do_key, shape, dtype=jnp.float32)
    q_mask = jnp.ones((args.batch_size, args.length), dtype=bool)
    k_mask = jnp.ones((args.batch_size, args.length), dtype=bool)
    return q, k, v, d_out, q_mask, k_mask


def _make_backward_fn(args, gradient_mode):
    def run(q, k, v, d_out, q_mask, k_mask):
        def loss_fn(q_in, k_in, v_in):
            out = ot_kernel.sinkhorn_attention(
                q_in,
                k_in,
                v_in,
                n_iters=args.n_iters,
                band_width=args.band_width,
                q_mask=q_mask,
                k_mask=k_mask,
                refine_steps=args.refine_steps,
                refine_tolerance=None,
                epsilon=args.epsilon,
                epsilon_scaling_steps=args.epsilon_scaling_steps,
                epsilon_scaling_factor=args.epsilon_scaling_factor,
                gradient_mode=gradient_mode,
                support_reconstruction=args.support_reconstruction,
            )
            return jnp.sum(out * d_out)

        return jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

    return run


def _tree_ready(value):
    return jax.block_until_ready(value)


def _time_compiled(compiled, inputs, repeats):
    _tree_ready(compiled(*inputs))
    times = []
    output = None
    for _ in range(int(repeats)):
        start = time.perf_counter()
        output = _tree_ready(compiled(*inputs))
        times.append(time.perf_counter() - start)
    return output, {
        "min_ms": min(times) * 1000.0,
        "mean_ms": (sum(times) / len(times)) * 1000.0,
        "repeats": int(repeats),
    }


def _compile_and_run(args, mode, inputs):
    fn = _make_backward_fn(args, mode)
    lowered = jax.jit(fn).lower(*inputs)
    compile_start = time.perf_counter()
    compiled = lowered.compile()
    compile_sec = time.perf_counter() - compile_start
    output, timing = _time_compiled(compiled, inputs, args.repeats)
    return {
        "mode": mode,
        "compile_sec": compile_sec,
        "timing": timing,
        "memory_analysis": _memory_analysis(compiled),
        "output": output,
    }


def _make_score_adjoint_stage_inputs(args, inputs):
    q, k, v, d_out, q_mask, k_mask = inputs
    batch, heads, length, _dim = q.shape
    q_mask = ot_kernel._canonicalize_mask(q_mask, batch, heads, length, "q_mask")
    k_mask = ot_kernel._canonicalize_mask(k_mask, batch, heads, length, "k_mask")
    scale = ot_kernel._score_scale(q.shape[-1], args.epsilon)
    block = 128

    u0, v0 = ot_kernel.compute_global_uv(
        q,
        k,
        n_iters=args.n_iters,
        band_width=args.band_width,
        q_mask=q_mask,
        k_mask=k_mask,
        epsilon=args.epsilon,
        epsilon_scaling_steps=args.epsilon_scaling_steps,
        epsilon_scaling_factor=args.epsilon_scaling_factor,
        support_reconstruction=args.support_reconstruction,
    )
    u_hist, v_hist = ot_kernel._fixed_refinement_history(
        q,
        k,
        u0,
        v0,
        refine_steps=2,
        band_width=args.band_width,
        q_mask=q_mask,
        k_mask=k_mask,
        score_scale=scale,
        support_reconstruction=args.support_reconstruction,
    )

    def matvec_row(u_a, v_b, vec):
        return ot_kernel._banded_matvec_P(
            q,
            k,
            u_a,
            v_b,
            vec,
            scale,
            block,
            q_mask,
            k_mask,
            args.band_width,
            support_reconstruction=args.support_reconstruction,
            direction="row",
        )

    def matvec_col(u_a, v_b, vec):
        return ot_kernel._banded_matvec_P(
            q,
            k,
            u_a,
            v_b,
            vec,
            scale,
            block,
            q_mask,
            k_mask,
            args.band_width,
            support_reconstruction=args.support_reconstruction,
            direction="col",
        )

    g_u, g_v, _d_v = ot_kernel._output_plan_pullback_stats(
        q,
        k,
        v,
        d_out,
        u_hist[-1],
        v_hist[-1],
        scale,
        block,
        args.band_width,
        q_mask,
        k_mask,
        support_reconstruction=args.support_reconstruction,
    )
    du_hist, dv_hist = ot_kernel._fixed_refinement_dual_cotangents(
        u_hist,
        v_hist,
        g_u,
        g_v,
        matvec_row,
        matvec_col,
    )
    return (
        q,
        k,
        v,
        d_out,
        q_mask,
        k_mask,
        u_hist[0],
        u_hist[1],
        u_hist[2],
        v_hist[0],
        v_hist[1],
        v_hist[2],
        du_hist[1],
        du_hist[2],
        dv_hist[1],
        dv_hist[2],
    )


def _make_score_adjoint_fn(args, mode):
    def run(
        q,
        k,
        v,
        d_out,
        q_mask,
        k_mask,
        u0,
        u1,
        u2,
        v0,
        v1,
        v2,
        du1,
        du2,
        dv1,
        dv2,
    ):
        block = 128
        scale = ot_kernel._score_scale(q.shape[-1], args.epsilon)
        profile = ot_kernel._resolve_support_profile(
            q,
            k,
            q_mask,
            k_mask,
            block,
            args.band_width,
            args.support_reconstruction,
        )
        num_blocks = profile["num_blocks"]
        slice_tokens = profile["row_slice_size"] * block
        local_block_offsets = jnp.arange(slice_tokens) // block
        d_q_init = jnp.zeros_like(q)
        d_k_init = jnp.zeros_like(k)

        def body(carry, i):
            d_q, d_k = carry
            q_block = jax.lax.dynamic_slice_in_dim(q, i * block, block, axis=-2)
            do_block = jax.lax.dynamic_slice_in_dim(d_out, i * block, block, axis=-2)
            q_mask_block = jax.lax.dynamic_slice_in_dim(q_mask, i * block, block, axis=-1)
            u1_block = jax.lax.dynamic_slice_in_dim(u1, i * block, block, axis=-1)
            u2_block = jax.lax.dynamic_slice_in_dim(u2, i * block, block, axis=-1)
            du1_block = jax.lax.dynamic_slice_in_dim(du1, i * block, block, axis=-1)
            du2_block = jax.lax.dynamic_slice_in_dim(du2, i * block, block, axis=-1)

            start_idx, block_mask = ot_kernel._row_band_window(i, profile)
            k_band = jax.lax.dynamic_slice_in_dim(k, start_idx * block, slice_tokens, axis=-2)
            v_band = jax.lax.dynamic_slice_in_dim(v, start_idx * block, slice_tokens, axis=-2)
            k_mask_band = jax.lax.dynamic_slice_in_dim(k_mask, start_idx * block, slice_tokens, axis=-1)
            v0_band = jax.lax.dynamic_slice_in_dim(v0, start_idx * block, slice_tokens, axis=-1)
            v1_band = jax.lax.dynamic_slice_in_dim(v1, start_idx * block, slice_tokens, axis=-1)
            v2_band = jax.lax.dynamic_slice_in_dim(v2, start_idx * block, slice_tokens, axis=-1)
            dv1_band = jax.lax.dynamic_slice_in_dim(dv1, start_idx * block, slice_tokens, axis=-1)
            dv2_band = jax.lax.dynamic_slice_in_dim(dv2, start_idx * block, slice_tokens, axis=-1)

            block_mask = block_mask[local_block_offsets]
            valid = q_mask_block[..., :, None] & k_mask_band[..., None, :] & block_mask[None, None, None, :]
            score = jnp.einsum("bhmd,bhnd->bhmn", q_block, k_band) * scale
            z_block = jnp.matmul(do_block, v_band.transpose(0, 1, 3, 2))

            if mode == "direct_four_plan":
                p22 = ot_kernel._exp_for_tpu(score + u2_block[..., :, None] + v2_band[..., None, :])
                p21 = ot_kernel._exp_for_tpu(score + u2_block[..., :, None] + v1_band[..., None, :])
                p11 = ot_kernel._exp_for_tpu(score + u1_block[..., :, None] + v1_band[..., None, :])
                p10 = ot_kernel._exp_for_tpu(score + u1_block[..., :, None] + v0_band[..., None, :])
                ds_block = (
                    p22 * z_block
                    - p22 * dv2_band[..., None, :]
                    - p21 * du2_block[..., :, None]
                    - p11 * dv1_band[..., None, :]
                    - p10 * du1_block[..., :, None]
                )
            elif mode == "one_reference":
                p22 = ot_kernel._exp_for_tpu(score + u2_block[..., :, None] + v2_band[..., None, :])
                alpha = ot_kernel._exp_for_tpu(u1_block - u2_block)
                beta = ot_kernel._exp_for_tpu(v1_band - v2_band)
                delta = ot_kernel._exp_for_tpu(v0_band - v2_band)
                modifier = (
                    z_block
                    - dv2_band[..., None, :]
                    - beta[..., None, :] * du2_block[..., :, None]
                    - alpha[..., :, None] * beta[..., None, :] * dv1_band[..., None, :]
                    - alpha[..., :, None] * delta[..., None, :] * du1_block[..., :, None]
                )
                ds_block = p22 * modifier
            else:
                raise ValueError(f"Unsupported score-adjoint mode={mode!r}.")

            ds_block = jnp.where(valid, ds_block, 0.0)
            d_q_block = jnp.einsum("bhmn,bhnd->bhmd", ds_block, k_band) * scale
            d_k_contrib = jnp.einsum("bhmn,bhmd->bhnd", ds_block, q_block) * scale

            d_q = jax.lax.dynamic_update_slice_in_dim(d_q, d_q_block, i * block, axis=-2)
            d_k_padded = jnp.zeros_like(k)
            d_k_padded = jax.lax.dynamic_update_slice_in_dim(d_k_padded, d_k_contrib, start_idx * block, axis=-2)
            return (d_q, d_k + d_k_padded), None

        (d_q, d_k), _ = jax.lax.scan(body, (d_q_init, d_k_init), jnp.arange(num_blocks))
        return d_q, d_k

    return run


def _compile_and_run_score_adjoint(args, mode, inputs):
    fn = _make_score_adjoint_fn(args, mode)
    lowered = jax.jit(fn).lower(*inputs)
    compile_start = time.perf_counter()
    compiled = lowered.compile()
    compile_sec = time.perf_counter() - compile_start
    output, timing = _time_compiled(compiled, inputs, args.repeats)
    return {
        "mode": mode,
        "compile_sec": compile_sec,
        "timing": timing,
        "memory_analysis": _memory_analysis(compiled),
        "output": output,
    }


def _score_adjoint_agreement(lhs, rhs):
    names = ("dQ_score", "dK_score")
    result = {}
    for name, lhs_grad, rhs_grad in zip(names, lhs, rhs, strict=True):
        diff = jnp.abs(lhs_grad - rhs_grad)
        denom = jnp.maximum(jnp.linalg.norm(rhs_grad), jnp.asarray(1e-30, dtype=rhs_grad.dtype))
        result[f"{name}_max_abs_diff"] = float(jnp.max(diff))
        result[f"{name}_mean_abs_diff"] = float(jnp.mean(diff))
        result[f"{name}_relative_l2_diff"] = float(jnp.linalg.norm(lhs_grad - rhs_grad) / denom)
    return result


def _run_score_adjoint_comparison(args, inputs):
    stage_inputs = _make_score_adjoint_stage_inputs(args, inputs)
    direct = _compile_and_run_score_adjoint(args, "direct_four_plan", stage_inputs)
    one_reference = _compile_and_run_score_adjoint(args, "one_reference", stage_inputs)
    return {
        "scope": (
            "Block-sparse R=2 score-adjoint stage after base/tail duals and "
            "dual cotangents are available; compares explicit stored-coordinate "
            "four-plan per-block materialization against the one-reference "
            "formula used by the centered implementation. This does not include "
            "dV, the complete production custom VJP, or ungauged gauge-scalar "
            "recovery for mixed-time plans."
        ),
        "coordinate_system": "stored_centered_representatives",
        "direct_four_plan_score_adjoint": {
            key: value
            for key, value in direct.items()
            if key != "output"
        },
        "one_reference_score_adjoint": {
            key: value
            for key, value in one_reference.items()
            if key != "output"
        },
        "agreement": _score_adjoint_agreement(direct["output"], one_reference["output"]),
    }


def _grad_agreement(lhs, rhs):
    lhs_loss, lhs_grads = lhs
    rhs_loss, rhs_grads = rhs
    names = ("dQ", "dK", "dV")
    result = {"loss_abs_diff": float(jnp.abs(lhs_loss - rhs_loss))}
    for name, lhs_grad, rhs_grad in zip(names, lhs_grads, rhs_grads, strict=True):
        diff = jnp.abs(lhs_grad - rhs_grad)
        result[f"{name}_max_abs_diff"] = float(jnp.max(diff))
        result[f"{name}_mean_abs_diff"] = float(jnp.mean(diff))
    return result


def _platform_info():
    return {
        "python": platform.python_version(),
        "backend": jax.default_backend(),
        "jax": jax.__version__,
        "jaxlib": jax.lib.__version__,
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
        "devices": [str(device) for device in jax.devices()],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--band_width", type=int, default=1024)
    parser.add_argument("--n_iters", type=int, default=15)
    parser.add_argument("--refine_steps", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_scaling_steps", type=int, default=0)
    parser.add_argument("--epsilon_scaling_factor", type=float, default=2.0)
    parser.add_argument("--support_reconstruction", default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--run_label", default=None)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--skip_reference",
        action="store_true",
        help="Only benchmark the optimized tail-refinement mode.",
    )
    parser.add_argument(
        "--include_score_adjoint_comparison",
        action="store_true",
        help=(
            "Also benchmark the block-sparse R=2 score-adjoint stage using "
            "explicit stored-coordinate four-plan blocks versus the "
            "one-reference formula."
        ),
    )
    args = parser.parse_args()

    if args.length % 128 != 0:
        raise ValueError("--length must be divisible by 128 for the production block layout.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if args.include_score_adjoint_comparison and int(args.refine_steps) != 2:
        raise ValueError("--include_score_adjoint_comparison currently requires --refine_steps 2.")

    global jax, jnp, ot_kernel  # noqa: PLW0603
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import ot_kernel  # noqa: PLC0415

    inputs = _make_inputs(args)
    optimized = _compile_and_run(args, "tail_refinement", inputs)
    settings = vars(args).copy()
    if settings["output"] is not None:
        settings["output"] = str(settings["output"])
    outputs = {
        "settings": settings,
        "platform": _platform_info(),
        "optimized_tail_refinement": {
            key: value
            for key, value in optimized.items()
            if key != "output"
        },
    }
    if not args.skip_reference:
        reference = _compile_and_run(args, "exact_autodiff", inputs)
        outputs["exact_autodiff_reference"] = {
            key: value
            for key, value in reference.items()
            if key != "output"
        }
        outputs["agreement"] = _grad_agreement(optimized["output"], reference["output"])
    if args.include_score_adjoint_comparison:
        outputs["block_sparse_score_adjoint_comparison"] = _run_score_adjoint_comparison(args, inputs)

    text = json.dumps(outputs, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
