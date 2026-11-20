"""Bounded-support capacity gates for Topology Hub Optimal Transport (THOT)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sparsemax(scores: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """Euclidean projection onto the probability simplex along ``axis``."""
    z = scores.astype(jnp.float32)
    z = z - jnp.max(z, axis=axis, keepdims=True)
    ordered = jnp.flip(jnp.sort(z, axis=axis), axis=axis)
    r_shape = [1] * z.ndim
    r_shape[axis] = z.shape[axis]
    r = jnp.arange(1, z.shape[axis] + 1, dtype=jnp.float32).reshape(r_shape)
    tau = (jnp.cumsum(ordered, axis=axis) - 1.0) / r
    support = jnp.clip(jnp.sum(ordered > tau, axis=axis, keepdims=True), 1, z.shape[axis])
    threshold = jnp.take_along_axis(tau, support - 1, axis=axis)
    return jnp.where(z > threshold, z - threshold, 0.0)


def _append_hub_capacities(
    a_real: jnp.ndarray,
    b_real: jnp.ndarray,
    hub_mass: jnp.ndarray | float,
    hubs: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if hubs < 0:
        raise ValueError(f"hubs must be non-negative, got {hubs}")
    if hubs == 0:
        return a_real.astype(jnp.float32), b_real.astype(jnp.float32)
    hub_shape = a_real.shape[:-1] + (hubs,)
    per_hub = jnp.asarray(hub_mass, dtype=jnp.float32) / float(hubs)
    hub_cap = jnp.broadcast_to(per_hub, hub_shape).astype(jnp.float32)
    return (
        jnp.concatenate((a_real.astype(jnp.float32), hub_cap), axis=-1),
        jnp.concatenate((b_real.astype(jnp.float32), hub_cap), axis=-1),
    )


def sparsemax_capacities(
    rec_scores: jnp.ndarray,
    src_scores: jnp.ndarray,
    *,
    hubs: int,
    excess_mass: float = 8.0,
    hub_total_mass: float = 8.0,
    adaptive_mass: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Allocate row/column marginal capacities ``(a, b)`` via ``sparsemax``.

    Args:
        rec_scores: Receiver gate logits of shape ``(..., L)``.
        src_scores: Source gate logits of shape ``(..., L)``.
        hubs: Number of global hub slots ``K``.
        excess_mass: Total excess marginal mass above the ``1/L`` uniform floor
            when ``adaptive_mass=False``.
        hub_total_mass: Aggregate capacity split uniformly across the ``K`` hubs
            when ``adaptive_mass=False``.
        adaptive_mass: If ``True``, scale excess mass per node using the clipped
            Renyi-2 effective cardinality ``N_eff = clip(1 / ||p||_2^2, 4, 32)``.

    Returns:
        Tuple ``(a, b)`` of ``float32`` marginals with shape ``(..., L + hubs)``.
    """
    if rec_scores.shape != src_scores.shape:
        raise ValueError(
            f"rec_scores and src_scores must match in shape, got {rec_scores.shape} and {src_scores.shape}"
        )
    length = rec_scores.shape[-1]
    if length < 1:
        raise ValueError("Sequence length L along the last axis must be >= 1")

    floor = jnp.asarray(1.0 / float(length), dtype=jnp.float32)
    p_rec = sparsemax(rec_scores, axis=-1)
    p_src = sparsemax(src_scores, axis=-1)
    if adaptive_mass:
        n_rec = jnp.clip(
            1.0 / jnp.maximum(jnp.sum(p_rec * p_rec, axis=-1, keepdims=True), 1e-6),
            4.0,
            32.0,
        )
        n_src = jnp.clip(
            1.0 / jnp.maximum(jnp.sum(p_src * p_src, axis=-1, keepdims=True), 1e-6),
            4.0,
            32.0,
        )
        a_real = floor + n_rec * p_rec
        b_real = floor + n_src * p_src
        hub_mass = 0.5 * (n_rec + n_src)
    else:
        a_real = floor + float(excess_mass) * p_rec
        b_real = floor + float(excess_mass) * p_src
        hub_mass = float(hub_total_mass)

    return _append_hub_capacities(a_real, b_real, hub_mass, hubs)


def relu_threshold_capacities(
    rec_scores: jnp.ndarray,
    src_scores: jnp.ndarray,
    theta_rec: jnp.ndarray | float,
    theta_src: jnp.ndarray | float,
    *,
    hubs: int,
    excess_mass: float = 8.0,
    hub_total_mass: float = 8.0,
    power: int = 1,
    normalize: bool = True,
    eps: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sorting-free ``O(L)`` thresholded-ReLU capacity gate.

    Supports normalized linear ReLU (``power=1, normalize=True``), normalized
    squared ReLU (``power=2, normalize=True``), and strictly local unnormalized
    ReLU (``normalize=False``).
    """
    if rec_scores.shape != src_scores.shape:
        raise ValueError(
            f"rec_scores and src_scores must match in shape, got {rec_scores.shape} and {src_scores.shape}"
        )
    if power not in (1, 2):
        raise ValueError(f"Unsupported power={power}; expected 1 or 2")
    length = rec_scores.shape[-1]
    if length < 1:
        raise ValueError("Sequence length L along the last axis must be >= 1")

    floor = jnp.asarray(1.0 / float(length), dtype=jnp.float32)
    w_rec = jax.nn.relu(rec_scores.astype(jnp.float32) - jnp.asarray(theta_rec, jnp.float32))
    w_src = jax.nn.relu(src_scores.astype(jnp.float32) - jnp.asarray(theta_src, jnp.float32))
    if power == 2:
        w_rec = w_rec * w_rec
        w_src = w_src * w_src

    if normalize:
        a_real = floor + float(excess_mass) * w_rec / (jnp.sum(w_rec, axis=-1, keepdims=True) + eps)
        b_real = floor + float(excess_mass) * w_src / (jnp.sum(w_src, axis=-1, keepdims=True) + eps)
    else:
        a_real = floor + w_rec
        b_real = floor + w_src

    return _append_hub_capacities(a_real, b_real, float(hub_total_mass), hubs)


def causal_threshold_gate(
    rec_scores: jnp.ndarray,
    src_scores: jnp.ndarray,
    theta_rec: jnp.ndarray | float,
    theta_src: jnp.ndarray | float,
    length: int,
    *,
    power: int = 1,
    cap_scale: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Strictly causal ``O(1)``-per-token thresholded capacity gate.

    Computes ``floor + cap_scale * ReLU(score - theta)^power`` token-by-token
    with zero future dependence and exact zero filler excess mass below ``theta``.
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    if power not in (1, 2):
        raise ValueError(f"Unsupported power={power}; expected 1 or 2")

    floor = jnp.asarray(1.0 / float(length), dtype=jnp.float32)
    ex_a = jax.nn.relu(rec_scores.astype(jnp.float32) - jnp.asarray(theta_rec, jnp.float32))
    ex_b = jax.nn.relu(src_scores.astype(jnp.float32) - jnp.asarray(theta_src, jnp.float32))
    if power == 2:
        ex_a = ex_a * ex_a
        ex_b = ex_b * ex_b
    a = floor + cap_scale * ex_a
    b = floor + cap_scale * ex_b
    return a, b

