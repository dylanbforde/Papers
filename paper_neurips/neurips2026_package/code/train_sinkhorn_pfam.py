#!/usr/bin/env python3

import argparse
import functools
import json
import os
import pickle
import time
import warnings
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import multihost_utils

import ot_kernel


DUSTBIN_BLOCK_SIZE = 128


class PfamShardDataset:
    def __init__(self, data_dir, seed=0):
        self.data_dir = data_dir
        self.shard_files = self._discover_shard_files()
        self.metadata = self._load_or_infer_metadata()
        self.total_shards = len(self.shard_files)
        self.curr_shard_idx = -1
        self.curr_shard_data = None
        self.rng = np.random.default_rng(seed)

    def _discover_shard_files(self):
        shard_files = sorted(
            name for name in os.listdir(self.data_dir) if name.startswith("shard_") and name.endswith(".npz")
        )
        if not shard_files:
            raise FileNotFoundError(f"No shard_*.npz files found in {self.data_dir!r}.")
        return shard_files

    def _infer_metadata(self):
        total_pairs = 0
        shard_size = 0
        for idx, shard_name in enumerate(self.shard_files):
            shard_path = os.path.join(self.data_dir, shard_name)
            with np.load(shard_path, allow_pickle=False) as shard:
                current_pairs = int(shard["emb1"].shape[0])
            if idx == 0:
                shard_size = current_pairs
            total_pairs += current_pairs
        return {"total_pairs": total_pairs, "shard_size": shard_size, "total_shards": len(self.shard_files)}

    def _load_or_infer_metadata(self):
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, encoding="utf-8") as f:
                return json.load(f)
        metadata = self._infer_metadata()
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        return metadata

    def _load_shard(self, shard_idx):
        if self.curr_shard_idx == shard_idx:
            return
        shard_path = os.path.join(self.data_dir, self.shard_files[shard_idx])
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
        self.curr_shard_data = {
            "emb1": emb1,
            "emb2": emb2,
            "target_cols": target_cols,
            "target_mask": target_mask,
            "q_mask": q_mask,
            "k_mask": k_mask,
        }
        self.curr_shard_idx = shard_idx

    def sample_batch(self, batch_size):
        shard_idx = int(self.rng.integers(0, self.total_shards))
        self._load_shard(shard_idx)
        shard_data = self.curr_shard_data
        if shard_data is None:
            raise RuntimeError(f"Failed to load shard {shard_idx}.")
        num_in_shard = shard_data["emb1"].shape[0]
        indices = self.rng.choice(num_in_shard, batch_size, replace=batch_size > num_in_shard)
        return (
            shard_data["emb1"][indices][:, None, :, :],
            shard_data["emb2"][indices][:, None, :, :],
            shard_data["target_cols"][indices],
            shard_data["target_mask"][indices],
            shard_data["q_mask"][indices],
            shard_data["k_mask"][indices],
        )


def muzzle_orthogonalize(grad, steps=5):
    if grad.ndim < 2:
        return grad
    original_shape = grad.shape
    flat = grad.reshape(grad.shape[0], -1)
    x = flat / (jnp.linalg.norm(flat) + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        x = 3.4445 * x + (-4.7750 * gram + 2.0315 * gram @ gram) @ x
    return x.reshape(original_shape)


def muon_optimizer(lr):
    def init_fn(params: optax.Params) -> optax.EmptyState:
        return optax.EmptyState()

    def update_fn(grads: optax.Updates, state: optax.EmptyState, params: optax.Params | None = None):
        del params
        updates = jax.tree_util.tree_map(muzzle_orthogonalize, grads)
        updates = jax.tree_util.tree_map(lambda x: -lr * x, updates)
        return updates, state

    return optax.GradientTransformation(init_fn, cast(optax.TransformUpdateFn, update_fn))


def stiefel_qr_retraction(matrix):
    q, r = jnp.linalg.qr(matrix, mode="reduced")
    phase = jnp.sign(jnp.diag(r))
    phase = jnp.where(phase == 0, 1.0, phase).astype(q.dtype)
    return q * phase[None, :]


def apply_projection_manifold(params, projection_manifold):
    if projection_manifold == "none":
        return params
    if projection_manifold == "stiefel_qk":
        return {**params, "W_q": stiefel_qr_retraction(params["W_q"]), "W_k": stiefel_qr_retraction(params["W_k"])}
    raise ValueError(f"Unsupported projection_manifold={projection_manifold!r}.")


def _unreplicate_tree(tree):
    return jax.tree_util.tree_map(lambda x: x[0], tree)


def _replicate_tree(tree, devices):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        return jax.device_put_replicated(tree, devices)


def save_checkpoint(params, opt_state, step, ckpt_dir, replicated=False):
    os.makedirs(ckpt_dir, exist_ok=True)
    if replicated:
        params = _unreplicate_tree(params)
        opt_state = _unreplicate_tree(opt_state)
    payload = {"params": jax.device_get(params), "opt_state": jax.device_get(opt_state), "step": int(step)}
    with open(os.path.join(ckpt_dir, "ckpt.pkl"), "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(ckpt_dir):
    path = os.path.join(ckpt_dir, "ckpt.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["params"], payload["opt_state"], int(payload["step"])


def _batch_take_rows(tensor, indices):
    return jax.vmap(lambda x, idx: jnp.take(x, idx, axis=-2))(tensor, indices)


def _batch_take_last(tensor, indices):
    return jax.vmap(lambda x, idx: jnp.take(x, idx, axis=-1))(tensor, indices)


def _rowwise_take_cols(tensor, indices):
    gathered = jnp.take_along_axis(tensor, indices[:, None, :, None], axis=-1)
    return gathered[..., 0]


def _project_qk(params, emb_q, emb_k):
    emb_q = emb_q.astype(jnp.float32)
    emb_k = emb_k.astype(jnp.float32)
    q_proj = jnp.einsum("bhld,dp->bhlp", emb_q, params["W_q"])
    k_proj = jnp.einsum("bhld,dp->bhlp", emb_k, params["W_k"])
    return q_proj, k_proj


def _project_qkv(params, emb_q, emb_k):
    q_proj, k_proj = _project_qk(params, emb_q, emb_k)
    v_proj = jnp.einsum("bhld,dp->bhlp", emb_k.astype(jnp.float32), params["W_v"])
    return q_proj, k_proj, v_proj


def _append_gap_block(sequence, gap_token):
    batch, heads, _length, dim = sequence.shape
    extra = jnp.zeros((batch, heads, DUSTBIN_BLOCK_SIZE, dim), dtype=sequence.dtype)
    extra = extra.at[:, :, 0, :].set(gap_token.astype(sequence.dtype)[None, None, :])
    return jnp.concatenate([sequence, extra], axis=-2)


def _append_gap_mask(mask):
    batch, _length = mask.shape
    extra = jnp.zeros((batch, DUSTBIN_BLOCK_SIZE), dtype=bool)
    extra = extra.at[:, 0].set(True)
    return jnp.concatenate([mask, extra], axis=-1)


def _apply_gap_mode(params, Q, K, V, target_cols, target_mask, q_mask, k_mask, gap_mode):
    if gap_mode == "none":
        return Q, K, V, target_cols, target_mask, q_mask, k_mask
    if gap_mode != "dustbin_block":
        raise ValueError(f"Unsupported gap_mode={gap_mode!r}.")
    dustbin_index = Q.shape[-2]
    Q = _append_gap_block(Q, params["gap_q"])
    K = _append_gap_block(K, params["gap_k"])
    V = _append_gap_block(V, params["gap_v"])
    q_mask_base = jnp.asarray(q_mask, dtype=bool)
    k_mask_base = jnp.asarray(k_mask, dtype=bool)
    unmatched_query = q_mask_base & ~jnp.asarray(target_mask, dtype=bool)
    target_cols = jnp.where(unmatched_query, dustbin_index, target_cols)
    target_mask = q_mask_base
    target_cols = jnp.concatenate(
        [target_cols, jnp.zeros((target_cols.shape[0], DUSTBIN_BLOCK_SIZE), dtype=target_cols.dtype)], axis=-1
    )
    target_mask = jnp.concatenate(
        [target_mask, jnp.zeros((target_mask.shape[0], DUSTBIN_BLOCK_SIZE), dtype=bool)], axis=-1
    )
    q_mask = _append_gap_mask(q_mask_base)
    k_mask = _append_gap_mask(k_mask_base)
    return Q, K, V, target_cols, target_mask, q_mask, k_mask


def _effective_band_width(band_width, gap_mode, base_length):
    if gap_mode == "dustbin_block" and band_width is not None:
        return max(int(band_width), int(base_length))
    return band_width


def _prepare_sparse_supervision(target_cols, target_mask, q_mask, k_mask):
    safe_cols = jnp.clip(target_cols, 0, k_mask.shape[-1] - 1)
    matched_k_mask = _batch_take_last(k_mask, safe_cols)
    active_mask = target_mask & q_mask & matched_k_mask
    return safe_cols, active_mask


def sinkhorn_outputs(
    Q,
    K,
    V,
    q_mask,
    k_mask,
    n_iters,
    band_width,
    gradient_mode,
    refine_steps,
    refine_tolerance,
    epsilon,
    epsilon_scaling_steps,
    epsilon_scaling_factor,
    support_reconstruction,
):
    return ot_kernel.sinkhorn_attention(
        Q,
        K,
        V,
        n_iters=n_iters,
        band_width=band_width,
        q_mask=q_mask,
        k_mask=k_mask,
        refine_steps=refine_steps,
        refine_tolerance=refine_tolerance,
        epsilon=epsilon,
        epsilon_scaling_steps=epsilon_scaling_steps,
        epsilon_scaling_factor=epsilon_scaling_factor,
        gradient_mode=gradient_mode,
        support_reconstruction=support_reconstruction,
    )


def sinkhorn_alignment_barycenters(
    Q,
    K,
    q_mask,
    k_mask,
    n_iters,
    band_width,
    epsilon,
    epsilon_scaling_steps,
    epsilon_scaling_factor,
    support_reconstruction,
):
    scale = 1.0 / (jnp.sqrt(Q.shape[-1]) * epsilon)
    u, v = ot_kernel.compute_global_uv(
        Q,
        K,
        n_iters=n_iters,
        band_width=band_width,
        q_mask=q_mask,
        k_mask=k_mask,
        epsilon=epsilon,
        epsilon_scaling_steps=epsilon_scaling_steps,
        epsilon_scaling_factor=epsilon_scaling_factor,
        support_reconstruction=support_reconstruction,
    )
    logits = jnp.einsum("bhmd,bhnd->bhmn", Q, K) * scale
    logits = logits + u[..., :, None] + v[..., None, :]
    logits = jnp.where(k_mask[:, None, None, :], logits, -jnp.inf)
    band = ot_kernel.materialize_support_mask(
        Q, K, band_width=band_width, q_mask=q_mask, k_mask=k_mask, support_reconstruction=support_reconstruction
    )
    if band is not None:
        logits = jnp.where(band[None, None, :, :], logits, -jnp.inf)
    weights = jnp.where(q_mask[:, None, :, None], jnp.exp(logits), 0.0)
    col_idx = jnp.arange(K.shape[-2], dtype=Q.dtype)
    numer = jnp.sum(weights * col_idx[None, None, None, :], axis=-1)
    denom = jnp.maximum(jnp.sum(weights, axis=-1), 1e-8)
    return jnp.where(q_mask[:, None, :], numer / denom, 0.0)


def monotonicity_violation_metric(barycenters, active_mask):
    pair_mask = active_mask[:, 1:] & active_mask[:, :-1]
    deltas = barycenters[..., 1:] - barycenters[..., :-1]
    violation = jnp.maximum(-deltas, 0.0)
    denom = jnp.maximum(jnp.sum(pair_mask) * barycenters.shape[1], 1)
    return jnp.sum(jnp.where(pair_mask[:, None, :], violation, 0.0)) / denom


def sparse_ce_metric(
    Q,
    K,
    target_cols,
    target_mask,
    q_mask,
    k_mask,
    n_iters,
    band_width,
    epsilon,
    epsilon_scaling_steps,
    epsilon_scaling_factor,
    support_reconstruction,
):
    u, v = ot_kernel.compute_global_uv(
        Q,
        K,
        n_iters=n_iters,
        band_width=band_width,
        q_mask=q_mask,
        k_mask=k_mask,
        epsilon=epsilon,
        epsilon_scaling_steps=epsilon_scaling_steps,
        epsilon_scaling_factor=epsilon_scaling_factor,
        support_reconstruction=support_reconstruction,
    )
    scale = 1.0 / (jnp.sqrt(Q.shape[-1]) * epsilon)
    safe_cols, active_mask = _prepare_sparse_supervision(target_cols, target_mask, q_mask, k_mask)
    matched_k = _batch_take_rows(K, safe_cols)
    matched_v = _batch_take_last(v, safe_cols)
    log_pos = jnp.sum(Q * matched_k, axis=-1) * scale + u + matched_v
    if band_width is not None:
        band = ot_kernel.materialize_support_mask(
            Q, K, band_width=band_width, q_mask=q_mask, k_mask=k_mask, support_reconstruction=support_reconstruction
        )
        in_band = jnp.take_along_axis(
            jnp.broadcast_to(band[None, :, :], (Q.shape[0],) + band.shape), safe_cols[..., None], axis=-1
        )[..., 0]
        log_pos = jnp.where(in_band[:, None, :], log_pos, -1e9)
    denom = jnp.maximum(jnp.sum(active_mask) * Q.shape[1], 1)
    return -jnp.sum(jnp.where(active_mask[:, None, :], log_pos, 0.0)) / denom


def _alignment_loss_terms(
    Q,
    K,
    V,
    target_cols,
    target_mask,
    q_mask,
    k_mask,
    n_iters,
    band_width,
    sinkhorn_gradient_mode,
    sinkhorn_refine_steps,
    sinkhorn_refine_tolerance,
    sinkhorn_epsilon,
    sinkhorn_epsilon_scaling_steps,
    sinkhorn_epsilon_scaling_factor,
    support_reconstruction,
):
    safe_cols, active_mask = _prepare_sparse_supervision(target_cols, target_mask, q_mask, k_mask)
    target_v = _batch_take_rows(V, safe_cols)
    outputs = sinkhorn_outputs(
        Q,
        K,
        V,
        q_mask=q_mask,
        k_mask=k_mask,
        n_iters=n_iters,
        band_width=band_width,
        gradient_mode=sinkhorn_gradient_mode,
        refine_steps=sinkhorn_refine_steps,
        refine_tolerance=sinkhorn_refine_tolerance,
        epsilon=sinkhorn_epsilon,
        epsilon_scaling_steps=sinkhorn_epsilon_scaling_steps,
        epsilon_scaling_factor=sinkhorn_epsilon_scaling_factor,
        support_reconstruction=support_reconstruction,
    )
    ce_loss = sparse_ce_metric(
        Q,
        K,
        target_cols,
        target_mask,
        q_mask,
        k_mask,
        n_iters,
        band_width,
        sinkhorn_epsilon,
        sinkhorn_epsilon_scaling_steps,
        sinkhorn_epsilon_scaling_factor,
        support_reconstruction,
    )
    barycenters = sinkhorn_alignment_barycenters(
        Q,
        K,
        q_mask,
        k_mask,
        n_iters,
        band_width,
        sinkhorn_epsilon,
        sinkhorn_epsilon_scaling_steps,
        sinkhorn_epsilon_scaling_factor,
        support_reconstruction,
    )
    sq_error = jnp.sum(jnp.square(outputs - target_v), axis=-1)
    denom = jnp.maximum(jnp.sum(active_mask) * Q.shape[1], 1)
    reconstruction_loss = jnp.sum(jnp.where(active_mask[:, None, :], sq_error, 0.0)) / denom
    monotonicity_loss = monotonicity_violation_metric(barycenters, target_mask)
    return {"reconstruction": reconstruction_loss, "sparse_ce": ce_loss, "monotonicity": monotonicity_loss}


def output_reconstruction_loss(
    Q,
    K,
    V,
    target_cols,
    target_mask,
    q_mask,
    k_mask,
    n_iters,
    band_width,
    sinkhorn_gradient_mode,
    sinkhorn_refine_steps,
    sinkhorn_refine_tolerance,
    sinkhorn_epsilon,
    sinkhorn_epsilon_scaling_steps,
    sinkhorn_epsilon_scaling_factor,
    support_reconstruction,
    train_reconstruction_weight,
    train_sparse_ce_weight,
    train_monotonicity_weight,
    gap_mode,
    params,
):
    Q, K, V, target_cols, target_mask, q_mask, k_mask = _apply_gap_mode(
        params, Q, K, V, target_cols, target_mask, q_mask, k_mask, gap_mode
    )
    band_width = _effective_band_width(band_width, gap_mode, target_cols.shape[-1] - DUSTBIN_BLOCK_SIZE)
    loss_terms = _alignment_loss_terms(
        Q,
        K,
        V,
        target_cols,
        target_mask,
        q_mask,
        k_mask,
        n_iters,
        band_width,
        sinkhorn_gradient_mode,
        sinkhorn_refine_steps,
        sinkhorn_refine_tolerance,
        sinkhorn_epsilon,
        sinkhorn_epsilon_scaling_steps,
        sinkhorn_epsilon_scaling_factor,
        support_reconstruction,
    )
    return (
        train_reconstruction_weight * loss_terms["reconstruction"]
        + train_sparse_ce_weight * loss_terms["sparse_ce"]
        + train_monotonicity_weight * loss_terms["monotonicity"]
    )


def build_train_step(
    optimizer,
    n_iters,
    band_width,
    sinkhorn_gradient_mode,
    sinkhorn_refine_steps,
    sinkhorn_refine_tolerance,
    sinkhorn_epsilon,
    sinkhorn_epsilon_scaling_steps,
    sinkhorn_epsilon_scaling_factor,
    support_reconstruction,
    train_reconstruction_weight,
    train_sparse_ce_weight,
    train_monotonicity_weight,
    projection_manifold,
    gap_mode,
):
    @functools.partial(jax.pmap, axis_name="devices", donate_argnums=(0, 1))
    def train_step(params, opt_state, emb_q, emb_k, target_cols, target_mask, q_mask, k_mask):
        def loss_fn(model_params):
            Q, K, V = _project_qkv(model_params, emb_q, emb_k)
            return output_reconstruction_loss(
                Q,
                K,
                V,
                target_cols,
                target_mask,
                q_mask,
                k_mask,
                n_iters,
                band_width,
                sinkhorn_gradient_mode,
                sinkhorn_refine_steps,
                sinkhorn_refine_tolerance,
                sinkhorn_epsilon,
                sinkhorn_epsilon_scaling_steps,
                sinkhorn_epsilon_scaling_factor,
                support_reconstruction,
                train_reconstruction_weight,
                train_sparse_ce_weight,
                train_monotonicity_weight,
                gap_mode,
                model_params,
            )

        loss, grads = jax.value_and_grad(loss_fn)(params)
        loss = jax.lax.pmean(loss, axis_name="devices")
        grads = jax.lax.pmean(grads, axis_name="devices")
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        next_params = apply_projection_manifold(next_params, projection_manifold)
        return next_params, next_opt_state, loss

    return train_step


def build_metric_step(
    n_iters,
    band_width,
    gap_mode,
    sinkhorn_epsilon,
    sinkhorn_epsilon_scaling_steps,
    sinkhorn_epsilon_scaling_factor,
    support_reconstruction,
):
    @functools.partial(jax.pmap, axis_name="devices")
    def metric_step(params, emb_q, emb_k, target_cols, target_mask, q_mask, k_mask):
        Q, K, V = _project_qkv(params, emb_q, emb_k)
        Q, K, V, target_cols, target_mask, q_mask, k_mask = _apply_gap_mode(
            params, Q, K, V, target_cols, target_mask, q_mask, k_mask, gap_mode
        )
        band_width_eff = _effective_band_width(band_width, gap_mode, target_cols.shape[-1] - DUSTBIN_BLOCK_SIZE)
        ce_metric = sparse_ce_metric(
            Q,
            K,
            target_cols,
            target_mask,
            q_mask,
            k_mask,
            n_iters,
            band_width_eff,
            sinkhorn_epsilon,
            sinkhorn_epsilon_scaling_steps,
            sinkhorn_epsilon_scaling_factor,
            support_reconstruction,
        )
        barycenters = sinkhorn_alignment_barycenters(
            Q,
            K,
            q_mask,
            k_mask,
            n_iters,
            band_width_eff,
            sinkhorn_epsilon,
            sinkhorn_epsilon_scaling_steps,
            sinkhorn_epsilon_scaling_factor,
            support_reconstruction,
        )
        mono_metric = monotonicity_violation_metric(barycenters, target_mask)
        return {
            "ce": jax.lax.pmean(ce_metric, axis_name="devices"),
            "mono": jax.lax.pmean(mono_metric, axis_name="devices"),
        }

    return metric_step


def reshape_for_pmap(x, local_device_count):
    return x.reshape(local_device_count, -1, *x.shape[1:])


def _broadcast_stop_flag(stop_flag):
    arr = np.array(stop_flag, dtype=np.bool_)
    out = multihost_utils.broadcast_one_to_all(arr, is_source=jax.process_index() == 0)
    return bool(out)


def _load_batch_size_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("per_device_batch_size") if isinstance(data, dict) else None
    return value if isinstance(value, int) and value > 0 else None


def _write_batch_size_file(path, per_device_batch_size):
    tmp_path = f"{path}.tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"per_device_batch_size": int(per_device_batch_size)}, f)
    os.replace(tmp_path, path)


def autotune_per_device_batch_size(
    optimizer,
    params,
    opt_state,
    ds,
    n_iters,
    band_width,
    sinkhorn_gradient_mode,
    sinkhorn_refine_steps,
    sinkhorn_refine_tolerance,
    sinkhorn_epsilon,
    sinkhorn_epsilon_scaling_steps,
    sinkhorn_epsilon_scaling_factor,
    support_reconstruction,
    train_reconstruction_weight,
    train_sparse_ce_weight,
    train_monotonicity_weight,
    projection_manifold,
    gap_mode,
    max_per_device_batch_size,
):
    local_device_count = jax.local_device_count()
    devices = jax.local_devices()
    last_error = None

    def try_size(per_device_batch_size):
        nonlocal last_error
        local_batch_size = int(per_device_batch_size) * local_device_count
        batch = ds.sample_batch(batch_size=local_batch_size)
        emb_q, emb_k, target_cols, target_mask, q_mask, k_mask = batch
        try:
            train_step = build_train_step(
                optimizer,
                n_iters,
                band_width,
                sinkhorn_gradient_mode,
                sinkhorn_refine_steps,
                sinkhorn_refine_tolerance,
                sinkhorn_epsilon,
                sinkhorn_epsilon_scaling_steps,
                sinkhorn_epsilon_scaling_factor,
                support_reconstruction,
                train_reconstruction_weight,
                train_sparse_ce_weight,
                train_monotonicity_weight,
                projection_manifold,
                gap_mode,
            )
            params_r = _replicate_tree(params, devices)
            opt_state_r = _replicate_tree(opt_state, devices)
            _, _, loss = train_step(
                params_r,
                opt_state_r,
                reshape_for_pmap(emb_q, local_device_count),
                reshape_for_pmap(emb_k, local_device_count),
                reshape_for_pmap(target_cols, local_device_count),
                reshape_for_pmap(target_mask, local_device_count),
                reshape_for_pmap(q_mask, local_device_count),
                reshape_for_pmap(k_mask, local_device_count),
            )
            jax.block_until_ready(loss)
            return True
        except Exception as exc:
            last_error = exc
            return False

    best = 0
    candidate = 1
    while candidate <= int(max_per_device_batch_size):
        ok = _broadcast_stop_flag(try_size(candidate))
        if not ok:
            break
        best = candidate
        candidate *= 2
    low = best
    high = min(int(max_per_device_batch_size), candidate - 1)
    while low < high:
        mid = (low + high + 1) // 2
        ok = _broadcast_stop_flag(try_size(mid))
        if ok:
            low = mid
        else:
            high = mid - 1
    if best == 0:
        raise RuntimeError("Autotune failed at per_device_batch_size=1.") from last_error
    return low


def train_pfam(
    max_hours,
    data_dir,
    ckpt_dir,
    band_width=1024,
    n_iters=15,
    per_device_batch_size=8,
    batch_size_file=None,
    autotune_batch_size=False,
    autotune_max_per_device_batch_size=8,
    checkpoint_every=250,
    seed=42,
    metric_every=50,
    sinkhorn_gradient_mode="tail_refinement",
    sinkhorn_refine_steps=2,
    sinkhorn_refine_tolerance=None,
    sinkhorn_epsilon=1.0,
    sinkhorn_epsilon_scaling_steps=0,
    sinkhorn_epsilon_scaling_factor=2.0,
    support_reconstruction="none",
    train_reconstruction_weight=1.0,
    train_sparse_ce_weight=0.0,
    train_monotonicity_weight=0.0,
    projection_manifold="stiefel_qk",
    gap_mode="none",
):
    process_index = jax.process_index()
    process_count = jax.process_count()
    local_device_count = jax.local_device_count()
    global_device_count = jax.device_count()
    is_primary = process_index == 0
    if sinkhorn_epsilon <= 0:
        raise ValueError("sinkhorn_epsilon must be positive.")
    if train_reconstruction_weight + train_sparse_ce_weight + train_monotonicity_weight <= 0:
        raise ValueError("At least one loss weight must be positive.")
    if support_reconstruction not in ("none", "affine_scout", "transport_scout"):
        raise ValueError(f"Unsupported support_reconstruction={support_reconstruction!r}.")
    if gap_mode not in ("none", "dustbin_block"):
        raise ValueError(f"Unsupported gap_mode={gap_mode!r}.")

    ds = PfamShardDataset(data_dir, seed=seed + process_index * 1000)
    d_in, d_proj = 320, 64
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    params = {
        "W_q": jax.random.normal(k1, (d_in, d_proj), dtype=jnp.float32) * (1.0 / jnp.sqrt(d_in)),
        "W_k": jax.random.normal(k2, (d_in, d_proj), dtype=jnp.float32) * (1.0 / jnp.sqrt(d_in)),
        "W_v": jax.random.normal(k3, (d_in, d_proj), dtype=jnp.float32) * (1.0 / jnp.sqrt(d_in)),
        "gap_q": jax.random.normal(k4, (d_proj,), dtype=jnp.float32) * (1.0 / jnp.sqrt(d_proj)),
        "gap_k": jax.random.normal(k5, (d_proj,), dtype=jnp.float32) * (1.0 / jnp.sqrt(d_proj)),
        "gap_v": jax.random.normal(k6, (d_proj,), dtype=jnp.float32) * (1.0 / jnp.sqrt(d_proj)),
    }
    params = apply_projection_manifold(params, projection_manifold)
    optimizer = muon_optimizer(3e-4)
    opt_state = optimizer.init(params)
    start_step = 0
    ckpt_data = load_checkpoint(ckpt_dir)
    if ckpt_data is not None:
        loaded_params, opt_state, start_step = ckpt_data
        params = {**params, **loaded_params}

    if batch_size_file and not autotune_batch_size:
        loaded = _load_batch_size_file(batch_size_file)
        loaded = 0 if loaded is None else int(loaded)
        loaded = int(multihost_utils.broadcast_one_to_all(np.array(loaded, dtype=np.int32), is_source=is_primary))
        if loaded > 0:
            per_device_batch_size = loaded

    if autotune_batch_size:
        tuned = autotune_per_device_batch_size(
            optimizer,
            params,
            opt_state,
            ds,
            n_iters,
            band_width,
            sinkhorn_gradient_mode,
            sinkhorn_refine_steps,
            sinkhorn_refine_tolerance,
            sinkhorn_epsilon,
            sinkhorn_epsilon_scaling_steps,
            sinkhorn_epsilon_scaling_factor,
            support_reconstruction,
            train_reconstruction_weight,
            train_sparse_ce_weight,
            train_monotonicity_weight,
            projection_manifold,
            gap_mode,
            autotune_max_per_device_batch_size,
        )
        tuned = int(multihost_utils.broadcast_one_to_all(np.array(tuned, dtype=np.int32), is_source=is_primary))
        per_device_batch_size = max(1, tuned)
        if is_primary and batch_size_file:
            _write_batch_size_file(batch_size_file, per_device_batch_size)

    local_batch_size = int(per_device_batch_size) * local_device_count
    global_batch_size = int(per_device_batch_size) * global_device_count
    devices = jax.local_devices()
    params = _replicate_tree(params, devices)
    opt_state = _replicate_tree(opt_state, devices)
    train_step = build_train_step(
        optimizer,
        n_iters,
        band_width,
        sinkhorn_gradient_mode,
        sinkhorn_refine_steps,
        sinkhorn_refine_tolerance,
        sinkhorn_epsilon,
        sinkhorn_epsilon_scaling_steps,
        sinkhorn_epsilon_scaling_factor,
        support_reconstruction,
        train_reconstruction_weight,
        train_sparse_ce_weight,
        train_monotonicity_weight,
        projection_manifold,
        gap_mode,
    )
    metric_step = (
        build_metric_step(
            n_iters,
            band_width,
            gap_mode,
            sinkhorn_epsilon,
            sinkhorn_epsilon_scaling_steps,
            sinkhorn_epsilon_scaling_factor,
            support_reconstruction,
        )
        if metric_every and metric_every > 0
        else None
    )

    if is_primary:
        print("Sinkhorn Pfam training")
        print("  expected accelerator: Cloud TPU v6e-8")
        print(f"  pairs: {ds.metadata['total_pairs']}")
        print(f"  local devices: {local_device_count}")
        print(f"  global devices: {global_device_count}")
        print(f"  processes: {process_count}")
        print(f"  per-device batch: {per_device_batch_size}")
        print(f"  global batch: {global_batch_size}")
        print(f"  max hours: {max_hours}")
        print(f"  checkpoint every: {checkpoint_every}")
        print(f"  band width: {band_width}")
        print(f"  base iterations: {n_iters}")
        print(f"  gradient mode: {sinkhorn_gradient_mode}")
        print(f"  refine steps: {sinkhorn_refine_steps}")
        print(f"  epsilon: {sinkhorn_epsilon}")
        print(f"  epsilon scaling steps: {sinkhorn_epsilon_scaling_steps}")
        print(f"  epsilon scaling factor: {sinkhorn_epsilon_scaling_factor}")
        print(f"  support reconstruction: {support_reconstruction}")
        print(f"  reconstruction weight: {train_reconstruction_weight}")
        print(f"  sparse CE weight: {train_sparse_ce_weight}")
        print(f"  monotonicity weight: {train_monotonicity_weight}")
        print(f"  projection manifold: {projection_manifold}")
        print(f"  gap mode: {gap_mode}")
        print(f"  metric every: {metric_every}")
        print(f"  resume step: {start_step}")

    start_time = time.perf_counter()
    loss = None
    metric = None
    step = start_step
    while True:
        emb_q, emb_k, target_cols, target_mask, q_mask, k_mask = ds.sample_batch(batch_size=local_batch_size)
        emb_q = reshape_for_pmap(emb_q, local_device_count)
        emb_k = reshape_for_pmap(emb_k, local_device_count)
        target_cols = reshape_for_pmap(target_cols, local_device_count)
        target_mask = reshape_for_pmap(target_mask, local_device_count)
        q_mask = reshape_for_pmap(q_mask, local_device_count)
        k_mask = reshape_for_pmap(k_mask, local_device_count)
        params, opt_state, loss = train_step(
            params, opt_state, emb_q, emb_k, target_cols, target_mask, q_mask, k_mask
        )
        if metric_step is not None and step % int(metric_every) == 0:
            metric = metric_step(params, emb_q, emb_k, target_cols, target_mask, q_mask, k_mask)
        if step % 50 == 0:
            jax.block_until_ready(loss)
            if is_primary:
                elapsed = time.perf_counter() - start_time
                exps = ((step - start_step + 1) * global_batch_size) / max(elapsed, 1e-6)
                loss_value = float(jax.device_get(loss[0]))
                ce_display = "off"
                mono_display = "off"
                if metric is not None:
                    ce_display = f"{float(jax.device_get(metric['ce'][0])):.4f}"
                    mono_display = f"{float(jax.device_get(metric['mono'][0])):.4f}"
                print(
                    f"step={step} loss={loss_value:.4f} ce={ce_display} mono={mono_display} "
                    f"elapsed_sec={int(elapsed)} examples_per_second={exps:.2f}"
                )
        if step > 0 and step % int(checkpoint_every) == 0:
            jax.block_until_ready(loss)
            if is_primary:
                save_checkpoint(params, opt_state, step, ckpt_dir, replicated=True)
        should_stop = False
        if is_primary:
            should_stop = (time.perf_counter() - start_time) >= (float(max_hours) * 3600.0)
        should_stop = _broadcast_stop_flag(should_stop)
        if should_stop:
            break
        step += 1

    if loss is not None:
        jax.block_until_ready(loss)
    if is_primary:
        save_checkpoint(params, opt_state, step, ckpt_dir, replicated=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_hours", type=float, default=3.0)
    parser.add_argument("--data_dir", default="./data/pfam_shards")
    parser.add_argument("--ckpt_dir", default="./checkpoints/sinkhorn_ot")
    parser.add_argument("--band_width", type=int, default=1024)
    parser.add_argument("--n_iters", type=int, default=15)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--batch_size_file")
    parser.add_argument("--autotune_batch_size", action="store_true")
    parser.add_argument("--autotune_max_per_device_batch_size", type=int, default=8)
    parser.add_argument("--checkpoint_every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric_every", type=int, default=50)
    parser.add_argument("--sinkhorn_gradient_mode", choices=["tail_refinement", "exact_autodiff"], default="tail_refinement")
    parser.add_argument("--sinkhorn_refine_steps", type=int, default=2)
    parser.add_argument("--sinkhorn_refine_tolerance", type=float)
    parser.add_argument("--sinkhorn_epsilon", type=float, default=1.0)
    parser.add_argument("--sinkhorn_epsilon_scaling_steps", type=int, default=0)
    parser.add_argument("--sinkhorn_epsilon_scaling_factor", type=float, default=2.0)
    parser.add_argument("--support_reconstruction", choices=["none", "affine_scout", "transport_scout"], default="none")
    parser.add_argument("--train_reconstruction_weight", type=float, default=1.0)
    parser.add_argument("--train_sparse_ce_weight", type=float, default=0.0)
    parser.add_argument("--train_monotonicity_weight", type=float, default=0.0)
    parser.add_argument("--projection_manifold", choices=["none", "stiefel_qk"], default="stiefel_qk")
    parser.add_argument("--gap_mode", choices=["none", "dustbin_block"], default="none")
    args = parser.parse_args()
    train_pfam(**vars(args))


if __name__ == "__main__":
    main()
