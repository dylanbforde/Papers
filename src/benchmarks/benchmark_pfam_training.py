"""
Pfam Protein Alignment Benchmark
================================

This training benchmark is tuned for TPU slices where cost is dominated by
wall-clock, not code elegance. The implementation avoids dense LxL supervision
on the critical path and uses all visible TPU devices via pmap.
"""

import argparse
import functools
import json
import os
import pickle
import subprocess
import sys
import time
import warnings

import numpy as np

import jax
import jax.numpy as jnp
import optax
from jax.experimental import multihost_utils


if "MEGATRON_WORKER" in os.environ or (
    "TPU_WORKER_HOSTNAMES" in os.environ
    and len(os.environ["TPU_WORKER_HOSTNAMES"].split(",")) > 1
):
    try:
        jax.distributed.initialize()
    except Exception:
        pass


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ot_kernel


DUSTBIN_BLOCK_SIZE = 128


class PfamShardDataset:
    def __init__(self, data_dir, seed=0):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, "metadata.json"), encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.total_shards = self.metadata["total_shards"]
        self.shard_size = self.metadata["shard_size"]
        self.curr_shard_idx = -1
        self.curr_shard_data = None
        self.rng = np.random.default_rng(seed)

    def _load_shard(self, shard_idx):
        if self.curr_shard_idx == shard_idx:
            return

        path = os.path.join(self.data_dir, f"shard_{shard_idx:03d}.npz")
        with np.load(path, allow_pickle=False) as shard:
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

        num_in_shard = self.curr_shard_data["emb1"].shape[0]
        replace = batch_size > num_in_shard
        indices = self.rng.choice(num_in_shard, batch_size, replace=replace)

        emb1 = self.curr_shard_data["emb1"][indices][:, None, :, :]
        emb2 = self.curr_shard_data["emb2"][indices][:, None, :, :]
        target_cols = self.curr_shard_data["target_cols"][indices]
        target_mask = self.curr_shard_data["target_mask"][indices]
        q_mask = self.curr_shard_data["q_mask"][indices]
        k_mask = self.curr_shard_data["k_mask"][indices]
        return emb1, emb2, target_cols, target_mask, q_mask, k_mask


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
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(grads, state, params=None):
        del params
        updates = jax.tree_util.tree_map(muzzle_orthogonalize, grads)
        updates = jax.tree_util.tree_map(lambda x: -lr * x, updates)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def stiefel_qr_retraction(matrix):
    """Retract a tall matrix onto the Stiefel manifold with orthonormal columns."""
    q, r = jnp.linalg.qr(matrix, mode="reduced")
    phase = jnp.sign(jnp.diag(r))
    phase = jnp.where(phase == 0, 1.0, phase).astype(q.dtype)
    return q * phase[None, :]


def apply_projection_manifold(params, projection_manifold):
    if projection_manifold == "none":
        return params
    if projection_manifold == "stiefel_qk":
        return {
            **params,
            "W_q": stiefel_qr_retraction(params["W_q"]),
            "W_k": stiefel_qr_retraction(params["W_k"]),
        }
    raise ValueError(f"Unsupported projection_manifold={projection_manifold!r}.")


def _gsutil_cp(src, dst):
    try:
        result = subprocess.run(
            ["gsutil", "cp", src, dst],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return result.returncode == 0


def _unreplicate_tree(tree):
    return jax.tree_util.tree_map(lambda x: x[0], tree)

def _replicate_tree(tree, devices):
    # Keep compatibility across JAX versions; the newer sharding APIs are in flux.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        return jax.device_put_replicated(tree, devices)


def save_checkpoint(params, opt_state, step, local_dir, gcs_dir=None, replicated=False, verbose=True):
    os.makedirs(local_dir, exist_ok=True)
    path = os.path.join(local_dir, "ckpt.pkl")

    if replicated:
        params = _unreplicate_tree(params)
        opt_state = _unreplicate_tree(opt_state)

    payload = {
        "params": jax.device_get(params),
        "opt_state": jax.device_get(opt_state),
        "step": step,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print(f"Checkpoint saved at step {step} -> {path}")
    if gcs_dir:
        gcs_path = os.path.join(gcs_dir, "ckpt.pkl")
        if _gsutil_cp(path, gcs_path):
            if verbose:
                print(f"Checkpoint synced to {gcs_path}")
        elif verbose:
            print(f"Warning: failed to sync checkpoint to {gcs_path}")


def load_checkpoint(local_dir, gcs_dir=None, verbose=True):
    os.makedirs(local_dir, exist_ok=True)
    path = os.path.join(local_dir, "ckpt.pkl")

    if gcs_dir:
        gcs_path = os.path.join(gcs_dir, "ckpt.pkl")
        _gsutil_cp(gcs_path, path)

    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        data = pickle.load(f)
    if verbose:
        print(f"Resuming from checkpoint at step {data['step']}")
    return data["params"], data["opt_state"], data["step"]


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
    batch, length = mask.shape
    del length
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
        [target_cols, jnp.zeros((target_cols.shape[0], DUSTBIN_BLOCK_SIZE), dtype=target_cols.dtype)],
        axis=-1,
    )
    target_mask = jnp.concatenate(
        [target_mask, jnp.zeros((target_mask.shape[0], DUSTBIN_BLOCK_SIZE), dtype=bool)],
        axis=-1,
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


def flash_attention_outputs(Q, K, V, q_mask, k_mask, band_width=None):
    scale = 1.0 / jnp.sqrt(Q.shape[-1])
    logits = jnp.einsum("bhmd,bhnd->bhmn", Q, K) * scale
    logits = jnp.where(k_mask[:, None, None, :], logits, -jnp.inf)
    if band_width is not None and int(band_width) < Q.shape[-2]:
        length = Q.shape[-2]
        row_idx = jnp.arange(length, dtype=jnp.int32)[:, None]
        col_idx = jnp.arange(length, dtype=jnp.int32)[None, :]
        band = jnp.abs(row_idx - col_idx) <= int(band_width)
        logits = jnp.where(band[None, None, :, :], logits, -jnp.inf)
    probs = jax.nn.softmax(logits, axis=-1)
    outputs = jnp.einsum("bhmn,bhnd->bhmd", probs, V)
    return jnp.where(q_mask[:, None, :, None], outputs, 0.0)


def _length_band_mask(length, band_width, dtype=jnp.bool_):
    if band_width is None or int(band_width) >= length:
        return None
    row_idx = jnp.arange(length, dtype=jnp.int32)[:, None]
    col_idx = jnp.arange(length, dtype=jnp.int32)[None, :]
    return (jnp.abs(row_idx - col_idx) <= int(band_width)).astype(dtype)


def flash_alignment_barycenters(Q, K, q_mask, k_mask, band_width):
    scale = 1.0 / jnp.sqrt(Q.shape[-1])
    logits = jnp.einsum("bhmd,bhnd->bhmn", Q, K) * scale
    logits = jnp.where(k_mask[:, None, None, :], logits, -jnp.inf)
    band = _length_band_mask(Q.shape[-2], band_width)
    if band is not None:
        logits = jnp.where(band[None, None, :, :], logits, -jnp.inf)
    probs = jax.nn.softmax(logits, axis=-1)
    col_idx = jnp.arange(K.shape[-2], dtype=Q.dtype)
    bary = jnp.sum(probs * col_idx[None, None, None, :], axis=-1)
    return jnp.where(q_mask[:, None, :], bary, 0.0)


def sinkhorn_alignment_barycenters(Q, K, q_mask, k_mask, n_iters, band_width):
    scale = 1.0 / jnp.sqrt(Q.shape[-1])
    u, v = ot_kernel.compute_global_uv(
        Q,
        K,
        n_iters=n_iters,
        band_width=band_width,
        q_mask=q_mask,
        k_mask=k_mask,
    )
    logits = jnp.einsum("bhmd,bhnd->bhmn", Q, K) * scale
    logits = logits + u[..., :, None] + v[..., None, :]
    logits = jnp.where(k_mask[:, None, None, :], logits, -jnp.inf)
    band = _length_band_mask(Q.shape[-2], band_width)
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
        gradient_mode=gradient_mode,
    )


def output_reconstruction_loss(
    method,
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
    gap_mode,
    params,
):
    Q, K, V, target_cols, target_mask, q_mask, k_mask = _apply_gap_mode(
        params,
        Q,
        K,
        V,
        target_cols,
        target_mask,
        q_mask,
        k_mask,
        gap_mode,
    )
    band_width = _effective_band_width(band_width, gap_mode, target_cols.shape[-1] - DUSTBIN_BLOCK_SIZE)
    safe_cols, active_mask = _prepare_sparse_supervision(target_cols, target_mask, q_mask, k_mask)
    target_v = _batch_take_rows(V, safe_cols)
    if method == "sinkhorn_ot":
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
        )
    else:
        outputs = flash_attention_outputs(Q, K, V, q_mask, k_mask, band_width=band_width)
    sq_error = jnp.sum(jnp.square(outputs - target_v), axis=-1)
    denom = jnp.maximum(jnp.sum(active_mask) * Q.shape[1], 1)
    return jnp.sum(jnp.where(active_mask[:, None, :], sq_error, 0.0)) / denom


def sparse_ce_metric(Q, K, target_cols, target_mask, q_mask, k_mask, n_iters, band_width):
    # Exact sparse CE metric computed from masked potentials; expensive, optional.
    u, v = ot_kernel.compute_global_uv(
        Q,
        K,
        n_iters=n_iters,
        band_width=band_width,
        q_mask=q_mask,
        k_mask=k_mask,
    )

    scale = 1.0 / jnp.sqrt(Q.shape[-1])
    safe_cols, active_mask = _prepare_sparse_supervision(target_cols, target_mask, q_mask, k_mask)
    matched_k = _batch_take_rows(K, safe_cols)
    matched_v = _batch_take_last(v, safe_cols)
    log_pos = jnp.sum(Q * matched_k, axis=-1) * scale + u + matched_v

    if band_width is not None:
        row_idx = jnp.arange(Q.shape[-2], dtype=safe_cols.dtype)[None, :]
        in_band = jnp.abs(row_idx - safe_cols) <= int(band_width)
        log_pos = jnp.where(in_band[:, None, :], log_pos, -1e9)

    denom = jnp.maximum(jnp.sum(active_mask) * Q.shape[1], 1)
    return -jnp.sum(jnp.where(active_mask[:, None, :], log_pos, 0.0)) / denom


def flash_sparse_ce_metric(Q, K, target_cols, target_mask, q_mask, k_mask, band_width):
    scale = 1.0 / jnp.sqrt(Q.shape[-1])
    logits = jnp.einsum("bhmd,bhnd->bhmn", Q, K) * scale
    logits = jnp.where(k_mask[:, None, None, :], logits, -jnp.inf)
    band = _length_band_mask(Q.shape[-2], band_width)
    if band is not None:
        logits = jnp.where(band[None, None, :, :], logits, -jnp.inf)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    safe_cols, active_mask = _prepare_sparse_supervision(target_cols, target_mask, q_mask, k_mask)
    matched_log_probs = _rowwise_take_cols(log_probs, safe_cols)
    denom = jnp.maximum(jnp.sum(active_mask) * Q.shape[1], 1)
    return -jnp.sum(jnp.where(active_mask[:, None, :], matched_log_probs, 0.0)) / denom


def build_train_step(
    method,
    optimizer,
    n_iters,
    band_width,
    sinkhorn_gradient_mode,
    sinkhorn_refine_steps,
    sinkhorn_refine_tolerance,
    projection_manifold,
    gap_mode,
):
    @functools.partial(jax.pmap, axis_name="devices", donate_argnums=(0, 1))
    def train_step(params, opt_state, emb_q, emb_k, target_cols, target_mask, q_mask, k_mask):
        def loss_fn(model_params):
            Q, K, V = _project_qkv(model_params, emb_q, emb_k)
            return output_reconstruction_loss(
                method,
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
                gap_mode,
                model_params,
            )

        loss, grads = jax.value_and_grad(loss_fn)(params)
        loss = jax.lax.pmean(loss, axis_name="devices")
        grads = jax.lax.pmean(grads, axis_name="devices")
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        next_params = jax.tree_util.tree_map(
            lambda x: x,
            apply_projection_manifold(next_params, projection_manifold),
        )
        return next_params, next_opt_state, loss

    return train_step


def build_metric_step(method, n_iters, band_width, gap_mode):
    @functools.partial(jax.pmap, axis_name="devices")
    def metric_step(params, emb_q, emb_k, target_cols, target_mask, q_mask, k_mask):
        Q, K, V = _project_qkv(params, emb_q, emb_k)
        Q, K, V, target_cols, target_mask, q_mask, k_mask = _apply_gap_mode(
            params,
            Q,
            K,
            V,
            target_cols,
            target_mask,
            q_mask,
            k_mask,
            gap_mode,
        )
        band_width_eff = _effective_band_width(band_width, gap_mode, target_cols.shape[-1] - DUSTBIN_BLOCK_SIZE)
        if method == "sinkhorn_ot":
            ce_metric = sparse_ce_metric(Q, K, target_cols, target_mask, q_mask, k_mask, n_iters, band_width_eff)
            barycenters = sinkhorn_alignment_barycenters(Q, K, q_mask, k_mask, n_iters, band_width_eff)
        else:
            ce_metric = flash_sparse_ce_metric(Q, K, target_cols, target_mask, q_mask, k_mask, band_width_eff)
            barycenters = flash_alignment_barycenters(Q, K, q_mask, k_mask, band_width_eff)
        mono_metric = monotonicity_violation_metric(barycenters, target_mask)
        metrics = {
            "ce": jax.lax.pmean(ce_metric, axis_name="devices"),
            "mono": jax.lax.pmean(mono_metric, axis_name="devices"),
        }
        return metrics

    return metric_step


def reshape_for_pmap(x, local_device_count):
    return x.reshape(local_device_count, -1, *x.shape[1:])


def _broadcast_stop_flag(stop_flag: bool) -> bool:
    # Prevents multi-host collective hangs by ensuring all processes stop together.
    arr = np.array(stop_flag, dtype=np.bool_)
    out = multihost_utils.broadcast_one_to_all(arr, is_source=jax.process_index() == 0)
    return bool(out)


def _load_batch_size_file(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError:
        return None
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("per_device_batch_size")
    if isinstance(value, int) and value > 0:
        return value
    return None


def _write_batch_size_file(path: str, per_device_batch_size: int):
    tmp_path = f"{path}.tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"per_device_batch_size": int(per_device_batch_size)}, f)
    os.replace(tmp_path, path)


def autotune_per_device_batch_size(
    method,
    optimizer,
    params,
    opt_state,
    ds,
    n_iters,
    band_width,
    sinkhorn_gradient_mode,
    sinkhorn_refine_steps,
    sinkhorn_refine_tolerance,
    projection_manifold,
    gap_mode,
    max_per_device_batch_size,
):
    # Budget-aware autotune: a handful of compile+run attempts at the start.
    local_device_count = jax.local_device_count()
    devices = jax.local_devices()

    def try_size(per_device_batch_size: int) -> bool:
        local_batch_size = per_device_batch_size * local_device_count
        batch = ds.sample_batch(batch_size=local_batch_size)
        emb_q, emb_k, target_cols, target_mask, q_mask, k_mask = batch
        try:
            emb_q_p = reshape_for_pmap(emb_q, local_device_count)
            emb_k_p = reshape_for_pmap(emb_k, local_device_count)
            target_cols_p = reshape_for_pmap(target_cols, local_device_count)
            target_mask_p = reshape_for_pmap(target_mask, local_device_count)
            q_mask_p = reshape_for_pmap(q_mask, local_device_count)
            k_mask_p = reshape_for_pmap(k_mask, local_device_count)

            params_r = _replicate_tree(params, devices)
            opt_state_r = _replicate_tree(opt_state, devices)
            train_step = build_train_step(
                method,
                optimizer,
                n_iters,
                band_width,
                sinkhorn_gradient_mode,
                sinkhorn_refine_steps,
                sinkhorn_refine_tolerance,
                projection_manifold,
                gap_mode,
            )
            _params_r, _opt_state_r, loss = train_step(
                params_r, opt_state_r, emb_q_p, emb_k_p, target_cols_p, target_mask_p, q_mask_p, k_mask_p
            )
            jax.block_until_ready(loss)
        except Exception:
            return False
        return True

    best = 1
    candidate = 1
    while candidate <= max_per_device_batch_size:
        ok = try_size(candidate)
        ok = _broadcast_stop_flag(ok)  # keep hosts in lockstep during probing
        if not ok:
            break
        best = candidate
        candidate *= 2

    low = best
    high = min(max_per_device_batch_size, candidate - 1)
    while low < high:
        mid = (low + high + 1) // 2
        ok = try_size(mid)
        ok = _broadcast_stop_flag(ok)
        if ok:
            low = mid
        else:
            high = mid - 1

    return low


def train_pfam(
    method,
    max_hours,
    data_dir,
    ckpt_dir,
    gcs_ckpt_dir=None,
    band_width=1024,
    n_iters=15,
    per_device_batch_size=1,
    batch_size_file=None,
    autotune_batch_size=False,
    autotune_max_per_device_batch_size=8,
    checkpoint_every=250,
    seed=42,
    metric_every=0,
    sinkhorn_gradient_mode="tail_refinement",
    sinkhorn_refine_steps=2,
    sinkhorn_refine_tolerance=None,
    projection_manifold="none",
    gap_mode="none",
):
    process_index = jax.process_index()
    process_count = jax.process_count()
    local_device_count = jax.local_device_count()
    global_device_count = jax.device_count()
    is_primary = process_index == 0

    ckpt_dir = os.path.join(ckpt_dir, method)
    if gcs_ckpt_dir:
        gcs_ckpt_dir = f"{gcs_ckpt_dir.rstrip('/')}/{method}"

    if per_device_batch_size <= 0 and not autotune_batch_size:
        raise ValueError("per_device_batch_size must be positive (or use --autotune_batch_size).")
    if sinkhorn_refine_steps < 0:
        raise ValueError("sinkhorn_refine_steps must be non-negative.")
    if sinkhorn_refine_tolerance is not None and sinkhorn_refine_tolerance <= 0:
        raise ValueError("sinkhorn_refine_tolerance must be positive when provided.")
    if gap_mode not in ("none", "dustbin_block"):
        raise ValueError(f"Unsupported gap_mode={gap_mode!r}.")

    local_batch_size = per_device_batch_size * local_device_count
    global_batch_size = per_device_batch_size * global_device_count

    ds = PfamShardDataset(data_dir, seed=seed + process_index * 1000)
    num_pairs = ds.metadata["total_pairs"]
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

    ckpt_data = load_checkpoint(ckpt_dir, gcs_ckpt_dir, verbose=is_primary)
    if ckpt_data is not None:
        loaded_params, opt_state, start_step = ckpt_data
        params = {**params, **loaded_params}

    if batch_size_file and not autotune_batch_size:
        loaded = _load_batch_size_file(batch_size_file)
        loaded = loaded if loaded is not None else 0
        loaded = int(
            multihost_utils.broadcast_one_to_all(np.array(loaded, dtype=np.int32), is_source=is_primary)
        )
        if loaded > 0:
            per_device_batch_size = loaded
            local_batch_size = per_device_batch_size * local_device_count
            global_batch_size = per_device_batch_size * global_device_count

    if autotune_batch_size:
        if is_primary:
            print(f"Autotuning per-device batch size (cap={autotune_max_per_device_batch_size})...")
        tuned = autotune_per_device_batch_size(
            method=method,
            optimizer=optimizer,
            params=params,
            opt_state=opt_state,
            ds=ds,
            n_iters=n_iters,
            band_width=band_width,
            sinkhorn_gradient_mode=sinkhorn_gradient_mode,
            sinkhorn_refine_steps=sinkhorn_refine_steps,
            sinkhorn_refine_tolerance=sinkhorn_refine_tolerance,
            projection_manifold=projection_manifold,
            gap_mode=gap_mode,
            max_per_device_batch_size=int(autotune_max_per_device_batch_size),
        )
        tuned = int(
            multihost_utils.broadcast_one_to_all(np.array(tuned, dtype=np.int32), is_source=is_primary)
        )
        per_device_batch_size = max(1, tuned)
        local_batch_size = per_device_batch_size * local_device_count
        global_batch_size = per_device_batch_size * global_device_count
        if is_primary:
            print(f"Autotune selected per_device_batch_size={per_device_batch_size}")
            if batch_size_file:
                _write_batch_size_file(batch_size_file, per_device_batch_size)

    devices = jax.local_devices()
    params = _replicate_tree(params, devices)
    opt_state = _replicate_tree(opt_state, devices)
    train_step = build_train_step(
        method,
        optimizer,
        n_iters,
        band_width,
        sinkhorn_gradient_mode,
        sinkhorn_refine_steps,
        sinkhorn_refine_tolerance,
        projection_manifold,
        gap_mode,
    )
    metric_step = build_metric_step(method, n_iters, band_width, gap_mode) if metric_every and metric_every > 0 else None

    if is_primary:
        print()
        print(f"Pfam training ({method})")
        print(f"  pairs: {num_pairs}")
        print(f"  local devices: {local_device_count}")
        print(f"  global devices: {global_device_count}")
        print(f"  processes: {process_count}")
        print(f"  per-device batch: {per_device_batch_size}")
        print(f"  global batch: {global_batch_size}")
        print(f"  Sinkhorn iterations: {n_iters}")
        print(f"  band width: {band_width}")
        if method == "sinkhorn_ot":
            print(f"  sinkhorn gradient mode: {sinkhorn_gradient_mode}")
            print(f"  sinkhorn refine steps: {sinkhorn_refine_steps}")
            print(f"  sinkhorn refine tolerance: {sinkhorn_refine_tolerance}")
        print(f"  projection manifold: {projection_manifold}")
        print(f"  gap mode: {gap_mode}")
        print(f"  metric every: {metric_every}")
        print(f"  resume step: {start_step}")

    start_time = time.perf_counter()
    loss = None
    metric = None
    step = start_step
    while True:
        batch = ds.sample_batch(batch_size=local_batch_size)
        emb_q, emb_k, target_cols, target_mask, q_mask, k_mask = batch

        emb_q = reshape_for_pmap(emb_q, local_device_count)
        emb_k = reshape_for_pmap(emb_k, local_device_count)
        target_cols = reshape_for_pmap(target_cols, local_device_count)
        target_mask = reshape_for_pmap(target_mask, local_device_count)
        q_mask = reshape_for_pmap(q_mask, local_device_count)
        k_mask = reshape_for_pmap(k_mask, local_device_count)

        params, opt_state, loss = train_step(
            params, opt_state, emb_q, emb_k, target_cols, target_mask, q_mask, k_mask
        )

        if metric_step is not None and (step % metric_every == 0):
            metric = metric_step(params, emb_q, emb_k, target_cols, target_mask, q_mask, k_mask)

        if step % 50 == 0:
            jax.block_until_ready(loss)
            if is_primary:
                elapsed = time.perf_counter() - start_time
                examples_per_second = ((step - start_step + 1) * global_batch_size) / max(elapsed, 1e-6)
                loss_value = float(jax.device_get(loss[0]))
                ce_display = "     off"
                mono_display = "     off"
                if metric is not None:
                    ce_value = float(jax.device_get(metric["ce"][0]))
                    mono_value = float(jax.device_get(metric["mono"][0]))
                    ce_display = f"{ce_value:>8.4f}"
                    mono_display = f"{mono_value:>8.4f}"
                print(
                    f"Step {step:>6} | Loss: {loss_value:>8.4f} | CE: {ce_display} | Mono: {mono_display} | "
                    f"Time: {int(elapsed):>6}s | Ex/s: {examples_per_second:>7.2f}"
                )

        if step > 0 and step % checkpoint_every == 0:
            jax.block_until_ready(loss)
            if is_primary:
                save_checkpoint(
                    params,
                    opt_state,
                    step,
                    ckpt_dir,
                    gcs_ckpt_dir,
                    replicated=True,
                    verbose=True,
                )

        should_stop = False
        if is_primary:
            should_stop = (time.perf_counter() - start_time) >= (max_hours * 3600)
        should_stop = _broadcast_stop_flag(should_stop)
        if should_stop:
            break

        step += 1

    if loss is not None:
        jax.block_until_ready(loss)
    if is_primary:
        save_checkpoint(
            params,
            opt_state,
            step,
            ckpt_dir,
            gcs_ckpt_dir,
            replicated=True,
            verbose=True,
        )
        print("Pfam training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["flash_attention", "sinkhorn_ot"], required=True)
    parser.add_argument("--max_hours", type=float, default=3.0)
    parser.add_argument("--data_dir", default="./data/pfam_shards")
    parser.add_argument("--ckpt_dir", default="./checkpoints")
    parser.add_argument("--gcs_ckpt_dir", help="gs://bucket/path for redundant checkpointing")
    parser.add_argument("--band_width", type=int, default=1024)
    parser.add_argument("--n_iters", type=int, default=15)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--batch_size_file", help="JSON file used to share tuned per-device batch size.")
    parser.add_argument("--autotune_batch_size", action="store_true", help="Autotune per-device batch size (costs extra compile time).")
    parser.add_argument("--autotune_max_per_device_batch_size", type=int, default=8)
    parser.add_argument("--checkpoint_every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric_every", type=int, default=0, help="Compute CE metric every N steps (0 disables).")
    parser.add_argument(
        "--sinkhorn_gradient_mode",
        choices=["tail_refinement", "exact_autodiff"],
        default="tail_refinement",
        help="Gradient rule for the Sinkhorn path.",
    )
    parser.add_argument(
        "--sinkhorn_refine_steps",
        type=int,
        default=2,
        help="Fixed refinement depth, or maximum depth when a refinement tolerance is provided.",
    )
    parser.add_argument(
        "--sinkhorn_refine_tolerance",
        type=float,
        help="Optional infinity-norm residual tolerance for adaptive tail refinement.",
    )
    parser.add_argument(
        "--projection_manifold",
        choices=["none", "stiefel_qk"],
        default="none",
        help="Optional manifold retraction applied to the projection heads after each optimizer step.",
    )
    parser.add_argument(
        "--gap_mode",
        choices=["none", "dustbin_block"],
        default="none",
        help="Optional gap-aware extension. dustbin_block adds a learned row/column dustbin in one extra 128-token block.",
    )
    args = parser.parse_args()

    train_pfam(
        method=args.method,
        max_hours=args.max_hours,
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        gcs_ckpt_dir=args.gcs_ckpt_dir,
        band_width=args.band_width,
        n_iters=args.n_iters,
        per_device_batch_size=args.per_device_batch_size,
        batch_size_file=args.batch_size_file,
        autotune_batch_size=args.autotune_batch_size,
        autotune_max_per_device_batch_size=args.autotune_max_per_device_batch_size,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        metric_every=args.metric_every,
        sinkhorn_gradient_mode=args.sinkhorn_gradient_mode,
        sinkhorn_refine_steps=args.sinkhorn_refine_steps,
        sinkhorn_refine_tolerance=args.sinkhorn_refine_tolerance,
        projection_manifold=args.projection_manifold,
        gap_mode=args.gap_mode,
    )
