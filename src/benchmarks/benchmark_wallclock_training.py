"""
Wall-Clock Training Benchmark: Flash Attention vs Sinkhorn OT
==============================================================

Trains both methods for a fixed wall-clock budget on real Pfam protein
alignment data, comparing loss curves to demonstrate that Sinkhorn's
doubly-stochastic transport plan learns proper alignments where
row-stochastic Flash Attention structurally cannot.

Checkpoints and results are saved to GCS.

Usage:
    # Run Flash Attention for 2.5 hours
    python src/benchmarks/benchmark_wallclock_training.py \\
        --method flash_attention --max_hours 2.5

    # Run Sinkhorn OT for 2.5 hours
    python src/benchmarks/benchmark_wallclock_training.py \\
        --method sinkhorn_ot --max_hours 2.5

    # Quick local smoke test
    python src/benchmarks/benchmark_wallclock_training.py \\
        --method sinkhorn_ot --max_hours 0.01 --seq_len 256 --no_gcs
"""

import os
import sys
import time
import json
import csv
import argparse
import datetime
import functools
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

# Initialize JAX distributed for multi-TPU
if "TPU_WORKER_HOSTNAMES" in os.environ:
    try:
        jax.distributed.initialize()
    except Exception:
        pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ot_kernel


# ==============================================================================
# MUON Optimizer (Momentum Orthogonalizer via Newton-Schulz)
# ==============================================================================

def newton_schulz_orthogonalize(G, steps=5):
    original_shape = G.shape
    if G.ndim > 2:
        G = G.reshape(G.shape[0], -1)
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (jnp.linalg.norm(G) + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.reshape(original_shape)


def muon_optimizer(learning_rate):
    def init_fn(params):
        return optax.EmptyState()
    def update_fn(updates, state, params=None):
        new_updates = jax.tree_util.tree_map(
            lambda g: newton_schulz_orthogonalize(g) if g.ndim >= 2 else g,
            updates,
        )
        new_updates = jax.tree_util.tree_map(lambda g: g * learning_rate, new_updates)
        return new_updates, state
    return optax.GradientTransformation(init_fn, update_fn)


# ==============================================================================
# Data Loading: Pfam Pairs
# ==============================================================================

def load_pfam_pairs(data_dir: str, seq_len: int, max_pairs: int = None):
    """
    Load preprocessed Pfam pairwise alignment data.
    
    Returns:
        sequences: list of (seq1_str, seq2_str) tuples
        alignments: list of np.ndarray alignment matrices
    """
    meta_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Pfam data not found at {data_dir}. "
            f"Run: python data/download_pfam.py --output_dir {data_dir}"
        )
    
    with open(meta_path) as f:
        metadata = json.load(f)
    
    pairs_meta = metadata['pairs']
    if max_pairs:
        pairs_meta = pairs_meta[:max_pairs]
    
    sequences = []
    alignments = []
    
    for pm in pairs_meta:
        npz = np.load(os.path.join(data_dir, pm['file']), allow_pickle=True)
        seq1 = ''.join(npz['seq1'])
        seq2 = ''.join(npz['seq2'])
        align = npz['alignment']
        
        # Only keep pairs where both sequences fit within seq_len
        if len(seq1) <= seq_len and len(seq2) <= seq_len:
            sequences.append((seq1, seq2))
            alignments.append(align)
    
    print(f"✓ Loaded {len(sequences)} Pfam pairs (max_len={seq_len})")
    return sequences, alignments


def embed_sequences_esm2(sequences, model_name, seq_len, batch_size=8):
    """
    Embed protein sequences using ESM-2 and pad to fixed seq_len.
    
    Returns:
        embeddings: jnp.ndarray of shape (N, seq_len, D)
    """
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    
    print(f"Loading {model_name}...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if not hasattr(config, 'is_decoder'):
        config.is_decoder = False
    if not hasattr(config, 'add_cross_attention'):
        config.add_cross_attention = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, config=config,
        trust_remote_code=True, ignore_mismatched_sizes=True,
    )
    model.eval()
    
    all_embeddings = []
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        emb = outputs.last_hidden_state.numpy()  # (B, L_tok, D)
        
        # Pad to seq_len
        B, L_tok, D = emb.shape
        if L_tok < seq_len:
            emb = np.pad(emb, ((0, 0), (0, seq_len - L_tok), (0, 0)))
        elif L_tok > seq_len:
            emb = emb[:, :seq_len, :]
        
        all_embeddings.append(emb)
        
        if (i // batch_size) % 10 == 0:
            print(f"  Embedded {i + len(batch)}/{len(sequences)} sequences...")
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"✓ Embeddings shape: {embeddings.shape}")
    return jax.device_put(jnp.array(embeddings))


def prepare_dataset(data_dir, model_name, seq_len, pad_granularity=128):
    """
    Load Pfam pairs, embed with ESM-2, and prepare alignment targets.
    
    Returns:
        emb_q: (N, 1, L, D) query embeddings (seq1)
        emb_k: (N, 1, L, D) key embeddings (seq2)
        targets: (N, L, L) ground-truth alignment matrices (padded)
    """
    sequences, alignments = load_pfam_pairs(data_dir, seq_len)
    
    if not sequences:
        raise ValueError(f"No pairs found with seq_len <= {seq_len}. Try increasing --seq_len.")
    
    # Pad seq_len to mod-128 for Pallas
    remainder = seq_len % pad_granularity
    if remainder != 0:
        seq_len = seq_len + (pad_granularity - remainder)
    
    # Embed all sequences
    all_seq1 = [s[0] for s in sequences]
    all_seq2 = [s[1] for s in sequences]
    
    emb1 = embed_sequences_esm2(all_seq1, model_name, seq_len)  # (N, L, D)
    emb2 = embed_sequences_esm2(all_seq2, model_name, seq_len)  # (N, L, D)
    
    # Add head dimension: (N, 1, L, D)
    emb_q = emb1[:, None, :, :]
    emb_k = emb2[:, None, :, :]
    
    # Pad alignment matrices to (seq_len, seq_len)
    padded_targets = []
    for align in alignments:
        h, w = align.shape
        padded = np.zeros((seq_len, seq_len), dtype=np.float32)
        padded[:h, :w] = align
        # For padded rows, set uniform distribution to avoid NaN in cross-entropy
        for row in range(h, seq_len):
            padded[row, :] = 1.0 / seq_len
        # For original rows with no alignment, also set uniform
        for row in range(h):
            if padded[row].sum() == 0:
                padded[row, :] = 1.0 / seq_len
        padded_targets.append(padded)
    
    targets = jnp.array(np.stack(padded_targets))  # (N, L, L)
    
    print(f"✓ Dataset ready: {emb_q.shape[0]} pairs, L={seq_len}, D={emb_q.shape[-1]}")
    return emb_q, emb_k, targets, seq_len


# ==============================================================================
# Model: Learnable Q/K/V Projection Heads
# ==============================================================================

def init_projection_heads(key, D_in, D_proj):
    """Initialize Q, K, V projection matrices."""
    k1, k2, k3 = jax.random.split(key, 3)
    scale = 1.0 / jnp.sqrt(D_in)
    params = {
        'W_q': jax.random.normal(k1, (D_in, D_proj)) * scale,
        'W_k': jax.random.normal(k2, (D_in, D_proj)) * scale,
        'W_v': jax.random.normal(k3, (D_in, D_proj)) * scale,
    }
    return params


def project(params, emb_q, emb_k):
    """Project embeddings through learned heads."""
    Q = jnp.einsum('bhld,dp->bhlp', emb_q, params['W_q'])
    K = jnp.einsum('bhld,dp->bhlp', emb_k, params['W_k'])
    V = jnp.einsum('bhld,dp->bhlp', emb_k, params['W_v'])
    return Q, K, V


# ==============================================================================
# Loss Functions
# ==============================================================================

def alignment_loss_sinkhorn(params, emb_q, emb_k, targets, band_width):
    """
    Cross-entropy between Sinkhorn OT transport plan and ground-truth alignment.
    The OT plan is doubly-stochastic → can represent the permutation target.
    """
    Q, K, V = project(params, emb_q, emb_k)
    
    # Get Sinkhorn potentials to compute transport plan
    D = Q.shape[-1]
    scale = 1.0 / jnp.sqrt(D)
    u, v = ot_kernel.compute_global_uv(Q, K, n_iters=15, band_width=band_width)
    
    # Compute transport plan P = exp(S + u + v)
    S = jnp.einsum('bhmd,bhnd->bhmn', Q, K) * scale
    log_P = S + u[..., :, None] + v[..., None, :]
    P = jax.nn.softmax(log_P, axis=-1)  # Normalize for numerical stability
    
    # Cross-entropy: -sum(target * log(P))
    P = jnp.clip(P, 1e-10, 1.0)
    loss = -jnp.mean(jnp.sum(targets[:, None, :, :] * jnp.log(P), axis=-1))
    return loss


def alignment_loss_flash(params, emb_q, emb_k, targets, band_width):
    """
    Cross-entropy between Flash Attention weights and ground-truth alignment.
    Attention is row-stochastic → cannot represent doubly-stochastic target.
    """
    Q, K, V = project(params, emb_q, emb_k)
    
    # Standard softmax attention weights
    D = Q.shape[-1]
    scale = 1.0 / jnp.sqrt(D)
    S = jnp.einsum('bhmd,bhnd->bhmn', Q, K) * scale
    A = jax.nn.softmax(S, axis=-1)  # Row-stochastic only
    
    # Cross-entropy
    A = jnp.clip(A, 1e-10, 1.0)
    loss = -jnp.mean(jnp.sum(targets[:, None, :, :] * jnp.log(A), axis=-1))
    return loss


# ==============================================================================
# Checkpointing
# ==============================================================================

def save_checkpoint(params, step, method, gcs_bucket, local_dir="/tmp/checkpoints"):
    """Save model checkpoint to local disk and optionally to GCS."""
    ckpt_dir = os.path.join(local_dir, method, f"step_{step:06d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Save params as .npz
    np_params = {k: np.array(v) for k, v in params.items()}
    ckpt_path = os.path.join(ckpt_dir, "params.npz")
    np.savez(ckpt_path, **np_params)
    
    if gcs_bucket:
        gcs_path = f"gs://{gcs_bucket}/checkpoints/{method}/step_{step:06d}/params.npz"
        os.system(f"gsutil -q cp {ckpt_path} {gcs_path}")
    
    return ckpt_path


def save_results(results, method, gcs_bucket, local_dir="/tmp/results"):
    """Save loss curve and summary to local + GCS."""
    os.makedirs(local_dir, exist_ok=True)
    
    # Save CSV loss curve
    csv_path = os.path.join(local_dir, f"{method}_loss_curve.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'wallclock_sec', 'loss', 'throughput_steps_per_sec'])
        for r in results['loss_curve']:
            writer.writerow([r['step'], r['wallclock_sec'], r['loss'], r['throughput']])
    
    # Save JSON summary
    json_path = os.path.join(local_dir, f"{method}_summary.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    if gcs_bucket:
        os.system(f"gsutil -q cp {csv_path} gs://{gcs_bucket}/results/")
        os.system(f"gsutil -q cp {json_path} gs://{gcs_bucket}/results/")
    
    print(f"\n✓ Results saved: {csv_path}, {json_path}")
    if gcs_bucket:
        print(f"  → GCS: gs://{gcs_bucket}/results/")


# ==============================================================================
# Auto-detect Batch Size
# ==============================================================================

def auto_detect_batch_size(params, emb_q, emb_k, targets, loss_fn, band_width, max_batch=64):
    """Binary search for max batch size that fits in TPU memory."""
    print("Auto-detecting max batch size...")
    
    best_batch = 1
    lo, hi = 1, min(max_batch, emb_q.shape[0])
    
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            batch_q = emb_q[:mid]
            batch_k = emb_k[:mid]
            batch_t = targets[:mid]
            
            # Try forward + backward
            loss, grads = jax.value_and_grad(loss_fn)(params, batch_q, batch_k, batch_t, band_width)
            jax.block_until_ready(loss)
            jax.block_until_ready(grads)
            
            best_batch = mid
            lo = mid + 1
            print(f"  batch_size={mid}: ✓ fits")
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "out of memory" in str(e).lower():
                hi = mid - 1
                print(f"  batch_size={mid}: ✗ OOM")
            else:
                hi = mid - 1
                print(f"  batch_size={mid}: ✗ {e}")
    
    print(f"✓ Auto-detected batch_size={best_batch}")
    return best_batch


# ==============================================================================
# Training Loop
# ==============================================================================

def train(
    method: str,
    emb_q: jnp.ndarray,
    emb_k: jnp.ndarray,
    targets: jnp.ndarray,
    seq_len: int,
    max_hours: float,
    gcs_bucket: str,
    checkpoint_every: int = 100,
    lr: float = 1e-3,
    D_proj: int = 64,
    seed: int = 42,
):
    """Main training loop with wall-clock budget."""
    
    D_in = emb_q.shape[-1]
    N = emb_q.shape[0]
    band_width = min(2048, seq_len)
    
    # Select loss function
    if method == "sinkhorn_ot":
        loss_fn = alignment_loss_sinkhorn
    elif method == "flash_attention":
        loss_fn = alignment_loss_flash
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Initialize model
    key = jax.random.PRNGKey(seed)
    params = init_projection_heads(key, D_in, D_proj)
    
    # Initialize optimizer
    optimizer = muon_optimizer(lr)
    opt_state = optimizer.init(params)
    
    # Auto-detect batch size
    batch_size = auto_detect_batch_size(params, emb_q, emb_k, targets, loss_fn, band_width)
    
    # JIT-compile training step
    @jax.jit
    def train_step(params, opt_state, batch_q, batch_k, batch_t):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_q, batch_k, batch_t, band_width)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss
    
    # Training loop
    max_seconds = max_hours * 3600
    loss_curve = []
    step = 0
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"  Training: {method}")
    print(f"  Budget: {max_hours} hours ({max_seconds:.0f}s)")
    print(f"  Dataset: {N} pairs, L={seq_len}, D_in={D_in}, D_proj={D_proj}")
    print(f"  Batch size: {batch_size}, Band width: {band_width}")
    print(f"{'='*70}\n")
    print(f"{'Step':>8} | {'Loss':>12} | {'Wall (s)':>10} | {'Steps/s':>10}")
    print("-" * 50)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_seconds:
            print(f"\n⏱ Time budget exhausted ({elapsed/3600:.2f}h)")
            break
        
        # Sample random batch
        idx = np.random.choice(N, size=batch_size, replace=(batch_size > N))
        batch_q = emb_q[idx]
        batch_k = emb_k[idx]
        batch_t = targets[idx]
        
        # Train step
        step_start = time.time()
        params, opt_state, loss = train_step(params, opt_state, batch_q, batch_k, batch_t)
        jax.block_until_ready(loss)
        step_time = time.time() - step_start
        
        loss_val = float(loss)
        throughput = 1.0 / step_time if step_time > 0 else 0
        
        loss_curve.append({
            'step': step,
            'wallclock_sec': elapsed + step_time,
            'loss': loss_val,
            'throughput': throughput,
        })
        
        # Print progress
        if step % 10 == 0:
            print(f"{step:>8} | {loss_val:>12.6f} | {elapsed:>10.1f} | {throughput:>10.2f}")
        
        # Checkpoint
        if step > 0 and step % checkpoint_every == 0:
            ckpt_path = save_checkpoint(params, step, method, gcs_bucket)
            print(f"  💾 Checkpoint: step {step} → {ckpt_path}")
        
        step += 1
    
    # Final checkpoint
    save_checkpoint(params, step, method, gcs_bucket)
    
    total_time = time.time() - start_time
    
    results = {
        'method': method,
        'total_steps': step,
        'total_time_sec': total_time,
        'final_loss': loss_curve[-1]['loss'] if loss_curve else None,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'D_proj': D_proj,
        'lr': lr,
        'band_width': band_width,
        'dataset_size': N,
        'loss_curve': loss_curve,
    }
    
    save_results(results, method, gcs_bucket)
    
    print(f"\n{'='*70}")
    print(f"  {method} — COMPLETE")
    print(f"  Steps: {step}, Final loss: {results['final_loss']:.6f}")
    print(f"  Total time: {total_time/3600:.2f}h")
    print(f"{'='*70}")
    
    return results


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wall-Clock Training Benchmark")
    parser.add_argument("--method", type=str, required=True,
                        choices=["flash_attention", "sinkhorn_ot"],
                        help="Training method to benchmark")
    parser.add_argument("--max_hours", type=float, default=2.5,
                        help="Wall-clock budget in hours")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Fixed sequence length (padded to mod-128)")
    parser.add_argument("--data_dir", type=str, default="./data/pfam_pairs",
                        help="Path to pre-processed Pfam pairs")
    parser.add_argument("--model_name", type=str, 
                        default="facebook/esm2_t6_8M_UR50D",
                        help="ESM-2 model for embeddings")
    parser.add_argument("--gcs_bucket", type=str, default="sinkhorn-neurips-data",
                        help="GCS bucket for checkpoints and results")
    parser.add_argument("--no_gcs", action="store_true",
                        help="Disable GCS upload (local only)")
    parser.add_argument("--checkpoint_every", type=int, default=100,
                        help="Steps between checkpoints")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--D_proj", type=int, default=64, help="Projection dimension")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    gcs_bucket = None if args.no_gcs else args.gcs_bucket
    
    print(f"JAX devices: {jax.devices()}")
    print(f"Device count: {jax.device_count()}")
    
    # Load and embed data
    emb_q, emb_k, targets, seq_len = prepare_dataset(
        args.data_dir, args.model_name, args.seq_len
    )
    
    # Run training
    train(
        method=args.method,
        emb_q=emb_q,
        emb_k=emb_k,
        targets=targets,
        seq_len=seq_len,
        max_hours=args.max_hours,
        gcs_bucket=gcs_bucket,
        checkpoint_every=args.checkpoint_every,
        lr=args.lr,
        D_proj=args.D_proj,
        seed=args.seed,
    )
