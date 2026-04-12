"""
Pfam Protein Alignment Benchmark (Sharded Production Edition)
============================================================

Optimized for A-grade TPU v6e-8 hardware using sharded float16 embeddings.
"""

import os
import sys
import time
import json
import argparse
import random
import numpy as np

import jax
import jax.numpy as jnp
import optax

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ot_kernel

# ==============================================================================
# JAX Production Shard Loader
# ==============================================================================

class PfamShardDataset:
    def __init__(self, data_dir, seq_len=1024):
        self.data_dir = data_dir
        self.seq_len = seq_len
        with open(os.path.join(data_dir, "metadata.json")) as f:
            self.metadata = json.load(f)
        
        self.total_shards = self.metadata['total_shards']
        self.shard_size = self.metadata['shard_size']
        self.curr_shard_idx = -1
        self.curr_shard_data = None

    def _load_shard(self, shard_idx):
        if self.curr_shard_idx == shard_idx:
            return
        path = os.path.join(self.data_dir, f"shard_{shard_idx:03d}.npz")
        self.curr_shard_data = np.load(path)
        self.curr_shard_idx = shard_idx

    def sample_batch(self, batch_size=1):
        # Pick a random shard
        shard_idx = random.randint(0, self.total_shards - 1)
        self._load_shard(shard_idx)
        
        # Pick random indices within the shard
        num_in_shard = self.curr_shard_data['emb1'].shape[0]
        indices = np.random.choice(num_in_shard, batch_size, replace=False)
        
        # Load and cast to float32 for JAX
        e1 = self.curr_shard_data['emb1'][indices].astype(np.float32)
        e2 = self.curr_shard_data['emb2'][indices].astype(np.float32)
        target = self.curr_shard_data['target'][indices].astype(np.float32)
        
        # Expand dims for (B, H, L, D) -> (batch_size, 1, 1024, 320)
        return e1[:, None, :, :], e2[:, None, :, :], target[:, None, :, :]

# ==============================================================================
# MUON Optimizer
# ==============================================================================
def muzzle_orthogonalize(G, steps=5):
    if G.ndim < 2: return G
    original_shape = G.shape
    G_flat = G.reshape(G.shape[0], -1)
    X = G_flat / (jnp.linalg.norm(G_flat) + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.reshape(original_shape)

def muon_optimizer(lr):
    def init_fn(params): return optax.EmptyState()
    def update_fn(grads, state, params=None):
        updates = jax.tree_util.tree_map(muzzle_orthogonalize, grads)
        return jax.tree_util.tree_map(lambda x: x * lr, updates), state
    return optax.GradientTransformation(init_fn, update_fn)

# ==============================================================================
# Training Loop
# ==============================================================================

def train_pfam(method, max_hours, data_dir):
    ds = PfamShardDataset(data_dir)
    num_pairs = ds.metadata['total_pairs']
    
    D_in, D_proj = 320, 64
    band_width = 1024
    
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    params = {
        "W_q": jax.random.normal(k1, (D_in, D_proj)) * (1.0/jnp.sqrt(D_in)),
        "W_k": jax.random.normal(k2, (D_in, D_proj)) * (1.0/jnp.sqrt(D_in)),
        "W_v": jax.random.normal(k3, (D_in, D_proj)) * (1.0/jnp.sqrt(D_in)),
    }
    
    optimizer = muon_optimizer(3e-4)
    opt_state = optimizer.init(params)

    def stable_sinkhorn_probs(Q, K, scale):
        """Compute potential-stabilized transport probabilities for Pfam training."""
        u, v = ot_kernel.compute_global_uv(Q, K, n_iters=15, band_width=band_width)
        u = jnp.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        v = jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        logits = jnp.einsum("bhmd,bhnd->bhmn", Q, K) * scale
        logits = logits + u[..., :, None] + v[..., None, :]
        logits = jnp.clip(logits, -40.0, 40.0)
        probs = jax.nn.softmax(logits, axis=-1)
        probs = jnp.nan_to_num(probs, nan=0.0)
        denom = jnp.sum(probs, axis=-1, keepdims=True) + 1e-10
        return probs / denom

    @jax.jit
    def train_step(p, opt, q, k, t):
        def loss_fn(params):
            Q = jnp.einsum("bhld,dp->bhlp", q, params["W_q"])
            K = jnp.einsum("bhld,dp->bhlp", k, params["W_k"])
            scale = 1.0 / jnp.sqrt(Q.shape[-1])
            if method == "sinkhorn_ot":
                O_pallas = ot_kernel.pallas_flash_sinkhorn(Q, K, K, n_iters=15, band_width=band_width)
                P = stable_sinkhorn_probs(Q, K, scale)
                aux_loss = 1e-4 * jnp.mean(jnp.square(jnp.nan_to_num(O_pallas)))
            else:
                P = jax.nn.softmax(jnp.einsum("bhmd,bhnd->bhmn", Q, K) * scale, axis=-1)
                aux_loss = 0.0
            ce_loss = -jnp.mean(jnp.sum(t * jnp.log(jnp.clip(P, 1e-10, 1.0)), axis=-1))
            return ce_loss + aux_loss

        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, next_opt = optimizer.update(grads, opt, p)
        return optax.apply_updates(p, updates), next_opt, loss

    print(f"\n🚀 Sharded Production Training ({method}) on {num_pairs} Pfam pairs...")
    print(f"   (Max Hours: {max_hours} | Shards: {ds.total_shards})")
    
    start = time.time()
    step = 0
    while (time.time() - start) < (max_hours * 3600):
        q, k, t = ds.sample_batch(batch_size=4)
        params, opt_state, loss = train_step(params, opt_state, q, k, t)
        
        if step % 50 == 0:
            jax.block_until_ready(loss)
            print(f"Step {step:>6} | Loss: {float(loss):>8.4f} | Time: {int(time.time()-start)}s")
        step += 1

    print("\n✓ Sharded Pfam training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["flash_attention", "sinkhorn_ot"], required=True)
    parser.add_argument("--max_hours", type=float, default=3.0)
    parser.add_argument("--data_dir", default="./data/pfam_shards")
    args = parser.parse_args()
    train_pfam(args.method, args.max_hours, args.data_dir)
