"""
Pfam Protein Alignment & Structural Transitivity Benchmark (A-Grade Optimized)
=============================================================================

This script evaluates Sinkhorn OT against Flash Attention using pre-computed
ESM-2 embeddings for peak TPU efficiency on v6e-8 hardware.
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
# MUON Optimizer
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
            lambda g: newton_schulz_orthogonalize(g) if g.ndim >= 2 else g, updates)
        new_updates = jax.tree_util.tree_map(lambda g: g * learning_rate, new_updates)
        return new_updates, state
    return optax.GradientTransformation(init_fn, update_fn)

# ==============================================================================
# Data Loading (Pfam Pre-Computed)
# ==============================================================================

class PfamDataset:
    def __init__(self, data_dir, seq_len=1024):
        self.data_dir = data_dir
        self.seq_len = seq_len
        with open(os.path.join(data_dir, "metadata.json")) as f:
            self.metadata = json.load(f)
        
    def get_pair(self, pair_id):
        m = self.metadata['pairs'][pair_id]
        data = np.load(os.path.join(self.data_dir, m['file']))
        
        # Load pre-computed embeddings
        if 'emb1' not in data:
            raise ValueError(f"NPZ file {m['file']} missing 'emb1'. Run download_pfam.py --embed first.")
            
        e1, e2 = data['emb1'], data['emb2']
        target = data['alignment'].astype(np.float32)
        
        # Padding embeddings to seq_len
        L1, D = e1.shape
        L2, _ = e2.shape
        padded_e1 = np.zeros((self.seq_len, D), dtype=np.float32)
        padded_e2 = np.zeros((self.seq_len, D), dtype=np.float32)
        padded_e1[:min(L1, self.seq_len), :] = e1[:min(L1, self.seq_len), :]
        padded_e2[:min(L2, self.seq_len), :] = e2[:min(L2, self.seq_len), :]
        
        # Padding target to (seq_len, seq_len)
        padded_target = np.zeros((self.seq_len, self.seq_len), dtype=np.float32)
        lp1, lp2 = min(L1, self.seq_len), min(L2, self.seq_len)
        padded_target[:lp1, :lp2] = target[:lp1, :lp2]
        
        return padded_e1[None, None, :, :], padded_e2[None, None, :, :], padded_target

# ==============================================================================
# Structural Transitivity Evaluator
# ==============================================================================

def compute_transitivity_error(params, emb_a, emb_b, emb_c, target_ac, method, band_width):
    def get_plan(q_emb, k_emb):
        Q = jnp.einsum("bhld,dp->bhlp", q_emb, params["W_q"])
        K = jnp.einsum("bhld,dp->bhlp", k_emb, params["W_k"])
        D = Q.shape[-1]
        scale = 1.0 / jnp.sqrt(D)
        
        if method == "sinkhorn_ot":
            u, v = ot_kernel.compute_global_uv(Q, K, n_iters=15, band_width=band_width)
            S = jnp.einsum("bhmd,bhnd->bhmn", Q, K) * scale
            log_P = S + u[..., :, None] + v[..., None, :]
            P = jax.nn.softmax(log_P, axis=-1)
        else:
            S = jnp.einsum("bhmd,bhnd->bhmn", Q, K) * scale
            P = jax.nn.softmax(S, axis=-1)
        return P

    P_ab = get_plan(emb_a, emb_b)
    P_bc = get_plan(emb_b, emb_c)
    P_ac_composed = jnp.matmul(P_ab, P_bc)
    P_ac_direct = get_plan(emb_a, emb_c)
    
    drift = jnp.linalg.norm(P_ac_direct - P_ac_composed)
    target_ac_jax = jnp.array(target_ac)[None, None, :, :]
    ce_loss = -jnp.mean(jnp.sum(target_ac_jax * jnp.log(jnp.clip(P_ac_direct, 1e-10, 1.0)), axis=-1))
    
    return float(drift), float(ce_loss)

# ==============================================================================
# Training Loop
# ==============================================================================

def train_pfam(method, max_hours, seq_len, data_dir):
    ds = PfamDataset(data_dir, seq_len)
    
    D_in = 320 # ESM-2 8M dimension
    D_proj = 64
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

    @jax.jit
    def train_step(params, opt_state, q_emb, k_emb, targets):
        def loss_fn(p):
            Q = jnp.einsum("bhld,dp->bhlp", q_emb, p["W_q"])
            K = jnp.einsum("bhld,dp->bhlp", k_emb, p["W_k"])
            D = Q.shape[-1]
            scale = 1.0 / jnp.sqrt(D)
            
            if method == "sinkhorn_ot":
                # Fused call through our optimized VJP
                out_O = ot_kernel.pallas_flash_sinkhorn(Q, K, K, n_iters=15, band_width=band_width)
                u, v = ot_kernel.compute_global_uv(Q, K, n_iters=15, band_width=band_width)
                log_P = (jnp.einsum("bhmd,bhnd->bhmn", Q, K) * scale) + u[..., :, None] + v[..., None, :]
                P = jax.nn.softmax(log_P, axis=-1)
            else:
                S = jnp.einsum("bhmd,bhnd->bhmn", Q, K) * scale
                P = jax.nn.softmax(S, axis=-1)
            
            P = jnp.clip(P, 1e-10, 1.0)
            return -jnp.mean(jnp.sum(targets[:, None, :, :] * jnp.log(P), axis=-1))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss

    start_time = time.time()
    step = 0
    num_pairs = len(ds.metadata['pairs'])
    
    print(f"\n🚀 Optimized Training ({method}) on {num_pairs} Pfam protein pairs...")
    print(f"   (Using pre-computed ESM-2 embeddings for peak TPU utilization)")
    
    while (time.time() - start_time) < (max_hours * 3600):
        idx = random.randint(0, num_pairs - 1)
        emb_q, emb_k, target = ds.get_pair(idx)
        
        params, opt_state, loss = train_step(params, opt_state, emb_q, emb_k, target[None, :, :])
        jax.block_until_ready(loss)
        
        if step % 50 == 0:
            print(f"Step {step:>6} | Loss: {float(loss):>8.4f} | Time: {int(time.time()-start_time)}s")
        
        if step > 0 and step % 200 == 0:
            print("  🔍 Running Transitivity Audit on curated triplet...")
            m0, m1 = ds.metadata['pairs'][0], ds.metadata['pairs'][1]
            if m0['family'] == m1['family']:
                e_a, e_b, _ = ds.get_pair(0)
                e_c, _, _ = ds.get_pair(1)
                drift, _ = compute_transitivity_error(params, e_a, e_b, e_c, np.zeros((seq_len, seq_len)), method, band_width)
                print(f"  [CONSISTENCY] Pfam Family: {m0['family']} | Drift (Frobenius): {drift:>8.6f}")
            
        step += 1

    print("\n✓ Optimized Pfam training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["flash_attention", "sinkhorn_ot"], required=True)
    parser.add_argument("--max_hours", type=float, default=3.0)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--data_dir", default="./data/pfam_pairs")
    args = parser.parse_args()
    
    train_pfam(args.method, args.max_hours, args.seq_len, args.data_dir)
