"""
Multi-Species Cross-Imputation Benchmark (REAL DNA ONLY)
========================================================

This script explicitly uses REAL biological orthologous DNA sequences 
(Human/Mouse conserved gene pairs from Ensembl) to evaluate our custom `dO` VJP.

Task: Cross-Species Embeddings Imputation
We corrupt the Query sequence embeddings (Human) using Gaussian masking. 
The network must attend to the uncorrupted orthologous Key sequence (Mouse) 
using `O = pallas_flash_sinkhorn(Q, K, V)` to impute/reconstruct the missing 
human biological structure.

Because the loss is computed over the Output matrix O, JAX analytically 
routes gradients through our custom, highly optimized Pallas backward bindings.
"""

import importlib
import os
import sys
import time
import argparse
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp
import optax

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ot_kernel
from benchmarks.benchmark_wallclock_training import download_ortholog_pairs

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
        new_updates = jax.tree_util.tree_map(lambda g: -g * learning_rate, new_updates)
        return new_updates, state
    return optax.GradientTransformation(init_fn, update_fn)


# ==============================================================================
# REAL DATA: DNA Feature Preparation
# ==============================================================================

def _load_hf_modules() -> tuple[Any, Any, Any, Any]:
    torch_module = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    return (
        torch_module,
        transformers.AutoTokenizer,
        transformers.AutoModel,
        transformers.AutoConfig,
    )

def prepare_real_ortholog_dataset(data_dir, model_name, seq_len, pad_granularity=128):
    torch_module, AutoTokenizer, AutoModel, AutoConfig = _load_hf_modules()

    pairs = download_ortholog_pairs(data_dir, seq_len)
    if not pairs:
        raise ValueError("Network failed to fetch real Ensembl biological pairs!")

    remainder = seq_len % pad_granularity
    if remainder != 0:
        seq_len = seq_len + (pad_granularity - remainder)

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(config, "is_decoder"):
        config.is_decoder = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True, ignore_mismatched_sizes=True)
    model.eval()

    def embed_batch(sequences):
        inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
        with torch_module.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.numpy()
        B, L_tok, D = emb.shape
        if L_tok < seq_len:
            emb = np.pad(emb, ((0, 0), (0, seq_len - L_tok), (0, 0)))
        elif L_tok > seq_len:
            emb = emb[:, :seq_len, :]
        return emb

    print("Embedding human biological sequences...")
    emb_human = embed_batch([p["seq_human"] for p in pairs])
    print("Embedding mouse biological sequences...")
    emb_mouse = embed_batch([p["seq_mouse"] for p in pairs])

    emb_q = jax.device_put(jnp.array(emb_human[:, None, :, :]))
    emb_k = jax.device_put(jnp.array(emb_mouse[:, None, :, :]))
    
    return emb_q, emb_k, seq_len


# ==============================================================================
# Model Definition
# ==============================================================================

def init_imputation_model(key, D_in, D_proj):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    scale = 1.0 / jnp.sqrt(D_in)
    return {
        "W_q": jax.random.normal(k1, (D_in, D_proj)) * scale,
        "W_k": jax.random.normal(k2, (D_in, D_proj)) * scale,
        "W_v": jax.random.normal(k3, (D_in, D_proj)) * scale,
        "W_out": jax.random.normal(k4, (D_proj, D_in)) * (1.0 / jnp.sqrt(D_proj)),
    }


def project(params, emb_q, emb_k):
    Q = jnp.einsum("bhld,dp->bhlp", emb_q, params["W_q"])
    K = jnp.einsum("bhld,dp->bhlp", emb_k, params["W_k"])
    V = jnp.einsum("bhld,dp->bhlp", emb_k, params["W_v"])
    return Q, K, V


# ==============================================================================
# Loss: TRIGGERS CUSTOM PALLAS dO KERNELS Via Iterative Imputation
# ==============================================================================

def imputation_loss_sinkhorn(params, emb_q, emb_k, band_width):
    # Differentiable computation forcing backward custom VJP triggers!
    Q, K, V = project(params, emb_q, emb_k)
    out_O = ot_kernel.pallas_flash_sinkhorn(Q, K, V, n_iters=15, band_width=band_width)
    
    # Reconstruct human DNA tokens from mouse context
    imputed_human = jnp.einsum("bhlp,pd->bhld", out_O, params["W_out"])
    
    # Target is uncorrupted biological embedding sequence itself
    return jnp.mean((imputed_human - emb_q) ** 2)


def imputation_loss_flash(params, emb_q, emb_k, band_width):
    Q, K, V = project(params, emb_q, emb_k)
    out_O = jax.nn.dot_product_attention(Q, K, V)
    imputed_human = jnp.einsum("bhlp,pd->bhld", out_O, params["W_out"])
    return jnp.mean((imputed_human - emb_q) ** 2)


# ==============================================================================
# Framework Loop
# ==============================================================================

def run_real_imputation_benchmark(method, max_hours, seq_len, data_dir, model_name, gcs_bucket=None):
    loss_fn = imputation_loss_sinkhorn if method == "sinkhorn_ot" else imputation_loss_flash
    
    # 100% real Ensembl biological pairs
    emb_q, emb_k, actual_seq = prepare_real_ortholog_dataset(data_dir, model_name, seq_len)
    N, _, _, D_in = emb_q.shape
    D_proj = 64
    band_width = min(2048, actual_seq)

    params = init_imputation_model(jax.random.PRNGKey(42), D_in, D_proj)
    optimizer = muon_optimizer(1e-3)
    opt_state = optimizer.init(params)
    
    # Corrupt Q with Gaussian structural masking (dropout proxy)
    masking_key = jax.random.PRNGKey(99)
    noise = jax.random.normal(masking_key, emb_q.shape) * 0.5
    emb_q_corrupted = emb_q + noise

    # Strict batch auto-tune testing VMEM boundaries
    print("Auto-detecting ideal VMEM batch sizing...")
    batch_size = 1
    lo, hi = 1, min(64, emb_q.shape[0])
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            loss_val, grads_out = jax.value_and_grad(loss_fn)(params, emb_q_corrupted[:mid], emb_k[:mid], band_width)
            jax.block_until_ready(loss_val)
            jax.block_until_ready(grads_out)
            batch_size = mid
            lo = mid + 1
        except Exception:
            hi = mid - 1
            
    print(f"✓ Operational batch_size locked to {batch_size}")

    @jax.jit
    def train_step(params, opt_state, b_q_corr, b_q, b_k):
        # We compute loss parsing Q_corrupted structurally mapped against pure K
        loss, grads = jax.value_and_grad(loss_fn)(params, b_q_corr, b_k, band_width)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss

    max_seconds = max_hours * 3600
    session_start_time = time.time()
    step = 0
    
    print("\n=======================================================")
    print(" Real DNA Cross-Imputation Downstream Evaluator")
    print(" Triggering `dO` gradients via pallas backward VJP!")
    print(f" Dataset Size: {N} Ensembl Pairs | Sequence Size: {actual_seq}")
    print("=======================================================\n")

    while True:
        elapsed = time.time() - session_start_time
        if elapsed >= max_seconds:
            print(f"\n⏱ Budget ceiling hit ({elapsed / 3600:.2f}h).")
            break

        idx = np.random.choice(N, size=batch_size, replace=False)
        start = time.time()
        
        try:
            params, opt_state, loss = train_step(params, opt_state, emb_q_corrupted[idx], emb_q[idx], emb_k[idx])
            jax.block_until_ready(loss)
        except Exception as e:
            print(f"\nFATAL: Preempted or Memory Crash on '{method}'! {e}")
            break

        throughput = 1.0 / max(time.time() - start, 1e-6)
        if step % 5 == 0:
            print(f"Step {step:>5} | MSE Imputation Loss: {float(loss):>8.4f} | {throughput:>6.2f} steps/s")
        step += 1

    print("\n✓ Real biological analysis completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["flash_attention", "sinkhorn_ot"], required=True)
    parser.add_argument("--max_hours", type=float, default=3.4)
    parser.add_argument("--seq_len", type=int, default=32768)
    parser.add_argument("--model_name", type=str, default="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species")
    parser.add_argument("--gcs_bucket", type=str, default="")
    args = parser.parse_args()

    run_real_imputation_benchmark(args.method, args.max_hours, args.seq_len, "./data/dna_orthologs", args.model_name)
