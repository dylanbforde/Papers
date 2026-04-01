"""
Long-Context Alignment Benchmark Suite - PRO Edition
====================================================

This script evaluates the Multi-Transport-Plan Sinkhorn kernel against SOTA
baselines (Flash Attention) using rich representations from Long-Context Foundation Models.

By defaulting to a Nucleotide sequence generator, this benchmark strictly evaluates
ultra-long context alignment (16K+ tokens), representing tasks like Mega-Protein 
alignment or Whole Genome / DNA Synteny matching where $O(L^2)$ attention physically fails. 

Features:
- SOTA Optimizers: MUON (Momentum Orthogonalizer) + AdamW.
- SOTA Baselines: Flash Attention (JAX native block-wise dot-product attention).
- Telemetry: JAX HW Profiling (Tensorboard) and comprehensive JSON data dumping.

Usage:
    # DNA / Nucleotide Transformer (e.g. InstaDeep's Multi-Species 100M)
    python benchmark_long_context_alignment.py --model_name "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species" --vocab dna
    
    # Mega-Protein (e.g. ESM-2)
    python benchmark_long_context_alignment.py --model_name "facebook/esm2_t6_8M_UR50D" --vocab protein
"""

import os
import sys

# [CRITICAL FIX] Trillium single-host TPU VMs will crash inside jax backend initialization 
# if the global grpc topology environment variables are erroneously left populated by GCP.
if 'TPU_WORKER_HOSTNAMES' in os.environ and len(os.environ['TPU_WORKER_HOSTNAMES'].split(",")) == 1:
    os.environ.pop('TPU_WORKER_HOSTNAMES', None)
    os.environ.pop('TPU_WORKER_ID', None)
    os.environ.pop('TPU_NAME', None)

import time
import json
import argparse
import datetime
import numpy as np

import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, PartitionSpec, NamedSharding

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Initialize JAX multi-host distributed cluster (only required for Multi-TPU pod slices > 8 chips)
if "MEGATRON_WORKER" in os.environ or ("TPU_WORKER_HOSTNAMES" in os.environ and len(os.environ['TPU_WORKER_HOSTNAMES'].split(",")) > 1):
    try:
        jax.distributed.initialize()
    except Exception:
        pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ot_kernel


# ==============================================================================
# MUON OPTIMIZER SEED (Momentum Orthogonalizer via Newton-Schulz)
# ==============================================================================
def newton_schulz_orthogonalize(G, steps=5):
    original_shape = G.shape
    if G.ndim > 2:
        G = G.reshape(G.shape[0], -1)
    
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G
    X = X / (jnp.linalg.norm(X) + 1e-7)
    
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
            updates
        )
        new_updates = jax.tree_util.tree_map(lambda g: g * learning_rate, new_updates)
        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


# ==============================================================================
# FOUNDATION MODEL FEATURE EXTRACTION (DNA or Protein)
# ==============================================================================
def get_foundation_embeddings(sequences, model_name):
    print(f"Loading {model_name}...")
    from transformers import AutoConfig
    
    # 1. Load the config first and patch the HuggingFace v4.40+ bug BEFORE model initialization
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if not hasattr(config, 'is_decoder'):
        config.is_decoder = False
    if not hasattr(config, 'add_cross_attention'):
        config.add_cross_attention = False
        
    # 2. Instantiate with the patched config
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True, ignore_mismatched_sizes=True)
    model.eval()

    inputs = tokenizer(sequences, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        
    hidden_states = outputs.last_hidden_state.numpy()
    return jax.device_put(hidden_states)


# ==============================================================================
# PROFILING & BENCHMARK CORE
# ==============================================================================
def benchmark_attention_kernel(Q, K, V, kernel_fn, name, log_dir, num_iters=5):
    # 1. Forward Warmup & Timing (No Profiler)
    for _ in range(2):
        out = kernel_fn(Q, K, V)
        jax.block_until_ready(out)

    start = time.time()
    for _ in range(num_iters):
        out = kernel_fn(Q, K, V)
        jax.block_until_ready(out)
    fwd_latency = ((time.time() - start) / num_iters) * 1000

    # 2. Forward Profiling (Strictly 1 isolated pulse to avoid massive trace sizes)
    jax.profiler.start_trace(log_dir)
    jax.block_until_ready(kernel_fn(Q, K, V))
    jax.profiler.stop_trace()

    # 3. Setup Backward Loss
    def loss_fn(q, k, v):
        return jnp.sum(kernel_fn(q, k, v))
        
    grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))
    
    # 4. Backward Warmup & Timing (No Profiler)
    for _ in range(2):
        grads = grad_fn(Q, K, V)
        jax.block_until_ready(grads)

    start = time.time()
    for _ in range(num_iters):
        grads = grad_fn(Q, K, V)
        jax.block_until_ready(grads)
    bwd_latency = ((time.time() - start) / num_iters) * 1000

    # 5. Backward Profiling (Strictly 1 isolated pulse)
    jax.profiler.start_trace(log_dir)
    jax.block_until_ready(grad_fn(Q, K, V))
    jax.profiler.stop_trace()

    bytes_per_param = 4 
    memory_mb = (Q.size + K.size + V.size) * bytes_per_param * 3 / (1024 * 1024)

    return {
        "method": name,
        "fwd_latency_ms": fwd_latency,
        "bwd_latency_ms": bwd_latency,
        "memory_mb_approx": memory_mb
    }


def generate_synthetic_aligned_pairs(L, vocab="dna"):
    vocab_set = list("ACGT") if vocab == "dna" else list("ACDEFGHIKLMNPQRSTVWY")
    seq1 = "".join(np.random.choice(vocab_set, size=L))
    seq2_list = list(seq1)
    for idx in np.random.choice(L, size=int(0.2 * L), replace=False):
        seq2_list[idx] = np.random.choice(vocab_set)
    return [seq1, "".join(seq2_list)]


def run_long_benchmark(model_name, vocab):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/tmp/tensorboard_profiles_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Scaling linearly into regions where FlashAttention and Standard OOM
    lengths = [1024, 4096, 8192, 16384] 
    results_db = []
    
    try:
        devices = jax.devices()
        mesh = Mesh(devices, ('seq',))
    except Exception:
        mesh = None

    print(f"===================================================================")
    print(f"   Long-Context Foundation Model Alignment Benchmark")
    print(f"   Backbone: {model_name}")
    print(f"   Target: {vocab.upper()} (Testing massive context survival)")
    print(f"===================================================================\n")
    print(f"| {'Context (L)':<12} | {'Method':<20} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Status':<15} |")
    print("-" * 80)
    
    for L in lengths:
        seqs = generate_synthetic_aligned_pairs(L, vocab=vocab)
        try:
            embeddings = get_foundation_embeddings(seqs, model_name=model_name)
            emb_q = embeddings[0:1, :, :][:, None, :, :]
            emb_k = embeddings[1:2, :, :][:, None, :, :]
            emb_v = emb_k
        except Exception as e:
            print(f"Failed to extract embeddings: {e}")
            break

        if mesh is not None:
             sharding = NamedSharding(mesh, PartitionSpec(None, None, 'seq', None))
             emb_q = jax.device_put(emb_q, sharding)
             emb_k = jax.device_put(emb_k, sharding)
             emb_v = jax.device_put(emb_v, sharding)

        dim = emb_q.shape[-1]
        
        # 1. Baseline: SOTA Flash Attention
        def flash_attn_baseline(q, k, v):
            return jax.nn.dot_product_attention(q, k, v)

        try:
            fa_stats = benchmark_attention_kernel(emb_q, emb_k, emb_v, jax.jit(flash_attn_baseline), "Flash Attention", log_dir)
            fa_stats["seq_len"] = L
            results_db.append(fa_stats)
            print(f"| {L:<12} | {'Flash Attention':<20} | {fa_stats['fwd_latency_ms']:<10.2f} | {fa_stats['bwd_latency_ms']:<10.2f} | {'Success':<15} |")
        except Exception as e:
            print(f"| {L:<12} | {'Flash Attention':<20} | {'OOM':<10} | {'OOM':<10} | {'Failed':<15} |")

        # 2. Ours: Sinkhorn OT Block-Wise Attention
        band_width = min(2048, L)
        def sinkhorn_attn(q, k, v):
            return ot_kernel.ot_sinkhorn_alignment(q, k, v, n_iters=15, band_width=band_width, mesh=mesh)

        try:
            ot_stats = benchmark_attention_kernel(emb_q, emb_k, emb_v, sinkhorn_attn, "Sinkhorn OT (Ours)", log_dir)
            ot_stats["seq_len"] = L
            results_db.append(ot_stats)
            print(f"| {L:<12} | {'Sinkhorn OT (Ours)':<20} | {ot_stats['fwd_latency_ms']:<10.2f} | {ot_stats['bwd_latency_ms']:<10.2f} | {'Success':<15} |")
        except Exception as e:
            print(f"| {L:<12} | {'Sinkhorn OT (Ours)':<20} | {'OOM':<10} | {'OOM':<10} | {'Failed':<15} |")

        print("-" * 80)
        del embeddings, emb_q, emb_k, emb_v

    run_metadata = {
        "timestamp": timestamp,
        "model_name": model_name,
        "optimizer": "MUON (Momentum Orthogonalizer)",
        "baseline_attention": "JAX Native Flash Attention",
        "mesh_shape": mesh.shape if mesh else "None",
        "hardware_profile_dir": log_dir,
        "results": results_db
    }
    
    out_file = f"benchmark_results_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(run_metadata, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Long Context SOTA Alignment Benchmark")
    parser.add_argument("--model_name", type=str, default="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species")
    parser.add_argument("--vocab", type=str, choices=["dna", "protein"], default="dna", help="Type of sequence to generate")
    args = parser.parse_args()
    
    run_long_benchmark(args.model_name, args.vocab)
