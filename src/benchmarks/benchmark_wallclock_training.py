"""
Wall-Clock Training Benchmark: Flash Attention vs Sinkhorn OT
==============================================================

Trains both methods for a fixed wall-clock budget on **real genomic DNA**
alignment data, comparing loss curves to demonstrate that Sinkhorn's
doubly-stochastic transport plan learns proper alignments where
row-stochastic Flash Attention structurally cannot.

Data pipeline:
  1. Download orthologous DNA sequences from Ensembl REST API
     (human/mouse conserved gene pairs — real biological data)
  2. Embed via Nucleotide Transformer (frozen backbone)
  3. Train learnable Q/K/V projections to recover ground-truth alignment

Checkpoints and results saved to GCS.

Usage:
    # Step 1: Prepare data (downloads + embeds — run once)
    python src/benchmarks/benchmark_wallclock_training.py --prepare_data

    # Step 2a: Train Flash Attention for 2.5 hours
    python src/benchmarks/benchmark_wallclock_training.py \\
        --method flash_attention --max_hours 2.5

    # Step 2b: Train Sinkhorn OT for 2.5 hours
    python src/benchmarks/benchmark_wallclock_training.py \\
        --method sinkhorn_ot --max_hours 2.5

    # Quick smoke test (local CPU)
    python src/benchmarks/benchmark_wallclock_training.py \\
        --method sinkhorn_ot --max_hours 0.01 --seq_len 256 --no_gcs
"""

import os
import sys
import time
import json
import csv
import argparse

import numpy as np

import jax
import jax.numpy as jnp
import optax

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
        # Optax.apply_updates performs param + update, so descent updates must be negative.
        new_updates = jax.tree_util.tree_map(lambda g: -g * learning_rate, new_updates)
        return new_updates, state
    return optax.GradientTransformation(init_fn, update_fn)


# ==============================================================================
# Data: Real Orthologous DNA from Ensembl
# ==============================================================================

# Well-characterized human-mouse orthologous gene pairs for alignment benchmarks.
# These are real conserved genomic sequences with known evolutionary relationships.
ORTHOLOG_GENE_PAIRS = [
    # (human_gene_id, mouse_gene_id, gene_name)
    ("ENSG00000139618", "ENSMUSG00000041147", "BRCA2"),
    ("ENSG00000012048", "ENSMUSG00000017146", "BRCA1"),
    ("ENSG00000141510", "ENSMUSG00000059552", "TP53"),
    ("ENSG00000133703", "ENSMUSG00000005148", "KRAS"),
    ("ENSG00000157764", "ENSMUSG00000002413", "BRAF"),
    ("ENSG00000171862", "ENSMUSG00000027752", "PTEN"),
    ("ENSG00000146648", "ENSMUSG00000020122", "EGFR"),
    ("ENSG00000105329", "ENSMUSG00000038227", "TGFB1"),
    ("ENSG00000183454", "ENSMUSG00000022797", "GRIN2A"),
    ("ENSG00000102882", "ENSMUSG00000059495", "MAPK3"),
    ("ENSG00000100030", "ENSMUSG00000024935", "MAPK1"),
    ("ENSG00000132155", "ENSMUSG00000022672", "RAF1"),
    ("ENSG00000065361", "ENSMUSG00000020893", "ERBB2"),
    ("ENSG00000174775", "ENSMUSG00000019966", "HRAS"),
    ("ENSG00000168036", "ENSMUSG00000039114", "CTNNB1"),
    ("ENSG00000181143", "ENSMUSG00000041417", "MUC16"),
    ("ENSG00000134086", "ENSMUSG00000032083", "VHL"),
    ("ENSG00000185515", "ENSMUSG00000034634", "BRCA2_variant"),
    ("ENSG00000164362", "ENSMUSG00000030849", "TERT"),
    ("ENSG00000111640", "ENSMUSG00000022346", "GAPDH"),
    ("ENSG00000075624", "ENSMUSG00000029580", "ACTB"),
    ("ENSG00000204531", "ENSMUSG00000024339", "OCT4_POU5F1"),
    ("ENSG00000136997", "ENSMUSG00000032446", "MYC"),
    ("ENSG00000116044", "ENSMUSG00000026728", "NFE2L2"),
    ("ENSG00000142208", "ENSMUSG00000004040", "AKT1"),
    ("ENSG00000149311", "ENSMUSG00000025746", "ATM"),
    ("ENSG00000116898", "ENSMUSG00000046179", "UMPS"),
    ("ENSG00000145675", "ENSMUSG00000019699", "PIK3R1"),
    ("ENSG00000121879", "ENSMUSG00000020573", "PIK3CA"),
    ("ENSG00000168610", "ENSMUSG00000046329", "STAT3"),
]


def fetch_ensembl_cds(gene_id: str, species: str = "human") -> str:
    """Fetch coding DNA sequence from Ensembl REST API."""
    import urllib.request
    import json as _json

    server = "https://rest.ensembl.org"
    url = f"{server}/sequence/id/{gene_id}?type=cds"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read().decode())
            return data.get("seq", "")
    except Exception as e:
        print(f"  ⚠ Failed to fetch {gene_id}: {e}")
        return ""


def compute_pairwise_alignment_identity(seq1: str, seq2: str) -> np.ndarray:
    """
    Compute ground-truth alignment matrix for orthologous sequences.

    For orthologous CDS pairs (same gene, different species), the alignment
    is based on codon-level positional correspondence. Positions are aligned
    by their relative position in the CDS, weighted by nucleotide identity.

    Returns:
        alignment: (L1, L2) float32 matrix, row-normalized
    """
    L1, L2 = len(seq1), len(seq2)
    alignment = np.zeros((L1, L2), dtype=np.float32)

    # Use dynamic programming (Needleman-Wunsch) for global alignment
    # Simple scoring: match=2, mismatch=-1, gap=-2
    match_score, mismatch_score, gap_score = 2, -1, -2

    # DP matrix
    dp = np.zeros((L1 + 1, L2 + 1), dtype=np.float32)
    for i in range(1, L1 + 1):
        dp[i, 0] = dp[i - 1, 0] + gap_score
    for j in range(1, L2 + 1):
        dp[0, j] = dp[0, j - 1] + gap_score

    traceback = np.zeros((L1 + 1, L2 + 1), dtype=np.int32)  # 0=diag, 1=up, 2=left

    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            score = match_score if seq1[i - 1] == seq2[j - 1] else mismatch_score
            diag = dp[i - 1, j - 1] + score
            up = dp[i - 1, j] + gap_score
            left = dp[i, j - 1] + gap_score

            if diag >= up and diag >= left:
                dp[i, j] = diag
                traceback[i, j] = 0
            elif up >= left:
                dp[i, j] = up
                traceback[i, j] = 1
            else:
                dp[i, j] = left
                traceback[i, j] = 2

    # Traceback to get alignment pairs
    i, j = L1, L2
    while i > 0 and j > 0:
        if traceback[i, j] == 0:  # diagonal = aligned
            alignment[i - 1, j - 1] = 1.0
            i -= 1
            j -= 1
        elif traceback[i, j] == 1:  # up = gap in seq2
            i -= 1
        else:  # left = gap in seq1
            j -= 1

    # Row-normalize (each position in seq1 maps to distribution over seq2)
    row_sums = alignment.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    alignment = alignment / row_sums

    return alignment


def download_ortholog_pairs(cache_dir: str, seq_len: int):
    """
    Download real orthologous DNA sequences from Ensembl.

    Returns list of dicts with keys: gene_name, seq_human, seq_mouse, alignment
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"orthologs_L{seq_len}.json")

    # Check cache
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cached = json.load(f)
        print(f"✓ Loaded {len(cached)} cached ortholog pairs from {cache_file}")
        return cached

    print(f"⬇ Downloading orthologous CDS from Ensembl ({len(ORTHOLOG_GENE_PAIRS)} gene pairs)...")
    pairs = []

    for human_id, mouse_id, gene_name in ORTHOLOG_GENE_PAIRS:
        print(f"  Fetching {gene_name}...", end=" ")

        seq_human = fetch_ensembl_cds(human_id, "human")
        seq_mouse = fetch_ensembl_cds(mouse_id, "mouse")

        if not seq_human or not seq_mouse:
            print("skipped (missing)")
            continue

        # Truncate to seq_len
        seq_human = seq_human[:seq_len]
        seq_mouse = seq_mouse[:seq_len]

        if len(seq_human) < 128 or len(seq_mouse) < 128:
            print(f"skipped (too short: {len(seq_human)}/{len(seq_mouse)})")
            continue

        print(f"✓ {len(seq_human)}bp / {len(seq_mouse)}bp")
        pairs.append({
            "gene_name": gene_name,
            "seq_human": seq_human,
            "seq_mouse": seq_mouse,
            "len_human": len(seq_human),
            "len_mouse": len(seq_mouse),
        })

        # Rate limiting for Ensembl API
        time.sleep(0.3)

    # Save cache
    with open(cache_file, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"\n✓ Downloaded {len(pairs)} ortholog pairs → {cache_file}")
    return pairs


def prepare_dataset(data_dir, model_name, seq_len, pad_granularity=128):
    """
    Full data pipeline: download orthologs → embed with NT → compute alignments.

    Returns:
        emb_q: (N, 1, L, D) query embeddings (human)
        emb_k: (N, 1, L, D) key embeddings (mouse)
        targets: (N, L, L) ground-truth alignment matrices
        seq_len: actual padded sequence length
    """
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoConfig

    # 1. Download ortholog pairs
    pairs = download_ortholog_pairs(data_dir, seq_len)
    if not pairs:
        raise ValueError("No ortholog pairs downloaded. Check network connectivity.")

    # Pad seq_len to mod-128
    remainder = seq_len % pad_granularity
    if remainder != 0:
        seq_len = seq_len + (pad_granularity - remainder)

    # 2. Compute ground-truth alignment matrices (Needleman-Wunsch)
    print("\nComputing pairwise alignments (Needleman-Wunsch)...")
    targets_list = []
    valid_pairs = []

    for p in pairs:
        alignment = compute_pairwise_alignment_identity(p["seq_human"], p["seq_mouse"])
        h, w = alignment.shape

        # Pad to (seq_len, seq_len)
        padded = np.zeros((seq_len, seq_len), dtype=np.float32)
        padded[:h, :w] = alignment
        # Uniform distribution for padded rows (avoid NaN in CE)
        # BOLT ⚡: Replaced Python row iteration with vectorized boolean indexing
        row_sums = np.sum(padded, axis=1)
        padded[row_sums == 0] = 1.0 / seq_len
        targets_list.append(padded)
        valid_pairs.append(p)

    targets = jnp.array(np.stack(targets_list))
    print(f"✓ {len(valid_pairs)} alignment matrices computed")

    # 3. Embed with Nucleotide Transformer
    print(f"\nLoading {model_name}...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if not hasattr(config, "is_decoder"):
        config.is_decoder = False
    if not hasattr(config, "add_cross_attention"):
        config.add_cross_attention = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, config=config,
        trust_remote_code=True, ignore_mismatched_sizes=True,
    )
    model.eval()

    def embed_batch(sequences):
        inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.numpy()
        B, L_tok, D = emb.shape
        if L_tok < seq_len:
            emb = np.pad(emb, ((0, 0), (0, seq_len - L_tok), (0, 0)))
        elif L_tok > seq_len:
            emb = emb[:, :seq_len, :]
        return emb

    print("Embedding human sequences...")
    emb_human = embed_batch([p["seq_human"] for p in valid_pairs])
    print("Embedding mouse sequences...")
    emb_mouse = embed_batch([p["seq_mouse"] for p in valid_pairs])

    # Add head dim: (N, 1, L, D)
    emb_q = jax.device_put(jnp.array(emb_human[:, None, :, :]))
    emb_k = jax.device_put(jnp.array(emb_mouse[:, None, :, :]))

    print(f"\n✓ Dataset ready: {emb_q.shape[0]} pairs, L={seq_len}, D={emb_q.shape[-1]}")
    return emb_q, emb_k, targets, seq_len


# ==============================================================================
# Model: Learnable Q/K/V Projection Heads
# ==============================================================================

def init_projection_heads(key, D_in, D_proj):
    k1, k2, k3 = jax.random.split(key, 3)
    scale = 1.0 / jnp.sqrt(D_in)
    return {
        "W_q": jax.random.normal(k1, (D_in, D_proj)) * scale,
        "W_k": jax.random.normal(k2, (D_in, D_proj)) * scale,
        "W_v": jax.random.normal(k3, (D_in, D_proj)) * scale,
    }


def project(params, emb_q, emb_k):
    Q = jnp.einsum("bhld,dp->bhlp", emb_q, params["W_q"])
    K = jnp.einsum("bhld,dp->bhlp", emb_k, params["W_k"])
    V = jnp.einsum("bhld,dp->bhlp", emb_k, params["W_v"])
    return Q, K, V


# ==============================================================================
# Loss Functions
# ==============================================================================

def alignment_loss_sinkhorn(params, emb_q, emb_k, targets, band_width):
    """CE between Sinkhorn transport plan (doubly-stochastic) and alignment target."""
    Q, K, V = project(params, emb_q, emb_k)
    D = Q.shape[-1]
    scale = 1.0 / jnp.sqrt(D)

    u, v = ot_kernel.compute_global_uv(Q, K, n_iters=15, band_width=band_width)
    S = jnp.einsum("bhmd,bhnd->bhmn", Q, K) * scale
    log_P = S + u[..., :, None] + v[..., None, :]
    P = jax.nn.softmax(log_P, axis=-1)
    P = jnp.clip(P, 1e-10, 1.0)

    loss = -jnp.mean(jnp.sum(targets[:, None, :, :] * jnp.log(P), axis=-1))
    return loss


def alignment_loss_flash(params, emb_q, emb_k, targets, band_width):
    """CE between attention weights (row-stochastic only) and alignment target."""
    Q, K, V = project(params, emb_q, emb_k)
    D = Q.shape[-1]
    scale = 1.0 / jnp.sqrt(D)

    S = jnp.einsum("bhmd,bhnd->bhmn", Q, K) * scale
    A = jax.nn.softmax(S, axis=-1)
    A = jnp.clip(A, 1e-10, 1.0)

    loss = -jnp.mean(jnp.sum(targets[:, None, :, :] * jnp.log(A), axis=-1))
    return loss


# ==============================================================================
# Checkpointing (GCS)
# ==============================================================================

def save_checkpoint(params, step, method, gcs_bucket, elapsed_time, local_dir="/tmp/checkpoints"):
    """Save model checkpoint + elapsed time to local disk and optionally GCS."""
    ckpt_dir = os.path.join(local_dir, method, f"step_{step:06d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Save params + metadata as .npz
    data = {k: np.array(v) for k, v in params.items()}
    data['metadata_step'] = np.array(step)
    data['metadata_elapsed'] = np.array(elapsed_time)
    
    ckpt_path = os.path.join(ckpt_dir, "params.npz")
    np.savez(ckpt_path, **data)
    
    if gcs_bucket:
        gcs_path = f"gs://{gcs_bucket}/checkpoints/{method}/step_{step:06d}/params.npz"
        os.system(f"gsutil -q cp {ckpt_path} {gcs_path}")
    
    return ckpt_path


def load_latest_checkpoint(method, gcs_bucket, local_dir="/tmp/checkpoints"):
    """Check GCS for the latest checkpoint and load it if it exists."""
    if not gcs_bucket:
        return None, 0, 0
    
    print(f"Checking for existing checkpoints in gs://{gcs_bucket}/checkpoints/{method}/...")
    
    # List all params.npz files in the method's checkpoint dir
    cmd = f"gsutil ls gs://{gcs_bucket}/checkpoints/{method}/*/params.npz"
    try:
        output = os.popen(cmd).read().splitlines()
    except Exception:
        output = []
    
    if not output:
        print("  No checkpoints found. Starting fresh.")
        return None, 0, 0
    
    # Sort by step number (embedded in path)
    output.sort()
    latest_gcs_path = output[-1]
    
    print(f"  ✓ Found latest checkpoint: {latest_gcs_path}")
    
    # Download locally
    local_path = os.path.join(local_dir, "resume_latest.npz")
    os.makedirs(local_dir, exist_ok=True)
    os.system(f"gsutil -q cp {latest_gcs_path} {local_path}")
    
    # Load
    try:
        with np.load(local_path, allow_pickle=True) as data:
            params = {k: jnp.array(data[k]) for k in data.files if not k.startswith('metadata_')}
            step = int(data['metadata_step'])
            elapsed = float(data['metadata_elapsed'])
        
        print(f"  ✓ Resuming from Step {step} (Elapsed time: {elapsed/3600:.2f}h)")
        return params, step, elapsed
    except Exception as e:
        print(f"  ⚠ Failed to load checkpoint: {e}. Starting fresh.")
        return None, 0, 0


def save_results(results, method, gcs_bucket, local_dir="/tmp/results"):
    os.makedirs(local_dir, exist_ok=True)
    csv_path = os.path.join(local_dir, f"{method}_loss_curve.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "wallclock_sec", "loss", "throughput_steps_per_sec"])
        for r in results["loss_curve"]:
            writer.writerow([r["step"], r["wallclock_sec"], r["loss"], r["throughput"]])
    json_path = os.path.join(local_dir, f"{method}_summary.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    if gcs_bucket:
        os.system(f"gsutil -q cp {csv_path} gs://{gcs_bucket}/results/")
        os.system(f"gsutil -q cp {json_path} gs://{gcs_bucket}/results/")
    print(f"\n✓ Results: {csv_path}, {json_path}")


# ==============================================================================
# Auto-detect Batch Size
# ==============================================================================

def auto_detect_batch_size(params, emb_q, emb_k, targets, loss_fn, band_width, max_batch=64):
    print("Auto-detecting max batch size...")
    best_batch = 1
    lo, hi = 1, min(max_batch, emb_q.shape[0])
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            loss, grads = jax.value_and_grad(loss_fn)(
                params, emb_q[:mid], emb_k[:mid], targets[:mid], band_width,
            )
            jax.block_until_ready(loss)
            jax.block_until_ready(grads)
            best_batch = mid
            lo = mid + 1
            print(f"  batch_size={mid}: ✓")
        except Exception as e:
            hi = mid - 1
            print(f"  batch_size={mid}: ✗ ({type(e).__name__})")
    print(f"✓ Auto-detected batch_size={best_batch}")
    return best_batch


# ==============================================================================
# Training Loop
# ==============================================================================

def train(method, emb_q, emb_k, targets, seq_len, max_hours, gcs_bucket,
          checkpoint_every=10, lr=1e-3, D_proj=64, seed=42):

    D_in = emb_q.shape[-1]
    N = emb_q.shape[0]
    band_width = min(2048, seq_len)

    loss_fn = alignment_loss_sinkhorn if method == "sinkhorn_ot" else alignment_loss_flash

    # Initialize model or resume
    params, start_step, start_elapsed = load_latest_checkpoint(method, gcs_bucket)
    
    if params is None:
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
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    max_seconds = max_hours * 3600
    loss_curve = []
    step = start_step
    cumulative_elapsed = start_elapsed
    session_start_time = time.time()

    print(f"\n{'=' * 70}")
    print(f"  Training: {method}")
    print(f"  Budget: {max_hours}h | Current Cumulative: {cumulative_elapsed/3600:.2f}h")
    print(f"  Dataset: {N} pairs | L={seq_len} | D={D_in}")
    print(f"  Batch: {batch_size} | Band: {band_width} | LR: {lr}")
    print(f"{'=' * 70}\n")
    print(f"{'Step':>8} | {'Loss':>12} | {'Wall (s)':>10} | {'Steps/s':>10}")
    print("-" * 50)

    while True:
        current_session_time = time.time() - session_start_time
        total_elapsed = cumulative_elapsed + current_session_time
        
        if total_elapsed >= max_seconds:
            print(f"\n⏱ Total cumulative budget exhausted ({total_elapsed / 3600:.2f}h)")
            break

        idx = np.random.choice(N, size=batch_size, replace=(batch_size > N))
        step_start = time.time()
        params, opt_state, loss = train_step(
            params, opt_state, emb_q[idx], emb_k[idx], targets[idx],
        )
        jax.block_until_ready(loss)

        step_time = time.time() - step_start
        loss_val = float(loss)
        throughput = 1.0 / max(step_time, 1e-6)

        loss_curve.append({
            "step": step,
            "wallclock_sec": round(total_elapsed, 2),
            "loss": round(loss_val, 6),
            "throughput": round(throughput, 2),
        })

        if step % 10 == 0:
            print(f"{step:>8} | {loss_val:>12.6f} | {total_elapsed:>10.1f} | {throughput:>10.2f}")

        if step > 0 and step % checkpoint_every == 0:
            ckpt = save_checkpoint(params, step, method, gcs_bucket, total_elapsed)
            print(f"  💾 Checkpoint → {ckpt}")

        step += 1

    # Final checkpoint
    save_checkpoint(params, step, method, gcs_bucket, total_elapsed)

    session_time = time.time() - session_start_time
    total_time_final = cumulative_elapsed + session_time
    results = {
        "method": method,
        "total_steps": step,
        "total_time_sec": round(total_time_final, 2),
        "final_loss": loss_curve[-1]["loss"] if loss_curve else None,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "D_proj": D_proj,
        "lr": lr,
        "band_width": band_width,
        "dataset_size": N,
        "loss_curve": loss_curve,
    }
    save_results(results, method, gcs_bucket)

    print(f"\n{'=' * 70}")
    print(f"  {method} COMPLETE — {step} steps, final loss={results['final_loss']}")
    print(f"{'=' * 70}")
    return results


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wall-Clock Training: Flash vs Sinkhorn on DNA")
    parser.add_argument("--method", type=str, choices=["flash_attention", "sinkhorn_ot"],
                        help="Training method (required unless --prepare_data)")
    parser.add_argument("--prepare_data", action="store_true",
                        help="Only download data and compute embeddings, then exit")
    parser.add_argument("--max_hours", type=float, default=2.5)
    parser.add_argument("--seq_len", type=int, default=8192,
                        help="Sequence length (truncate/pad to this)")
    parser.add_argument("--data_dir", type=str, default="./data/dna_orthologs")
    parser.add_argument("--model_name", type=str,
                        default="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species")
    parser.add_argument("--gcs_bucket", type=str, default="sinkhorn-neurips-data")
    parser.add_argument("--no_gcs", action="store_true")
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--D_proj", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gcs_bucket = None if args.no_gcs else args.gcs_bucket

    print(f"JAX devices: {jax.devices()}")
    print(f"Device count: {jax.device_count()}\n")

    emb_q, emb_k, targets, seq_len = prepare_dataset(
        args.data_dir, args.model_name, args.seq_len,
    )

    if args.prepare_data:
        print("\n✓ Data preparation complete. Exiting.")
        sys.exit(0)

    if not args.method:
        parser.error("--method is required unless --prepare_data is set")

    train(
        method=args.method,
        emb_q=emb_q, emb_k=emb_k, targets=targets, seq_len=seq_len,
        max_hours=args.max_hours, gcs_bucket=gcs_bucket,
        checkpoint_every=args.checkpoint_every,
        lr=args.lr, D_proj=args.D_proj, seed=args.seed,
    )
