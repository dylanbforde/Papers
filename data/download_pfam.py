"""
Pfam Sharded Data Pipeline (Production Grade)
============================================

Downloads, aligns, and embeds 50,000+ Pfam protein pairs into compressed shards.
Optimized for high-throughput TPU training on A-grade hardware (v6e-8).

Output Schema (Shards):
    - emb1, emb2: (N, L, D) float16
    - alignment:  (N, L, L) float16
    - mask1, mask2: (N, L) bool (padding info)

Usage:
    python data/download_pfam.py --shard_size 500 --max_pairs 50000 --embed
"""

import importlib
import os
import gzip
import json
import random
import argparse
import urllib.request
from typing import Any

import numpy as np

PFAM_SEED_URL = "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.seed.gz"
PFAM_SEED_FILENAME = "Pfam-A.seed.gz"

GAP_CHARS = set(".-~")
GAP_TRANS = str.maketrans("", "", ".-~")

def download_pfam_seed(cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, PFAM_SEED_FILENAME)
    if os.path.exists(local_path):
        print(f"✓ Using cached {local_path}")
        return local_path
    print("⬇ Downloading Pfam-A.seed.gz (~200MB)...")
    urllib.request.urlretrieve(PFAM_SEED_URL, local_path)
    return local_path

def parse_stockholm_families(gz_path: str):
    family_id = None
    sequences = []
    with gzip.open(gz_path, "rt", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# STOCKHOLM"):
                family_id, sequences = None, []
            elif line.startswith("#=GF AC"):
                parts = line.split()
                if len(parts) >= 3:
                    family_id = parts[2].split(".")[0]
            elif line.startswith("//"):
                if family_id and len(sequences) >= 2:
                    yield {"id": family_id, "sequences": sequences}
                family_id, sequences = None, []
            elif line and not line.startswith("#"):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    sequences.append((parts[0], parts[1]))

def extract_alignment(s1, s2):
    # Performance: Use module-level GAP_CHARS to avoid redundant set creation
    r1, r2, pairs = [], [], []
    i, j = 0, 0
    for col in range(len(s1)):
        c1, c2 = s1[col], s2[col]
        g1, g2 = c1 in GAP_CHARS, c2 in GAP_CHARS
        if not g1 and not g2:
            pairs.append((i, j))
            r1.append(c1)
            r2.append(c2)
            i += 1
            j += 1
        elif not g1:
            r1.append(c1)
            i += 1
        elif not g2:
            r2.append(c2)
            j += 1
    if i == 0 or j == 0:
        return None, None, None
    mat = np.zeros((i, j), dtype=np.float16)
    for r, c in pairs:
        mat[r, c] = 1.0
    return "".join(r1), "".join(r2), mat

def _load_embedder() -> tuple[Any, Any, Any]:
    model_name = "facebook/esm2_t6_8M_UR50D"
    torch_module = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModel.from_pretrained(model_name)
    model.eval()
    return torch_module, tokenizer, model

def process_sharded(gz_path, out_dir, shard_size=500, max_pairs=50000, max_len=1024, embed=False):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(42)

    torch_module: Any | None = None
    tokenizer: Any | None = None
    model: Any | None = None
    if embed:
        print("Loading ESM-2 8M for production embedding...")
        torch_module, tokenizer, model = _load_embedder()

    pair_count, shard_count = 0, 0
    buffer = []

    def flush_shard(shard_data):
        nonlocal shard_count
        name = f"shard_{shard_count:03d}.npz"
        path = os.path.join(out_dir, name)

        # Stack into fixed-size tensors (B, L, D)
        N = len(shard_data)
        emb1 = np.zeros((N, max_len, 320), dtype=np.float16)
        emb2 = np.zeros((N, max_len, 320), dtype=np.float16)
        targets = np.zeros((N, max_len, max_len), dtype=np.float16)

        for i, pair in enumerate(shard_data):
            l1, l2 = pair["emb1"].shape[0], pair["emb2"].shape[0]
            emb1[i, :l1, :] = pair["emb1"]
            emb2[i, :l2, :] = pair["emb2"]
            targets[i, :l1, :l2] = pair["target"]

        np.savez_compressed(path, emb1=emb1, emb2=emb2, target=targets)
        print(f" ✓ Saved Shard {shard_count} -> {name}")

        if args.gcs_bucket:
            gcs_path = f"gs://{args.gcs_bucket}/pfam_shards/{name}"
            print(f" ⬆ Uploading to {gcs_path}...")
            os.system(f"gsutil cp {path} {gcs_path}")

        shard_count += 1

    for family in parse_stockholm_families(gz_path):
        if pair_count >= max_pairs:
            break
        valid = []
        for name, ali in family["sequences"]:
            # Performance: str.translate is significantly faster than generator expressions
            raw = ali.translate(GAP_TRANS)
            if 128 <= len(raw) <= max_len:
                valid.append((name, ali, raw))

        if len(valid) < 2:
            continue
        rng.shuffle(valid)

        for i in range(0, len(valid) - 1, 2):
            if pair_count >= max_pairs:
                break
            s1, s2, mat = extract_alignment(valid[i][1], valid[i + 1][1])
            if s1 is None:
                continue

            p = {"target": mat}
            if embed:
                assert tokenizer is not None and model is not None and torch_module is not None
                t = tokenizer([s1, s2], return_tensors="pt", padding=True)
                with torch_module.no_grad():
                    e = model(**t).last_hidden_state.numpy().astype(np.float16)
                p["emb1"], p["emb2"] = e[0, : len(s1), :], e[1, : len(s2), :]

            buffer.append(p)
            pair_count += 1
            if len(buffer) >= shard_size:
                flush_shard(buffer)
                buffer = []

    if buffer:
        flush_shard(buffer)

    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_pairs": pair_count,
                "shard_size": shard_size,
                "total_shards": shard_count,
            },
            f,
        )

    if args.gcs_bucket:
        os.system(f"gsutil cp {meta_path} gs://{args.gcs_bucket}/pfam_shards/metadata.json")
    print(f"✓ Pipeline Complete: {pair_count} pairs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data/pfam_shards")
    parser.add_argument("--max_pairs", type=int, default=50000)
    parser.add_argument("--shard_size", type=int, default=500)
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--gcs_bucket", type=str, help="Upload shards to this GCS bucket")
    args = parser.parse_args()
    
    path = download_pfam_seed("./data/cache")
    process_sharded(path, args.output_dir, args.shard_size, args.max_pairs, embed=args.embed)
