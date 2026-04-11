"""
Pfam Seed Alignment Data Loader
================================

Downloads Pfam-A.seed (Stockholm format MSAs) and extracts pairwise
training pairs with ground-truth alignment matrices derived from MSA columns.

Usage:
    python data/download_pfam.py --output_dir ./data/pfam_pairs --embed
"""

import os
import gzip
import json
import random
import argparse
import urllib.request
import numpy as np

PFAM_SEED_URL = "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.seed.gz"
PFAM_SEED_FILENAME = "Pfam-A.seed.gz"


def download_pfam_seed(cache_dir: str) -> str:
    """Download Pfam-A.seed.gz if not already cached."""
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, PFAM_SEED_FILENAME)
    if os.path.exists(local_path):
        print(f"✓ Using cached {local_path}")
        return local_path
    print("⬇ Downloading Pfam-A.seed.gz from EBI (~200MB)...")
    urllib.request.urlretrieve(PFAM_SEED_URL, local_path)
    print(f"✓ Saved to {local_path}")
    return local_path


def parse_stockholm_families(gz_path: str, min_seqs: int = 2):
    """Parse a gzipped Stockholm file into families."""
    family_id = None
    sequences = []
    with gzip.open(gz_path, 'rt', encoding='latin-1') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('# STOCKHOLM'):
                family_id, sequences = None, []
            elif line.startswith('#=GF AC'):
                parts = line.split()
                if len(parts) >= 3:
                    family_id = parts[2].split('.')[0]
            elif line.startswith('//'):
                if family_id and len(sequences) >= min_seqs:
                    yield {'id': family_id, 'sequences': sequences}
                family_id, sequences = None, []
            elif not line.startswith('#') and line.strip():
                parts = line.split(None, 1)
                if len(parts) == 2:
                    sequences.append((parts[0], parts[1]))


def extract_pairwise_alignment(seq1_aligned: str, seq2_aligned: str):
    """Derive ground-truth alignment matrix from MSA columns."""
    gap_chars = set('.-~')
    raw1, raw2, pairs = [], [], []
    i, j = 0, 0
    for col in range(len(seq1_aligned)):
        c1, c2 = seq1_aligned[col], seq2_aligned[col]
        is_g1, is_g2 = c1 in gap_chars, c2 in gap_chars
        if not is_g1 and not is_g2:
            pairs.append((i, j))
            raw1.append(c1); raw2.append(c2)
            i += 1; j += 1
        elif not is_g1:
            raw1.append(c1); i += 1
        elif not is_g2:
            raw2.append(c2); j += 1
    if i == 0 or j == 0: return None, None, None
    mat = np.zeros((i, j), dtype=np.float32)
    for r, c in pairs: mat[r, c] = 1.0
    return ''.join(raw1), ''.join(raw2), mat


def process_pfam_to_pairs(
    gz_path: str, output_dir: str, 
    min_len: int = 128, max_len: int = 1024,
    max_pairs_per_family: int = 5, max_total_pairs: int = 50000, 
    seed: int = 42, embed: bool = False,
    model_name: str = "facebook/esm2_t6_8M_UR50D"
):
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    
    tokenizer, model = None, None
    if embed:
        print(f"Loading {model_name} for pre-embedding...")
        import torch
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

    pair_count, family_count = 0, 0
    metadata = []
    
    for family in parse_stockholm_families(gz_path):
        if pair_count >= max_total_pairs: break
        
        valid_seqs = []
        gap_chars = set('.-~')
        for name, aligned in family['sequences']:
            raw = ''.join(c for c in aligned if c not in gap_chars)
            if min_len <= len(raw) <= max_len:
                valid_seqs.append((name, aligned, raw))
        
        if len(valid_seqs) < 2: continue
        family_count += 1
        rng.shuffle(valid_seqs)
        f_pairs = 0
        
        for i in range(0, len(valid_seqs) - 1, 2):
            if f_pairs >= max_pairs_per_family or pair_count >= max_total_pairs: break
            
            s1, s2, mat = extract_pairwise_alignment(valid_seqs[i][1], valid_seqs[i+1][1])
            if s1 is None: continue
            
            save_dict = {
                "seq1": np.array(list(s1)), "seq2": np.array(list(s2)),
                "alignment": mat.astype(np.float16)
            }
            if embed:
                tokens = tokenizer([s1, s2], return_tensors="pt", padding=True)
                with torch.no_grad():
                    outs = model(**tokens)
                embs = outs.last_hidden_state.numpy()
                save_dict["emb1"] = embs[0, :len(s1), :].astype(np.float32)
                save_dict["emb2"] = embs[1, :len(s2), :].astype(np.float32)

            pair_file = os.path.join(output_dir, f"pair_{pair_count:05d}.npz")
            np.savez_compressed(pair_file, **save_dict)
            metadata.append({
                'pair_id': pair_count, 'family': family['id'],
                'len1': len(s1), 'len2': len(s2), 'file': f"pair_{pair_count:05d}.npz"
            })
            pair_count += 1; f_pairs += 1
        
        if family_count % 100 == 0:
            print(f"  ... {family_count} families, {pair_count} pairs extracted")

    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump({'total_pairs': pair_count, 'pairs': metadata}, f, indent=2)
    print(f"✓ Created {pair_count} pairs in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./data/pfam_pairs")
    parser.add_argument("--cache_dir", type=str, default="./data/cache")
    parser.add_argument("--min_len", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--max_pairs", type=int, default=50000)
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    
    path = download_pfam_seed(args.cache_dir)
    process_pfam_to_pairs(
        path, args.output_dir, min_len=args.min_len, 
        max_len=args.max_len, max_total_pairs=args.max_pairs, 
        embed=args.embed
    )
