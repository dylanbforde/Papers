"""
Pfam Seed Alignment Data Loader
================================

Downloads Pfam-A.seed (Stockholm format MSAs) and extracts pairwise
training pairs with ground-truth alignment matrices derived from MSA columns.

The processed dataset is cached locally and optionally synced to GCS.

Usage:
    python data/download_pfam.py --output_dir ./data/pfam_pairs --min_len 128 --max_len 2048
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
    """
    Parse a gzipped Stockholm file (Pfam seed) into families.
    
    Yields: dict with keys:
        - 'id': Pfam family accession (e.g. PF00001)
        - 'sequences': list of (name, aligned_sequence) tuples
    """
    family_id = None
    sequences = []
    
    with gzip.open(gz_path, 'rt', encoding='latin-1') as f:
        for line in f:
            line = line.rstrip('\n')
            
            if line.startswith('# STOCKHOLM'):
                family_id = None
                sequences = []
            elif line.startswith('#=GF AC'):
                # Extract accession: "#=GF AC   PF00001.23" -> "PF00001"
                parts = line.split()
                if len(parts) >= 3:
                    family_id = parts[2].split('.')[0]
            elif line.startswith('//'):
                # End of family block
                if family_id and len(sequences) >= min_seqs:
                    yield {
                        'id': family_id,
                        'sequences': sequences
                    }
                family_id = None
                sequences = []
            elif not line.startswith('#') and line.strip():
                # Sequence line: "name/start-end    ALIGNED-SEQ"
                parts = line.split(None, 1)
                if len(parts) == 2:
                    name, aligned_seq = parts
                    sequences.append((name, aligned_seq))


def extract_pairwise_alignment(seq1_aligned: str, seq2_aligned: str):
    """
    Given two aligned sequences from an MSA (same number of columns),
    derive the ground-truth pairwise alignment matrix.
    
    Returns:
        raw_seq1: str - ungapped sequence 1
        raw_seq2: str - ungapped sequence 2
        alignment_matrix: np.ndarray of shape (len1, len2) - binary alignment matrix
    """
    # Standard gap characters in Stockholm format
    gap_chars = set('.-~')
    
    raw_seq1_chars = []
    raw_seq2_chars = []
    pairs = []  # (i, j) pairs where i indexes seq1, j indexes seq2
    
    idx1, idx2 = 0, 0
    for col in range(len(seq1_aligned)):
        c1 = seq1_aligned[col]
        c2 = seq2_aligned[col]
        
        is_gap1 = c1 in gap_chars
        is_gap2 = c2 in gap_chars
        
        if not is_gap1 and not is_gap2:
            # Both have residues at this column -> aligned pair
            pairs.append((idx1, idx2))
            raw_seq1_chars.append(c1)
            raw_seq2_chars.append(c2)
            idx1 += 1
            idx2 += 1
        elif not is_gap1:
            raw_seq1_chars.append(c1)
            idx1 += 1
        elif not is_gap2:
            raw_seq2_chars.append(c2)
            idx2 += 1
    
    len1, len2 = idx1, idx2
    if len1 == 0 or len2 == 0:
        return None, None, None
    
    alignment_matrix = np.zeros((len1, len2), dtype=np.float32)
    for i, j in pairs:
        alignment_matrix[i, j] = 1.0
    
    # Normalize rows so each row sums to 1 (or 0 for unaligned positions)
    row_sums = alignment_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    alignment_matrix = alignment_matrix / row_sums
    
    return ''.join(raw_seq1_chars), ''.join(raw_seq2_chars), alignment_matrix


def process_pfam_to_pairs(
    gz_path: str,
    output_dir: str,
    min_len: int = 128,
    max_len: int = 2048,
    max_pairs_per_family: int = 3,
    max_total_pairs: int = 5000,
    seed: int = 42,
):
    """
    Process Pfam seed families into pairwise alignment training pairs.
    
    Filters families by sequence length and saves pairs with ground-truth
    alignment matrices as .npz files.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    
    pair_count = 0
    family_count = 0
    metadata = []
    
    print(f"Processing Pfam families (min_len={min_len}, max_len={max_len})...")
    
    for family in parse_stockholm_families(gz_path, min_seqs=2):
        if pair_count >= max_total_pairs:
            break
        
        # Filter sequences by length (ungapped)
        gap_chars = set('.-~')
        valid_seqs = []
        for name, aligned in family['sequences']:
            ungapped = ''.join(c for c in aligned if c not in gap_chars)
            if min_len <= len(ungapped) <= max_len:
                valid_seqs.append((name, aligned, ungapped))
        
        if len(valid_seqs) < 2:
            continue
        
        family_count += 1
        
        # Sample pairs from this family
        rng.shuffle(valid_seqs)
        pairs_from_family = 0
        
        for i in range(0, len(valid_seqs) - 1, 2):
            if pairs_from_family >= max_pairs_per_family:
                break
            if pair_count >= max_total_pairs:
                break
            
            name1, aligned1, raw1 = valid_seqs[i]
            name2, aligned2, raw2 = valid_seqs[i + 1]
            
            seq1, seq2, align_mat = extract_pairwise_alignment(aligned1, aligned2)
            if seq1 is None:
                continue
            
            # Save pair
            pair_file = os.path.join(output_dir, f"pair_{pair_count:05d}.npz")
            np.savez_compressed(
                pair_file,
                seq1=np.array(list(seq1)),
                seq2=np.array(list(seq2)),
                alignment=align_mat,
            )
            
            metadata.append({
                'pair_id': pair_count,
                'family': family['id'],
                'name1': name1,
                'name2': name2,
                'len1': len(seq1),
                'len2': len(seq2),
                'n_aligned': int(align_mat.sum()),
                'file': f"pair_{pair_count:05d}.npz",
            })
            
            pair_count += 1
            pairs_from_family += 1
        
        if family_count % 500 == 0:
            print(f"  ... {family_count} families processed, {pair_count} pairs extracted")
    
    # Save metadata
    meta_file = os.path.join(output_dir, "metadata.json")
    with open(meta_file, 'w') as f:
        json.dump({
            'total_pairs': pair_count,
            'total_families': family_count,
            'min_len': min_len,
            'max_len': max_len,
            'seed': seed,
            'pairs': metadata,
        }, f, indent=2)
    
    print(f"\n✓ Done: {pair_count} pairs from {family_count} families → {output_dir}")
    print(f"  Metadata: {meta_file}")
    
    # Length distribution summary
    if metadata:
        lens = [m['len1'] for m in metadata] + [m['len2'] for m in metadata]
        print(f"  Length stats: min={min(lens)}, median={sorted(lens)[len(lens)//2]}, max={max(lens)}")
    
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process Pfam seed alignments")
    parser.add_argument("--output_dir", type=str, default="./data/pfam_pairs")
    parser.add_argument("--cache_dir", type=str, default="./data/cache")
    parser.add_argument("--min_len", type=int, default=128, help="Minimum ungapped sequence length")
    parser.add_argument("--max_len", type=int, default=2048, help="Maximum ungapped sequence length")
    parser.add_argument("--max_pairs", type=int, default=5000, help="Maximum total pairs to extract")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    gz_path = download_pfam_seed(args.cache_dir)
    process_pfam_to_pairs(
        gz_path, args.output_dir,
        min_len=args.min_len, max_len=args.max_len,
        max_total_pairs=args.max_pairs, seed=args.seed,
    )
