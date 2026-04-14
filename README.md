# Block-Wise Differentiable Sinkhorn Attention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/Framework-JAX-blue.svg)](https://github.com/google/jax)
[![Hardware](https://img.shields.io/badge/Hardware-TPU%20v6e-red.svg)](https://cloud.google.com/tpu/docs/v6e)

**Scaling differentiable optimal-transport alignment to long biological contexts.**

This repository contains the reference implementation of **Block-Wise Differentiable Sinkhorn Attention** for long-context sequence alignment. The current codebase supports both research-grade numerical validation and high-performance TPU execution paths.

---

## What is in the code right now

### Core kernel (`src/ot_kernel.py`)

- **Banded Sinkhorn forward pass** with alternating dual updates (`u`, `v`) and optional masks.
- **Custom-VJP tail-refinement backward** with:
  - optimized **K=2 staircase transport-plan** path,
  - fixed-depth refinement path,
  - adaptive residual-tolerance fallback via exact JAX pullback.
- **Two gradient modes** exposed through `sinkhorn_attention(...)`:
  - `tail_refinement` (default; custom VJP path),
  - `exact_autodiff` (reference validation path).
- **Entropic OT controls**:
  - `epsilon` temperature,
  - optional epsilon scaling schedule (`epsilon_scaling_steps`, `epsilon_scaling_factor`).
- **Long-context efficiency controls**:
  - block-wise tiling,
  - configurable `band_width` for `O(L · W)` style complexity.

### Benchmark and training scripts

- `src/benchmarks/benchmark_long_context_alignment.py`
  - latency benchmark suite for long-context DNA/protein settings,
  - compares Sinkhorn OT vs Flash Attention,
  - intended for 1K → 16K+ context scaling experiments.
- `src/benchmarks/benchmark_wallclock_training.py`
  - fixed wall-clock training comparison on real human/mouse orthologous CDS,
  - uses Needleman–Wunsch-derived pairwise alignment targets,
  - compares convergence behavior of Sinkhorn OT vs Flash Attention.
- `src/benchmarks/benchmark_pfam_training.py`
  - supervised PFAM alignment training utilities,
  - optional **Stiefel QR projection** on Q/K heads,
  - optional monotonicity and sparse CE terms,
  - configurable Sinkhorn refinement controls.

---

## Installation

```bash
git clone https://github.com/dylanbforde/sinkhorn-aligner.git
cd sinkhorn-aligner

# Recommended
uv sync

# Or
pip install .
```

---

## Quick usage

```python
import jax.numpy as jnp
from src.ot_kernel import sinkhorn_attention

Q = jnp.ones((1, 8, 4096, 64))
K = jnp.ones((1, 8, 4096, 64))
V = jnp.ones((1, 8, 4096, 64))

out = sinkhorn_attention(
    Q, K, V,
    n_iters=15,
    band_width=2048,
    gradient_mode="tail_refinement",  # or "exact_autodiff"
    refine_steps=2,
    epsilon=1.0,
)
```

---

## Benchmarks already run (repository-reported)

### 1) Long-context latency benchmark

The repository currently reports TPU v6e-1 latency trends (Table 1 style) showing stable scaling of the Sinkhorn kernel up to 16K context lengths:

| Context (L) | Method | Fwd (ms) | Bwd (ms) | Status |
| :--- | :--- | :--- | :--- | :--- |
| 1024 | Flash Attention | 0.14 | 1.54 | Success |
| 1024 | Sinkhorn OT (Ours) | 363.24 | 4270.36 | Success |
| 4096 | Sinkhorn OT (Ours) | 525.13 | 5033.42 | Success |
| 8192 | Sinkhorn OT (Ours) | 711.66 | 5662.79 | Success |
| 16384 | Sinkhorn OT (Ours) | 1264.84 | 7401.55 | Success |

### 2) Wall-clock training benchmark

`docs/GCP_DEPLOYMENT_GUIDE.md` documents a **2.5h vs 2.5h** training protocol:
- train Sinkhorn OT for 2.5 hours,
- train Flash Attention for 2.5 hours,
- compare loss curves under equal wall-clock budget on real orthologous DNA data.

---

## Reproducibility commands

### Long-context latency (DNA)

```bash
python src/benchmarks/benchmark_long_context_alignment.py \
  --model_name "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species" \
  --vocab dna
```

### Long-context latency (protein)

```bash
python src/benchmarks/benchmark_long_context_alignment.py \
  --model_name "facebook/esm2_t6_8M_UR50D" \
  --vocab protein
```

### Wall-clock training

```bash
# Prepare data
python src/benchmarks/benchmark_wallclock_training.py --prepare_data

# 2.5h Flash baseline
python src/benchmarks/benchmark_wallclock_training.py --method flash_attention --max_hours 2.5

# 2.5h Sinkhorn
python src/benchmarks/benchmark_wallclock_training.py --method sinkhorn_ot --max_hours 2.5
```

---

## Current development focus (updated roadmap)

1. **Stabilize and standardize pairwise benchmark reporting**
   - keep wall-clock pairwise as the core regression gate for correctness + training stability,
   - add clearer seed-averaged summaries and metric dashboards.

2. **Translate benchmark wins toward biological/clinical relevance**
   - move from pairwise-only evaluation toward multi-sequence and structure-aware settings,
   - prioritize tasks where long-context alignment quality affects downstream interpretability.

3. **Next major milestones**
   - MSA-aware objectives and evaluation,
   - structure-linked validation (contacts/folds/functional constraints),
   - larger real-world cohorts after pairwise benchmark hardening.

---

## Scientific background

See the NeurIPS manuscript for derivations and proofs of the block-wise Sinkhorn formulation and the truncated multi-plan backward pass.

```bibtex
@inproceedings{forde2024sinkhorn,
  title={Block-Wise Differentiable Sinkhorn Attention: Scaling Optimal Transport to Massive Contexts},
  author={Forde, Dylan B.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

---

## License

MIT. See [LICENSE](LICENSE).
