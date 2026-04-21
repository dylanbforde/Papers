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
- `src/benchmarks/benchmark_pfam_training.py`
  - supervised Pfam alignment training utilities,
  - optional **Stiefel QR projection** on Q/K heads,
  - optional monotonicity and sparse CE terms,
  - configurable Sinkhorn refinement controls.
- `scripts/prepare_data.sh`
  - offline Pfam shard generation and upload to GCS,
  - keeps expensive TPU time focused on training instead of embedding.
- `scripts/tpu_sinkhorn_screen.sh`
  - queues a TPU v6e-8 slice,
  - runs a short Sinkhorn-only config screen on Pfam shards,
  - syncs logs/checkpoints to GCS before teardown.
- `scripts/tpu_k_ablation.sh`
  - runs the fixed-budget Sinkhorn tail-refinement ablation on TPU,
  - keeps per-`K` logs and checkpoints isolated for comparison.
- `scripts/tpu_auto_run.sh`
  - runs the promoted Pfam Sinkhorn vs Flash comparison on TPU,
  - checkpoints to GCS for restartability.
- `scripts/tpu_multi_zone_race.sh`
  - races queued TPU requests across multiple zones,
  - keeps the first zone that becomes `ACTIVE`,
  - deletes losing queues before handing off to the chosen launcher.
- `scripts/tpu_screen_ensure.sh`
  - one-shot operational helper for `command-center`,
  - normalizes `gcloud`, cleans up the known broken stray race queues,
  - resumes the canonical screen queue if it already exists or launches a fresh race if it does not.
- `scripts/validate_tail_refinement.py`
  - numerical validation harness for the custom-VJP tail-refinement path,
  - checks exact-surrogate gradients at length 1024 by default,
  - checks module agreement across 512, 1024, 2048, 4096, 8192, and 16384 token contexts,
  - reports support-reconstruction behavior separately from the core regression gate.
- `scripts/tpu_jax_probe.py`
  - preflight TPU runtime probe for JAX/libtpu availability and distributed init state.

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

## Current Benchmark Workflows

### 1) Long-context latency benchmark

The repository currently reports TPU v6e-1 latency trends (Table 1 style) showing stable scaling of the Sinkhorn kernel up to 16K context lengths:

| Context (L) | Method | Fwd (ms) | Bwd (ms) | Status |
| :--- | :--- | :--- | :--- | :--- |
| 1024 | Flash Attention | 0.14 | 1.54 | Success |
| 1024 | Sinkhorn OT (Ours) | 363.24 | 4270.36 | Success |
| 4096 | Sinkhorn OT (Ours) | 525.13 | 5033.42 | Success |
| 8192 | Sinkhorn OT (Ours) | 711.66 | 5662.79 | Success |
| 16384 | Sinkhorn OT (Ours) | 1264.84 | 7401.55 | Success |

### 2) Pfam screening and production benchmark

The current TPU workflow is:
- run a short Sinkhorn-only screen on Pfam shards to select the best config by sparse CE,
- promote the selected config into the main Pfam Sinkhorn vs Flash comparison,
- checkpoint to GCS and preserve logs/checkpoints before queue teardown.

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

### Pfam screen

```bash
./scripts/tpu_sinkhorn_screen.sh
```

### Offline Pfam shard preparation

```bash
./scripts/prepare_data.sh
```

### Multi-zone Pfam screen race

Prefer same-region zones first to avoid unnecessary cross-region data movement:

```bash
RACE_ZONES="us-east5-a us-east5-b" \
./scripts/tpu_multi_zone_race.sh scripts/tpu_sinkhorn_screen.sh
```

Detached version:

```bash
RACE_ZONES="us-east5-a us-east5-b" \
./scripts/start_tpu_job_detached.sh scripts/tpu_multi_zone_race.sh scripts/tpu_sinkhorn_screen.sh
```

### One-shot screen recovery / launch

On `command-center`, if you just want the machine to do the right thing with the current screen state:

```bash
GCLOUD_ACCOUNT=forde.dylan@gmail.com \
RACE_ZONES="us-east5-a us-east5-b" \
./scripts/tpu_screen_ensure.sh
```

This helper will:
- keep an existing canonical `sinkhorn-screen-queue` in `us-east5-a`,
- delete the known broken stray queues from the earlier buggy race wrapper if they exist,
- restart the detached single-zone screen launcher if the canonical queue already exists,
- otherwise launch a fresh detached same-region multi-zone race.

Check queue state while it is waiting for capacity:

```bash
./scripts/tpu_queue_status.sh sinkhorn-screen-queue
```

### Pfam production run

```bash
./scripts/tpu_auto_run.sh
```

### Sinkhorn K ablation

```bash
./scripts/tpu_k_ablation.sh
```

### Tail-refinement validation

```bash
python scripts/validate_tail_refinement.py
```

Default behavior:
- exact-autodiff regression at length 1024,
- module agreement at lengths 512, 1024, 2048, 4096, 8192, and 16384,
- Stiefel projection and transport-scout reporting.

---

## Current development focus (updated roadmap)

1. **Stabilize and standardize pairwise benchmark reporting**
   - keep the Pfam screen + promoted production run as the core regression gate for correctness + training stability,
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
