# Block-Wise Differentiable Sinkhorn Attention

Reference implementation for long-context differentiable optimal-transport
attention with a fixed-depth tail-refinement backward pass.

The repository is kept intentionally small for public use: source code,
validation scripts, TPU launch helpers, and the two current manuscript builds are
included. Generated logs, checkpoints, local Pfam shards, LaTeX build products,
and submission archives are not committed.

## Repository Layout

- `src/ot_kernel.py` - masked banded Sinkhorn attention, Pallas forward kernels,
  and custom-VJP tail-refinement backward paths.
- `src/benchmarks/` - Pfam training and long-context benchmark entrypoints.
- `src/unbalanced_ot.py` - experimental unbalanced OT reference utilities.
- `scripts/` - validation, figure-generation, evaluation, and TPU launch
  helpers.
- `data/download_pfam.py` - Pfam shard preparation utility.
- `paper_neurips/` - the two maintained paper builds:
  - `neurips2026_preprint.pdf` / `.tex` for arXiv.
  - `neurips2026_submission.pdf` / `.tex` for the OpenReview submission.

## Install

```bash
git clone https://github.com/dylanbforde/sinkhorn-aligner.git
cd sinkhorn-aligner

uv sync
```

The project can also be installed with `pip install .` in an environment that
already has a compatible JAX build.

## Quick Start

```python
import jax.numpy as jnp
from src.ot_kernel import sinkhorn_attention

q = jnp.ones((1, 8, 4096, 64))
k = jnp.ones((1, 8, 4096, 64))
v = jnp.ones((1, 8, 4096, 64))

out = sinkhorn_attention(
    q,
    k,
    v,
    n_iters=15,
    band_width=1024,
    refine_steps=2,
    epsilon=1.0,
    gradient_mode="tail_refinement",
)
```

Use `gradient_mode="exact_autodiff"` for small reference checks.

## Validation

Run the main CPU/GPU-safe numerical checks:

```bash
python scripts/validate_tail_refinement.py --exactness_only --length 1024
python scripts/validate_orbit_factorization.py
python scripts/validate_tail_bias_certificate.py
python scripts/validate_dustbin_augmentation.py
```

Optional diagnostics used for the paper:

```bash
python scripts/benchmark_four_plan_adjoint.py
python scripts/estimate_r2_backward_memory.py
python scripts/summarize_pfam_eval_metrics.py <evaluation-json> [...]
```

## Data and Checkpoints

Large datasets and checkpoints are intentionally excluded. Pfam inputs can be
prepared with:

```bash
BUCKET_NAME=<your-gcs-bucket> ./scripts/prepare_data.sh
```

Local generated data should live under ignored paths such as `data/cache/`,
`data/pfam_shards/`, `data/pfam_shards_test/`, and `checkpoints/`.

## TPU Runs

The TPU scripts are operational helpers for Cloud TPU v6e workflows. They require
caller-provided cloud configuration:

```bash
export PROJECT_ID=<your-gcp-project>
export BUCKET_NAME=<your-gcs-bucket>
export ZONE=us-east5-a
```

Examples:

```bash
./scripts/tpu_sinkhorn_screen.sh
./scripts/tpu_k_ablation.sh
./scripts/tpu_auto_run.sh
```

These scripts package the current checkout, copy it to the TPU VM, fetch Pfam
shards from GCS, run the configured job, and sync logs/checkpoints back to the
configured bucket.

## Papers

The maintained manuscript files are in `paper_neurips/`. See
`paper_neurips/README.md` for the exact source/PDF mapping. Old generated
submission bundles and duplicate arXiv upload trees have been removed to keep
the public repository readable.

## License

MIT. See `LICENSE`.
