# Topology Hub Optimal Transport

Standalone reference kernels, capacity gates, causal operator, and manuscript sources for **Topology Hub Optimal Transport: Eliminating Long-Context Dilution at Flash-Competitive Speed**.

## Repository Layout

- `src/gates.py`: Bounded-support capacity gates (`sparsemax_capacities`, sorting-free $O(L)$ `relu_threshold_capacities`, and `causal_threshold_gate`).
- `src/kernels.py`: Batched multi-head Pallas/Triton kernels, Two-Call One-Reference-Tile `custom_vjp` (`thot_transport`, `thot_attention`), and exact pure-JAX reference (`thot_reference`) for the 1+2 annealed row-closed Sinkhorn operator.
- `src/causal.py`: Strictly causal chunked block-hub OT operator (`CausalTHOTConfig`, `causal_thot_head`, `causal_thot_attention`) and exact zero-future-leakage verifier (`verify_causal_invariance`).
- `paper/`: LaTeX source (`topology_hub_ot.tex`), BibTeX (`references.bib`), vector figures (`figures/`), compiled PDF (`topology_hub_ot.pdf`), and build script (`build.sh`).

## Environment & Install

On NixOS, enter the pinned environment:

```sh
nix-shell
```

Or install in any Python >= 3.10 environment with a compatible JAX build:

```sh
pip install -e .
```

(`backend="pallas"` targets GPU Triton via `jax.experimental.pallas.triton`; use `backend="interpret"` or `backend="reference"` on CPU.)

## Quick Start

### Bidirectional THOT (`src/kernels.py`, `src/gates.py`)

```python
import jax
import jax.numpy as jnp
from src import sparsemax_capacities, thot_attention

B, H, L, K, D = 1, 8, 2048, 16, 64
key = jax.random.PRNGKey(0)
kq, kk, kv, kr, ks = jax.random.split(key, 5)

q = jax.random.normal(kq, (B, H, L + K, D), dtype=jnp.bfloat16)
k = jax.random.normal(kk, (B, H, L + K, D), dtype=jnp.bfloat16)
v = jax.random.normal(kv, (B, H, L + K, D), dtype=jnp.bfloat16)

rec_logits = jax.random.normal(kr, (B, H, L))
src_logits = jax.random.normal(ks, (B, H, L))
a, b = sparsemax_capacities(rec_logits, src_logits, hubs=K)

# On GPU use backend="pallas" (default); on CPU use backend="interpret" or "reference".
out = thot_attention(q, k, v, a, b, length=L, radius=4, hubs=K, backend="pallas")
```

### Strictly Causal THOT (`src/causal.py`, `src/gates.py`)

```python
import jax
import jax.numpy as jnp
from src import CausalTHOTConfig, causal_thot_attention, causal_threshold_gate

B, H, L, K, D = 2, 4, 1024, 16, 32
key = jax.random.PRNGKey(1)
kq, kk, kv, kr, ks, kh = jax.random.split(key, 6)

q = jax.random.normal(kq, (B, L, H, D))
k = jax.random.normal(kk, (B, L, H, D))
v = jax.random.normal(kv, (B, L, H, D))
q_hub = jax.random.normal(kh, (K, H, D))
k_hub, v_hub = q_hub, jax.random.normal(kh, (K, H, D))

rec_s = jax.random.normal(kr, (B, L, H))
src_s = jax.random.normal(ks, (B, L, H))
a, b = causal_threshold_gate(rec_s, src_s, theta_rec=-0.25, theta_src=-0.25, length=L)

cfg = CausalTHOTConfig(chunk_size=32, radius=4, hubs=K)
y = causal_thot_attention(q, k, v, a, b, q_hub, k_hub, v_hub, cfg=cfg)
```

## Verification

Verify zero future leakage and finite VJP gradients for causal THOT:

```sh
python -c "from src import verify_causal_invariance; r = verify_causal_invariance(); print(r); assert r['passed']"
```

## Manuscript

```sh
sh paper/build.sh
```

## License

MIT; see [LICENSE](LICENSE).

