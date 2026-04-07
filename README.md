# Block-Wise Differentiable Sinkhorn Attention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/Framework-JAX-blue.svg)](https://github.com/google/jax)
[![Hardware](https://img.shields.io/badge/Hardware-TPU%20v6e-red.svg)](https://cloud.google.com/tpu/docs/v6e)

**Scaling Optimal Transport to Massive Contexts with Hardware-Aware Kernels.**

This repository contains the official implementation of **Block-Wise Differentiable Sinkhorn Attention**, a memory-efficient and numerically stable Optimal Transport (OT) kernel designed for ultra-long context alignment (16K+) on TPU hardware.

## 🚀 Key Features

- **Multi-Transport-Plan Backward Pass**: A novel approach that eliminates the divergent gradient bias of standard implicit differentiation by explicitly unrolling $k=2$ Sinkhorn steps using four distinct transport matrices ($P^{(2,2)}, P^{(2,1)}, P^{(1,1)}, P^{(1,0)}$).
- **$O(L)$ Memory Scaling**: High-performance Pallas kernels implement block-wise tiling, maintaining a linear HBM footprint and enabling alignment at scales where $O(L^2)$ methods crash.
- **TPU v6e (Trillium) Optimization**: Utilizes an **8-channel packing strategy** to minimize VMEM pressure and maximizes throughput via Cody-Waite range reduction and minimax polynomial approximations for exponentials.
- **Banded Sinkhorn**: Supports configurable query-key banding to further reduce computational complexity to $O(L \cdot W)$.

## 🛠 Installation

The project uses `uv` for lightning-fast dependency management but can also be installed via `pip`.

```bash
# Clone the repository
git clone https://github.com/dylanbforde/sinkhorn-aligner.git
cd sinkhorn-aligner

# Install dependencies using uv
uv sync

# Or using pip
pip install .
```

## 📖 Usage

The core API is provided via the `pallas_flash_sinkhorn` function in `src/ot_kernel.py`.

```python
import jax.numpy as jnp
from src.ot_kernel import pallas_flash_sinkhorn

# Setup inputs (Batch, Heads, Length, Dim)
Q = jnp.ones((1, 8, 4096, 64))
K = jnp.ones((1, 8, 4096, 64))
V = jnp.ones((1, 8, 4096, 64))

# Call the kernel with 15 iterations and 2048 banding
output = pallas_flash_sinkhorn(
    Q, K, V, 
    n_iters=15, 
    band_width=2048
)
```

## 📊 Performance

Benchmarks conducted on **Google Cloud TPU v6e-1** show stable linear scaling compared to standard FlashAttention baselines.

| Context (L) | Method | Fwd (ms) | Bwd (ms) | Status |
| :--- | :--- | :--- | :--- | :--- |
| 1024 | Flash Attention | 0.14 | 1.54 | Success |
| 1024 | Sinkhorn OT (Ours) | 363.24 | 4270.36 | Success |
| 4096 | Sinkhorn OT (Ours) | 525.13 | 5033.42 | Success |
| 8192 | Sinkhorn OT (Ours) | 711.66 | 5662.79 | Success |
| 16384 | Sinkhorn OT (Ours) | 1264.84 | 7401.55| Success |

## 🔬 Scientific Background

The technical details and proofs are provided in the accompanying NeurIPS manuscript:

- **Forward Pass**: Implements block-wise tiling conceptually similar to FlashAttention, but for the doubly-stochastic constraints of Sinkhorn iterations.
- **Backward Pass**: Proves that truncated backpropagation through $k$ iterations is more stable than implicit methods, provided intermediate transport plans are handled correctly (Proposition 1).

### Citation

```bibtex
@inproceedings{forde2024sinkhorn,
  title={Block-Wise Differentiable Sinkhorn Attention: Scaling Optimal Transport to Massive Contexts},
  author={Forde, Dylan B.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
