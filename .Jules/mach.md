## 2024-05-24 - Sinkhorn Multi-Transport Matrix Factoring
**Learning:** Computing `safe_exp` multiple times over dense `(B, M, N)` block scores in Pallas unrolled backward loops causes extreme SRAM thrashing and bottlenecked VPU throughput.
**Action:** Always factor out common terms (like `P22`) algebraically. Use cheap 1D marginal differences (`(B, M)` or `(B, N)`) to scale the common term, significantly reducing memory footprint and duplicate computation inside JAX/Pallas optimal transport kernels.
