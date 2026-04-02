## 2024-10-24 - [Factorized Block Exponentials]
**Learning:** `safe_exp` (or standard `jnp.exp`) on TPU inside unrolled Pallas loops is a massive FLOP bottleneck. Matrix exponentials dominate loop latency.
**Action:** When computing related transport plans (like $P^{(2,2)}, P^{(2,1)}$ etc.), evaluate the full $M \times N$ matrix exponential only once for the base plan, and apply the variations using 1D vector exponentials broadcasted over the block. This converts heavy transcendals into native FMA scalar multiplications.
