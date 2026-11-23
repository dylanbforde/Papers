# Papers
## A repository for the collection of papers

- 2 -> [Support Topology and Gradient Mixing in Sinkhorn Layers](https://arxiv.org/abs/2609.07954)
- 1 -> [Block-Wise Differentiable Sinkhorn Attention: Tail-Refinement Gradients with a Gap-Aware Dustbin Bridge](https://arxiv.org/abs/2605.08123)

## Paper 2 Abstract

Sinkhorn attention layers normalize weights over a fixed sparse mask by alternating row and column scaling. When training these layers, sparse masks are usually judged by their edge count, graph diameter, or forward reachability. We show that the sparsity pattern and marginal budgets also govern how gradients flow backward through the scaling steps.

Differentiating one row-column cycle on column log-potentials yields an explicit row-stochastic Markov kernel built from the two half-step transport plans, which reduces to a reversible two-step random walk at a balanced fixed point. Because potentials are defined only up to additive constants on each connected component, reverse-mode gradients propagate as zero-sum vectors whose total-variation norm contracts at a rate given by the kernel's Dobrushin coefficient. This coefficient bounds both unrolled backpropagation and implicit-differentiation truncation error.

Crucially, graph connectivity alone does not prevent gradient trapping: we prove a face-lattice dichotomy showing that a mask and marginals guarantee score-uniform one-step contraction if and only if every nonempty face of the transportation polytope retains pairwise two-hop column overlap; otherwise, large learned scores can push the contraction rate arbitrarily close to one. We extend this criterion to multi-layer schedules, prove exact one-sweep mixing for coordinate-factorized and butterfly masks, and derive closed-form bounds for expanders, local bands, and global hub tokens. Deterministic numerical checks verify our finite-difference adjoints, canonical sparse masks, adversarial score limits, and spectral formulas.

## Paper 1 Abstract

We study long-context balanced entropic optimal transport (OT) attention on TPU hardware through a stopped-base, fixed-depth tail-refinement surrogate. After a stopped $T$-step Sinkhorn solve, we unroll a short refinement tail and differentiate that surrogate exactly. For the reported $R=2$ TPU path, the backward pass contains four staircase plan factors.

We prove an exact one-reference-tile schedule: the $R=2$ score cotangent is a single reference plan tile times an explicit modifier field built from vector cotangents and dual differences. This yields block-wise cost $O((T+R)LW)$, $O(Ld)$ input storage, and $O(L)$ additional HBM usage for fixed head dimension $d$ and band width $W$ on the balanced fixed-support path.

We also formalize the current `dustbin_block` path as the same unit-target surrogate on an augmented support, so the adjoint schedule lifts to the single-active-dustbin path used in our TPU runs; this bridge is algebraic and does not claim a general KL-unbalanced or arbitrary-capacity gap model. We provide a local surrogate-bias bound, an a posteriori bias certificate, and a projective contraction certificate for strictly positive active blocks.

On synthetic masked problems, the optimized kernel matches exact autodiff of the same centered surrogate to within $10^{-5}$–$10^{-10}$. On TPU v6e-8, a four-configuration Pfam screen completes end-to-end, and a promoted balanced $R=2$ run sustains roughly $8.5$ examples per second through a three-hour budget, reaching step $1437$.

Held-out Pfam test shards improve reconstruction from $5.57$ to $2.05$ and sparse CE from $5.53$ to $5.30$ relative to step $0$, with CE logged diagnostically rather than optimized directly; target-barycenter alignment metrics do not materially improve, and a deterministic diagonal reference remains stronger on those metrics.
