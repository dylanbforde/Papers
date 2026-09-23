# Papers
## A repository for the collection of papers

- 2 -> [Support Topology and Gradient Mixing in Sinkhorn Layers](https://arxiv.org/abs/2609.07954)
- 1 -> [Block-Wise Differentiable Sinkhorn Attention: Tail-Refinement Gradients with a Gap-Aware Dustbin Bridge](https://arxiv.org/abs/2605.08123)

## Paper 2 Abstract

Sparse Sinkhorn layers use a fixed support graph to restrict transport between tokens. How does this graph control gradient propagation through the scaling iterations. We develop a fixed-support calculus showing that each row-column cycle induces a row-stochastic operator on column-potential perturbations modulo constants. Its transpose propagates zero-mass reverse-mode cotangents.

The finite-cycle operator uses two distinct half-step transport plans; at a balanced fixed point it reduces to a two-step walk determined by a single plan. We derive the accompanying score and marginal source terms and use Dobrushin contraction and minorization to bound homogeneous and source-driven tail cotangents.

Our main result characterizes when support and marginals guarantee one-step contraction uniformly over finite scores: every feasible face of the transportation polytope must have pairwise two-hop column overlap. Otherwise, suitable score directions make the contraction coefficient arbitrarily close to one. We extend this analysis to ordered support schedules and derive certificates for partition heat-bath layers, coordinate sweeps, forced shared mass, and register-augmented supports.

These results provide mathematical criteria for support design in differentiable transport layers, with guarantees restricted to the fixed-support quotient-gradient component.

## Paper 1 Abstract

We study long-context balanced entropic optimal transport (OT) attention on TPU hardware through a stopped-base, fixed-depth tail-refinement surrogate. After a stopped $T$-step Sinkhorn solve, we unroll a short refinement tail and differentiate that surrogate exactly. For the reported $R=2$ TPU path, the backward pass contains four staircase plan factors.

We prove an exact one-reference-tile schedule: the $R=2$ score cotangent is a single reference plan tile times an explicit modifier field built from vector cotangents and dual differences. This yields block-wise cost $O((T+R)LW)$, $O(Ld)$ input storage, and $O(L)$ additional HBM usage for fixed head dimension $d$ and band width $W$ on the balanced fixed-support path.

We also formalize the current `dustbin_block` path as the same unit-target surrogate on an augmented support, so the adjoint schedule lifts to the single-active-dustbin path used in our TPU runs; this bridge is algebraic and does not claim a general KL-unbalanced or arbitrary-capacity gap model. We provide a local surrogate-bias bound, an a posteriori bias certificate, and a projective contraction certificate for strictly positive active blocks.

On synthetic masked problems, the optimized kernel matches exact autodiff of the same centered surrogate to within $10^{-5}$–$10^{-10}$. On TPU v6e-8, a four-configuration Pfam screen completes end-to-end, and a promoted balanced $R=2$ run sustains roughly $8.5$ examples per second through a three-hour budget, reaching step $1437$.

Held-out Pfam test shards improve reconstruction from $5.57$ to $2.05$ and sparse CE from $5.53$ to $5.30$ relative to step $0$, with CE logged diagnostically rather than optimized directly; target-barycenter alignment metrics do not materially improve, and a deterministic diagonal reference remains stronger on those metrics.
