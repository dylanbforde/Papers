# Papers
## A repository for the collection of papers

- 3 -> [Topology Hub Optimal Transport: Sparse Capacity Allocation, Exact Tail Differentiation, and Long-Context Efficiency](topology-hub-optimal-transport/paper/topology_hub_ot.pdf)
- 2 -> [Support Topology and Gradient Mixing in Sinkhorn Layers](https://arxiv.org/abs/2609.07954)
- 1 -> [Block-Wise Differentiable Sinkhorn Attention: Tail-Refinement Gradients with a Gap-Aware Dustbin Bridge](https://arxiv.org/abs/2605.08123)

## Paper 3 Abstract

Long-context models use sparse or structured attention to avoid quadratic cost, yet retrieving a small set of relevant facts embedded among hundreds of thousands of distractors degrades as normalization mass disperses with length. Hub-and-spoke optimal transport (OT) replaces the dense $L \times L$ matrix with a linear-size graph over local windows and $K$ shared hubs balanced by Sinkhorn scaling.

Building on block-wise tail refinement and fixed-support transport topology, we introduce *Topology Hub Optimal Transport* (THOT): (1) bounded-support capacity gating via simplex `sparsemax` ($O(L\log L)$) or sorting-free token-local thresholded ReLU ($O(L)$), driving excess distractor mass to zero; (2) a three-sweep $1+2$ annealed row-closed Sinkhorn schedule that damps bipartite residuals; (3) an exact rank-$(D+2R+1)$ One-Reference-Tile adjoint in batched split-factor Triton kernels (BF16 Tensor Cores, FP32 potentials); and (4) a chunked exclusive-prefix causal variant with verified zero future leakage.

On NVIDIA L4, sparsemax THOT trained at $L=128$ reaches $97.40\%$ zero-shot recall of fixed bindings at $L=262,144$ under pure task loss ($\alpha=0$) and $98.83\%$ at $\alpha=0.05$ (matched by $O(L)$ thresholded ReLU at $\alpha=0.05$, vs. $80.60\%$ at $\alpha=0$), while softmax-gated hub OT drops to $3.52\%$. In bidirectional operator microbenchmarks, our kernel runs forward+VJP in $100.1\text{ ms}$ at $H=8, D=64, L=262\text{K}$ ($2.2\times$ faster with $4.0\times$ smaller compiler temporaries than latent softmax) and $1.34\text{ ms}$ vs. $4.37\text{ ms}$ for cuDNN FlashAttention at $L=4,096$; multi-layer LM runs use the unfused path. Zero-shot WikiText-103 multi-needle retrieval hits $100.00\%$ ($128/128$ queries at $L=65\text{K}$) vs. $0.78\%$ for short-trained RoPE FlashAttention, and compact 4-layer WikiText-4K runs show modest causal and masked LM gains.

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
