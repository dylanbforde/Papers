## 2024-04-04 - Pallas Unrolled Loop Factorization
**Learning:** JAX XLA doesn't perfectly hoist common subexpressions inside unrolled `fori_loop` blocks with complex dynamic slicing. Matrix-scalar additions like `s_block + u_block[:,:,None]` and algebraic grouping around broadcasted scaling factors (like P22) must be explicitly factored by the user.
**Action:** When unrolling Pallas loops across the memory band, always manually pre-compute combinations of `s`, `u`, and `v` blocks and factor out the exponential weights `P` to minimize redundant element-wise additions and multiplications.
