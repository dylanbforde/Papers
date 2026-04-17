## 2024-05-24 - [Outdated Architecture] **Learning:** The paper had an outdated architecture description, referring to ResNet models instead of the Foundation models used in benchmarks. **Action:** Next time, check if the benchmark configurations match the architectures described in the Experiments section.

## 2026-04-03 - [Variable Mappings]
**Learning:** The scale variable in `src/ot_kernel.py` corresponds to /\sqrt{d}$ in the paper, without an explicit $lpha$ hyperparameter.
**Action:** Ensure math formulas for the score matrix only use the scale /\sqrt{d}$ unless an explicit multiplier is introduced in the codebase.

## 2024-05-24 - [Dual-Tensor Packing Strategy]
**Learning:** The paper claimed a "single tensor" with $C=8$ channels for the packing strategy, but the codebase actually utilizes two distinct tensors (`M_data` for query-side and `N_data` for key-side), each with 8 channels, to pack potential histories and gradient intermediates without overloading a single tensor.
**Action:** Always verify memory mapping architecture claims in the text against the actual JAX Pallas kernel inputs, specifically looking for split query/key state representations.

## 2024-05-24 - [Base Reference Plan Mismatch]
**Learning:** The codebase explicitly uses $P^{(2,2)}$ as the base reference plan for the algebraic factorization in the multi-transport dS equation.
**Action:** When documenting the hardware-efficient factorization or the mathematical formulas for the tail-refinement estimator, always write the scaling relationships relative to $P^{(2,2)}$ (using potentials $u^{(2)}, v^{(2)}$), rather than $P^{(1,1)}$.
