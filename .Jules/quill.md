## 2024-05-24 - [Outdated Architecture] **Learning:** The paper had an outdated architecture description, referring to ResNet models instead of the Foundation models used in benchmarks. **Action:** Next time, check if the benchmark configurations match the architectures described in the Experiments section.

## 2026-04-03 - [Variable Mappings]
**Learning:** The scale variable in `src/ot_kernel.py` corresponds to /\sqrt{d}$ in the paper, without an explicit $lpha$ hyperparameter.
**Action:** Ensure math formulas for the score matrix only use the scale /\sqrt{d}$ unless an explicit multiplier is introduced in the codebase.

## 2024-05-24 - [Dual-Tensor Packing Strategy]
**Learning:** The paper claimed a "single tensor" with $C=8$ channels for the packing strategy, but the codebase actually utilizes two distinct tensors (`M_data` for query-side and `N_data` for key-side), each with 8 channels, to pack potential histories and gradient intermediates without overloading a single tensor.
**Action:** Always verify memory mapping architecture claims in the text against the actual JAX Pallas kernel inputs, specifically looking for split query/key state representations.
## 2024-04-16 - [Reference indices corresponding to P22 in hardware kernel]
**Learning:** The hardware kernel in `src/ot_kernel.py` uses `P22` (corresponding to reference indices $(2,2)$) as the base reference plan for its element-wise scaling relation factorization. The paper incorrectly stated $(1,1)$.
**Action:** When referring to the base reference plan in hardware-optimized formulations, ensure the variables correctly map to $(2,2)$ and `P22` in the underlying Pallas kernel code.
