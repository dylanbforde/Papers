## 2024-05-24 - [Outdated Architecture] **Learning:** The paper had an outdated architecture description, referring to ResNet models instead of the Foundation models used in benchmarks. **Action:** Next time, check if the benchmark configurations match the architectures described in the Experiments section.

## 2026-04-03 - [Variable Mappings]
**Learning:** The scale variable in `src/ot_kernel.py` corresponds to /\sqrt{d}$ in the paper, without an explicit $lpha$ hyperparameter.
**Action:** Ensure math formulas for the score matrix only use the scale /\sqrt{d}$ unless an explicit multiplier is introduced in the codebase.

## 2024-05-24 - [Tensor Packing Strategy Drift]
**Learning:** The initial paper draft assumed all potential histories and gradients were packed into a single 8-channel tensor. However, the codebase (`src/ot_kernel.py`) implements this using two distinct tensors: `M_data` for query-side variables and `N_data` for key-side variables, each with 8 channels.
**Action:** When updating architectural descriptions of kernels, always verify the exact tensor argument signatures in the `pallas_call` function rather than relying on high-level conceptual overviews.
