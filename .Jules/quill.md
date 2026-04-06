## 2024-05-24 - [Outdated Architecture] **Learning:** The paper had an outdated architecture description, referring to ResNet models instead of the Foundation models used in benchmarks. **Action:** Next time, check if the benchmark configurations match the architectures described in the Experiments section.

## 2026-04-03 - [Variable Mappings]
**Learning:** The scale variable in `src/ot_kernel.py` corresponds to /\sqrt{d}$ in the paper, without an explicit $lpha$ hyperparameter.
**Action:** Ensure math formulas for the score matrix only use the scale /\sqrt{d}$ unless an explicit multiplier is introduced in the codebase.

## 2026-04-06 - Dual-Tensor Packing Sync & Overfull \hbox Fix
**Learning:** The 8-channel packing strategy utilizes two distinct tensors (M_data, N_data) for query and key sides, rather than a single tensor. Additionally, long model names in 	exttt{} blocks cause Overfull \hbox warnings; \url{} combined with xurl should be used instead.
**Action:** Ensure architectural descriptions of tensor packing match the exact memory layout in src/ot_kernel.py. Use \url{} combined with \usepackage{xurl} for long HuggingFace model strings.
