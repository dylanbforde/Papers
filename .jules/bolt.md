## 2026-04-07 - Avoid Redundant safe_exp in Pallas Kernels
**Learning:** Computing multiple transport matrices or attention masks with shared structure can be extremely costly on TPU VMEM if `safe_exp` (or standard `exp`) is invoked for each full block independently. It wastes FMA/MUFU cycles.
**Action:** Always compute the base full-block matrix once (e.g. `P22 = safe_exp(S + u + v)`), then derive related matrices (like `P21`, `P11`) using O(N) broadcasted multiplications of the exponentiated differences (e.g. `P21 = P22 * safe_exp(v1 - v2)`).
