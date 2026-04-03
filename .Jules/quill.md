## 2024-05-24 - [Outdated Architecture] **Learning:** The paper had an outdated architecture description, referring to ResNet models instead of the Foundation models used in benchmarks. **Action:** Next time, check if the benchmark configurations match the architectures described in the Experiments section.
## 2026-04-03 - [Tensor Packing Architecture Discrepancy]
**Learning:** The code actually stacks potential histories and intermediate gradient vectors into two separate tensors (`M_data` and `N_data`), not a single one as previously claimed in the paper. Both tensors are =8$ channels.
**Action:** Always verify the tensor inputs to the Pallas kernels instead of relying on high-level comments when updating memory/packing descriptions in the paper.
