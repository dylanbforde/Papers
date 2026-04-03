## 2024-05-24 - [Outdated Architecture] **Learning:** The paper had an outdated architecture description, referring to ResNet models instead of the Foundation models used in benchmarks. **Action:** Next time, check if the benchmark configurations match the architectures described in the Experiments section.
## 2026-04-03 - Sync Tensor Packing Architecture
**Learning:** The codebase explicitly unpacks and packs potential histories into *two* tensors (`M_data` and `N_data`) each with C=8 channels to reduce kernel signature sizes, rather than a single unified tensor as previously described in the paper.
**Action:** Ensure hardware architectural descriptions strictly match the exact variable packing and arguments found in the source code.
