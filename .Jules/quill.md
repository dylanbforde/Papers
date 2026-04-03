## 2024-05-24 - [Outdated Architecture] **Learning:** The paper had an outdated architecture description, referring to ResNet models instead of the Foundation models used in benchmarks. **Action:** Next time, check if the benchmark configurations match the architectures described in the Experiments section.

## 2024-06-15 - [Hardware Variable Packing]
**Learning:** TPU/Pallas VMEM constraints often require theoretical monolithic variables (like "8-channels packed into a single tensor") to be split into separate inputs per sequence dimension (e.g., `M_data` for query-side and `N_data` for key-side).
**Action:** When verifying Pallas architecture claims in the paper, look for separated `M` (query) and `N` (key) dimension tensors rather than assuming a single packed tensor.
