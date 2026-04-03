## 2024-05-24 - [Outdated Architecture] **Learning:** The paper had an outdated architecture description, referring to ResNet models instead of the Foundation models used in benchmarks. **Action:** Next time, check if the benchmark configurations match the architectures described in the Experiments section.

## 2025-04-03 - [Hyperparameter Drift]
**Learning:** The paper included a hardcoded `\alpha` hyperparameter in the forward pass equation that did not exist in the code implementation. Additionally, the description of tensor packing in the paper did not accurately reflect the codebase's use of two separate 8-channel tensors ($M_{\text{data}}$ and $N_{\text{data}}$).
**Action:** When verifying equations against code, ensure every single hyperparameter and variable maps exactly to the implementation. Similarly, when checking hardware optimization descriptions, confirm that the tensor layouts match the latest Pallas kernel specifications.
