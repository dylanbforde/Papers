## 2024-05-24 - [Outdated Architecture] **Learning:** The paper had an outdated architecture description, referring to ResNet models instead of the Foundation models used in benchmarks. **Action:** Next time, check if the benchmark configurations match the architectures described in the Experiments section.
## 2025-01-28 - [Variable Mismatch]
**Learning:** The code variable `log_alpha` (representing log attention weights) was misinterpreted in the paper's Equation 1 as a mathematical scalar `\alpha` multiplying the dot product. The code accurately scales only by `1.0 / jnp.sqrt(D)`.
**Action:** Next time, ensure that variables like `alpha` in the paper have a direct mathematical equivalent in the code, rather than being confused with intermediate variable names like `log_alpha`.
