## 2024-05-24 - [Outdated Architecture] **Learning:** The paper had an outdated architecture description, referring to ResNet models instead of the Foundation models used in benchmarks. **Action:** Next time, check if the benchmark configurations match the architectures described in the Experiments section.
## 2024-06-03 - Long Context Alignment Benchmark Outdated Sync

**Learning:** The benchmark script `src/benchmarks/benchmark_long_context_alignment.py` was evaluating on foundation models for DNA and proteins, but the paper abstract and intro claimed evaluation on "a protein family alignment benchmark from Pfam." Also discovered Overfull `\hbox` issues with long HuggingFace model names and unbroken inline math.

**Action:** Updated paper abstract and intro to reflect foundation models. Fixed Overfull `\hbox` by using `\nolinkurl` for model names and allowing math environments to break by separating them with commas. When updating papers based on code, make sure to check for internal consistency and format issues.
