## 2026-04-03 - [Benchmark Caching Foundation Models]
**Learning:** Caching foundation models in long-context benchmark loops prevents redundant disk I/O, memory allocations, and network checks, mitigating OOM errors and significant overhead.
**Action:** Use a global dictionary _MODEL_CACHE to cache the tokenizer and model by model_name when iterating through different sequence lengths in benchmark_long_context_alignment.py.
