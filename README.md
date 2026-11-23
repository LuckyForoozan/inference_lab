# Inference Lab: Production LLM Optimization & Benchmarking

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://inferencelab-7xct2q7pusy6ncjc5ifkjb.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![vLLM](https://img.shields.io/badge/served%20with-vLLM-orange.svg)](https://github.com/vllm-project/vllm)

An end-to-end laboratory and interactive dashboard for evaluating, benchmarking, and optimizing Large Language Model (LLM) serving performance in production. 

Using **Qwen2.5-1.5B-Instruct** served via **vLLM** on cloud GPUs, this project systematically benchmarks the primary throughput, latency, memory, and cost levers used in modern AI engineering.

---

##  Live Interactive Dashboard

Explore all benchmark results, interactive charts, and multi-currency cost calculators 24/7 without running any code:

👉 **[Launch Live Dashboard on Streamlit Cloud](https://inferencelab-7xct2q7pusy6ncjc5ifkjb.streamlit.app)**

---

##  Benchmark Highlights

| Optimization Technique | Key Finding | Production Impact |
| :--- | :--- | :--- |
| **Quantization (4-bit AWQ)** | **112.2 tokens/s** at **95% accuracy** | ~2x faster throughput than FP16 while preserving precision and halving VRAM. |
| **Dynamic Batching** | Throughput scaled from **128.4 to 1267.8 tokens/s** at concurrency 32 | Maximizes GPU compute saturation under high traffic loads. |
| **Prefix Caching** | TTFT dropped from **1.135s to 0.06s** | Dramatic latency reduction on recurring system prompts and multi-turn conversations. |
| **Streaming** | First token rendered in **0.03s** vs **1.46s** | Drastic improvement in perceived user experience without affecting total generation time. |
| **Speculative Decoding** | Evaluated Qwen2.5-1.5B with a 0.5B draft model | Parallel verification mechanics for low-latency single-stream inference. |
| **Tail Latency (p95/p99)** | Median total latency: **0.936s**; p95 tail: **1.331s** | Uncovers queue delays and prefill contention under concurrent loads. |

---

##  System Architecture 





##  Bring Your Own Dataset

You can easily adapt this project to evaluate custom domain prompts:
1. Replace or extend the questions inside `evaluation_data.json`:
   ```json
   [
     {"prompt": "Your custom prompt here", "expected": "expected keyword"}
   ]
   ```
2. Re-run `bench.ipynb` on GPU.
3. Commit and push the new results to GitHub — your deployed Streamlit dashboard will automatically update!

---

