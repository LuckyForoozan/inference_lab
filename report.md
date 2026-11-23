# Inference Lab Report

Model: Qwen2.5-1.5B-Instruct, served with vLLM on a Colab T4 GPU.

- **Quantization:** 4-bit AWQ was fastest (112.2 tokens/s, accuracy 95%).
- **Batching:** throughput rose from 128.4 to 1267.8 tokens/s at concurrency 32, while average latency went from 0.24s to 0.6s.
- **Prefix caching:** time to first token after the first request dropped from 1.135s to 0.06s.
- **Streaming:** first text appeared after 0.03s instead of 1.464s, with the same total time.
- **Cost assumption:** GPU price of $0.5/hour, shown in EUR at 1 USD = 0.85 EUR.