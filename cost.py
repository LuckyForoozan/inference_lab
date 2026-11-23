import json

GPU_PRICE_PER_HOUR = 0.50  # assumed price of a cloud T4 in USD; change it to a real price
REQUESTS = 40              # number of requests in the batching test

def per_million_tokens(tokens_per_s):
    return GPU_PRICE_PER_HOUR / (tokens_per_s * 3600) * 1_000_000

print("Quantization (cost per 1M generated tokens)")
for r in json.load(open("quant_results.json")):
    print(f"{r['version']:<10} latency {r['latency_s']}s | {r['tokens_per_s']} tok/s | ${per_million_tokens(r['tokens_per_s']):.2f}")

print("\nBatching (cost per request)")
for r in json.load(open("batching_results.json")):
    cost = GPU_PRICE_PER_HOUR * r["total_time_s"] / 3600 / REQUESTS
    print(f"concurrency {r['concurrency']:<3} avg latency {r['avg_latency_s']}s | {r['tokens_per_s']} tok/s | ${cost:.6f}/request")