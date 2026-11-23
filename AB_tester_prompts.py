from evaluation import run_eval
import json

prompts = {
    "A_brief": "You are a helpful assistant. Answer briefly.",
    "B_detailed": "You are an expert. Think step by step, then give the final answer.",
}

for name, p in prompts.items():
    res = run_eval(system_prompt=p)
    acc = sum(r["correct"] for r in res) / len(res)
    lat = sum(r["latency"] for r in res) / len(res)
    tok = sum(r["tokens"] for r in res) / len(res)
    print(f"{name}: accuracy {acc:.0%} | latency {lat:.2f}s | avg tokens {tok:.0f}")
    json.dump(res, open(f"results_{name}.json", "w"), indent=2)
