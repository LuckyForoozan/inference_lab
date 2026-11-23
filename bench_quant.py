import subprocess, time, json, requests
from evaluation import run_eval

QUANTS = ["F16", "Q8_0", "Q4_K_M"]
REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"

def start_server(quant):
    proc = subprocess.Popen(
        ["llama-server", "-hf", f"{REPO}:{quant}", "--port", "8080"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(600):  # wait up to 10 min (first run downloads the model)
        try:
            if requests.get("http://localhost:8080/health", timeout=2).status_code == 200:
                return proc
        except requests.RequestException:
            pass
        time.sleep(1)
    proc.terminate()
    raise RuntimeError(f"Server for {quant} did not start")

summary = []
for q in QUANTS:
    print(f"Running {q}...")
    proc = start_server(q)
    try:
        run_eval()  # warm-up
        res = run_eval()
    finally:
        proc.terminate()
        proc.wait()
        time.sleep(3)

    acc = sum(r["correct"] for r in res) / len(res)
    lat = sum(r["latency"] for r in res) / len(res)
    tps = sum(r["tokens"] for r in res) / sum(r["latency"] for r in res)
    summary.append({"quant": q, "accuracy": acc, "latency_s": round(lat, 2), "tokens_per_s": round(tps, 1)})
    print(f"{q}: accuracy {acc:.0%} | latency {lat:.2f}s | {tps:.1f} tokens/s")

json.dump(summary, open("quant_results.json", "w"), indent=2)