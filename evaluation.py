import json, time
from openai import OpenAI
from config import BASE_URL, MODEL


client = OpenAI(base_url=BASE_URL, api_key="none")
def run_eval(system_prompt="You are a helpful assistant. Answer briefly.", data_file="evaluation_data.json"):
    data = json.load(open(data_file, encoding="utf-8"))
    results = []
    for item in data:
        start = time.time()
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item["prompt"]},
            ],
            temperature=0,
        )
        answer = r.choices[0].message.content
        results.append({
            "prompt": item["prompt"],
            "expected": item["expected"],
            "answer": answer,
            "correct": item["expected"].lower() in answer.lower(),
            "latency": round(time.time() - start, 2),
            "tokens": r.usage.completion_tokens,
        })
    return results

if __name__ == "__main__":
    res = run_eval()
    acc = sum(x["correct"] for x in res) / len(res)
    avg = sum(x["latency"] for x in res) / len(res)
    print(f"Accuracy: {acc:.0%} | Avg latency: {avg:.2f}s")
    json.dump(res, open("results.json", "w"), indent=2)