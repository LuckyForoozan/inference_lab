from openai import OpenAI
import time
from config import BASE_URL, MODEL


client = OpenAI(base_url=BASE_URL, api_key="none")

start = time.time()
first = None
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Explain KV cache in one sentence."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        if first is None:
            first = time.time() - start
        print(chunk.choices[0].delta.content, end="", flush=True)

print(f"\nTime to first token: {first:.2f}s, total: {time.time()-start:.2f}s")