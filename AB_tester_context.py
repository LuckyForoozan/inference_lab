from openai import OpenAI
import time
from config import BASE_URL, MODEL
client = OpenAI(base_url=BASE_URL, api_key="none")


FACTS = [
    ("The secret code for room 7 is 4821.", "What is the secret code for room 7?", "4821"),
    ("Anna's favorite fruit is mango.", "What is Anna's favorite fruit?", "mango"),
    ("The meeting is on Thursday at 3 PM.", "When is the meeting?", "Thursday"),
    ("The server password hint is blue-tiger.", "What is the server password hint?", "blue-tiger"),
]

def filler(n):
    return [f"Note {i}: the weather report for day {i} was cloudy with light wind." for i in range(n)]

def build(fact, n, pos):
    s = filler(n)
    i = {"start": 0, "middle": n // 2, "end": n}[pos]
    s.insert(i, fact)
    return s

def retrieve(sentences, question, k=3):
    q = set(question.lower().split())
    return sorted(sentences, key=lambda s: -len(q & set(s.lower().split())))[:k]

def apply(strategy, sentences, question):
    if strategy == "full": return sentences
    if strategy == "truncate_last20": return sentences[-20:]
    if strategy == "retrieval_top3": return retrieve(sentences, question)

def ask(context, question):
    start = time.time()
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Answer briefly using only the context."},
            {"role": "user", "content": "Context:\n" + "\n".join(context) + f"\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    return r.choices[0].message.content, time.time() - start, r.usage.prompt_tokens

configs = [(s, p) for s in ["full", "truncate_last20", "retrieval_top3"] for p in ["start", "middle", "end"]]

for n in [20, 80, 150]:
    for strategy, pos in configs:
        ok = lat = tok = 0
        for fact, q, exp in FACTS:
            ctx = apply(strategy, build(fact, n, pos), q)
            ans, l, t = ask(ctx, q)
            ok += exp.lower() in ans.lower(); lat += l; tok += t
        m = len(FACTS)
        print(f"n={n:<4}{strategy:<16}fact@{pos:<7}acc {ok/m:.0%} | {lat/m:.2f}s | {tok/m:.0f} prompt tokens")