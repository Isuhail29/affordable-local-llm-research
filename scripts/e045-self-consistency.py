"""E045: self-consistency (majority vote over K samples) vs single sample.

One batched run yields both numbers: single-sample accuracy is the mean over the K
individual samples; vote accuracy is the mode of the K. Requires a server started
with -np K so the K samples run concurrently (E044 measured the cost at 2.32x for K=4).
"""
import json
import re
import statistics
import sys
import threading
import time
import urllib.request
from collections import Counter

URL = "http://127.0.0.1:8080/v1/chat/completions"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SUFFIX = "\n\nWork through it step by step, then end your reply with a final line exactly of the form:\nANSWER: <number>"
ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+)", re.IGNORECASE)


def extract(text, truncated=False):
    """Return the declared ANSWER, or None. Never guess from a truncated reply:
    the old 'last integer anywhere' fallback silently turned cut-off working into
    fabricated wrong answers (the E045 harness bug)."""
    if truncated:
        return None
    hits = ANS_RE.findall(text or "")
    if not hits:
        return None
    return hits[-1].replace(",", "").lstrip("0") or "0"


def sample(prompt, seed, out, idx):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt + SUFFIX}],
        "max_tokens": 2500, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": seed,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.loads(r.read())
        ch = d["choices"][0]
        out[idx] = (ch["message"]["content"], ch.get("finish_reason") == "length")
    except Exception as e:
        out[idx] = (f"ERROR {e}", False)


def main():
    probs = [json.loads(l) for l in open("datasets/e045-reasoning.jsonl", encoding="utf-8")]
    single_hits = 0
    single_total = 0
    vote_hits = 0
    detail = []
    t0 = time.time()

    for p in probs:
        gold = p["answer"].lstrip("0") or "0"
        outs = [None] * K
        ths = [threading.Thread(target=sample, args=(p["q"], 7000 + p["id"] * 10 + i, outs, i)) for i in range(K)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()

        answers = [extract(o[0], o[1]) if isinstance(o, tuple) else None for o in outs]
        n_trunc = sum(1 for o in outs if isinstance(o, tuple) and o[1])
        if n_trunc:
            print(f"    warning: {n_trunc}/{K} samples hit the token cap and were excluded", flush=True)
        correct_flags = [a == gold for a in answers]
        single_hits += sum(correct_flags)
        single_total += K

        valid = [a for a in answers if a is not None]
        winner = Counter(valid).most_common(1)[0][0] if valid else None
        voted = (winner == gold)
        vote_hits += 1 if voted else 0

        detail.append({"id": p["id"], "gold": gold, "answers": answers,
                       "n_correct": sum(correct_flags), "vote": winner, "vote_correct": voted})
        print(f"  q{p['id']:>2} gold={gold:<5} samples={answers} vote={winner} {'OK' if voted else 'MISS'}", flush=True)

    wall = time.time() - t0
    n = len(probs)
    res = {
        "K": K, "problems": n, "wall_sec": round(wall, 1),
        "single_sample_accuracy_pct": round(100 * single_hits / single_total, 1),
        "majority_vote_accuracy_pct": round(100 * vote_hits / n, 1),
        "gain_points": round(100 * vote_hits / n - 100 * single_hits / single_total, 1),
    }
    print(json.dumps(res))
    with open("benchmarks/e045-detail.json", "w", encoding="utf-8") as f:
        json.dump({"summary": res, "detail": detail}, f, indent=1)


if __name__ == "__main__":
    main()
