"""E044: fire K concurrent identical-length generations and measure aggregate throughput.

Usage: python e044-batch-gate.py <K> [tag]
Every request generates exactly TOKENS tokens (ignore_eos) so runs are comparable.
"""
import json
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
TOKENS = 200
PROMPT = "Explain how a bicycle stays upright when moving. Be thorough."

results = []
lock = threading.Lock()


def one(i):
    body = json.dumps({
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": TOKENS,
        "ignore_eos": True,
        "temperature": 1.0,
        "seed": 1000 + i,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    n = d.get("timings", {}).get("predicted_n", 0)
    tps = d.get("timings", {}).get("predicted_per_second", 0)
    with lock:
        results.append({"i": i, "sec": dt, "n": n, "tps": tps})


if __name__ == "__main__":
    K = int(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else f"K{K}"
    threads = [threading.Thread(target=one, args=(i,)) for i in range(K)]
    wall0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - wall0

    total_tokens = sum(r["n"] for r in results)
    agg = total_tokens / wall if wall else 0
    per = sum(r["tps"] for r in results) / len(results) if results else 0
    out = {
        "tag": tag, "K": K, "wall_sec": round(wall, 2),
        "total_tokens": total_tokens,
        "aggregate_tps": round(agg, 2),
        "avg_per_request_tps": round(per, 2),
    }
    print(json.dumps(out))
    with open("benchmarks/e044-results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")
