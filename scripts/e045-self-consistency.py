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
        "max_tokens": 6000, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": seed,
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
    dataset = sys.argv[2] if len(sys.argv) > 2 else "datasets/e045-reasoning.jsonl"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    probs = [json.loads(l) for l in open(dataset, encoding="utf-8")]
    if limit:
        probs = probs[:limit]
    print(f"dataset={dataset} problems={len(probs)} K={K}", flush=True)
    single_hits = 0
    single_total = 0
    vote_hits = 0
    vote_total = 0
    trunc_samples = 0
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
        trunc_samples += n_trunc
        if n_trunc:
            print(f"    warning: {n_trunc}/{K} samples truncated (excluded from BOTH metrics)", flush=True)
        # Apples-to-apples: a truncated sample is an invalid trial, excluded from BOTH
        # the single-sample metric and the vote. Counting it wrong for one and dropping
        # it from the other manufactured a +25 point phantom gain in the E046 pre-check.
        valid = [a for a in answers if a is not None]
        single_hits += sum(1 for a in valid if a == gold)
        single_total += len(valid)

        winner = Counter(valid).most_common(1)[0][0] if valid else None
        voted = (winner == gold)
        if valid:
            vote_hits += 1 if voted else 0
            vote_total += 1

        n_valid_correct = sum(1 for a in valid if a == gold)
        detail.append({"id": p["id"], "gold": gold, "answers": answers, "n_valid": len(valid),
                       "n_correct": n_valid_correct, "vote": winner, "vote_correct": voted})
        print(f"  q{p['id']:>2} gold={gold:<5} valid={len(valid)}/{K} correct={n_valid_correct} "
              f"samples={answers} vote={winner} {'OK' if voted else 'MISS'}", flush=True)

    wall = time.time() - t0
    n = len(probs)
    single_pct = round(100 * single_hits / single_total, 1) if single_total else None
    vote_pct = round(100 * vote_hits / vote_total, 1) if vote_total else None
    res = {
        "K": K, "problems": n, "wall_sec": round(wall, 1),
        "valid_samples": single_total, "truncated_samples": trunc_samples,
        "problems_scored": vote_total,
        "single_sample_accuracy_pct": single_pct,
        "majority_vote_accuracy_pct": vote_pct,
        "gain_points": (round(vote_pct - single_pct, 1)
                        if single_pct is not None and vote_pct is not None else None),
    }
    print(json.dumps(res))
    with open("benchmarks/e045-detail.json", "w", encoding="utf-8") as f:
        json.dump({"summary": res, "detail": detail}, f, indent=1)


if __name__ == "__main__":
    main()
