"""E047: does self-critique help or hurt?

Two passes per trial on the same problem:
  pass 1  answer normally
  pass 2  the model is shown its own answer and asked to review and revise

The headline metric is NOT net accuracy (at a 97.7% baseline there is almost no room
to gain) but the flip counts: repairs (wrong -> right) versus damages (right -> wrong).

Truncation discipline from E046: a trial is valid only if BOTH passes produce a parseable
ANSWER. Invalid trials are excluded from both arms, never charged to one.
"""
import json
import re
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
DATASET = sys.argv[2] if len(sys.argv) > 2 else "datasets/e046-hard.jsonl"

FMT = ("\n\nWork through it step by step, then end your reply with a final line exactly of "
       "the form:\nANSWER: <number>")
CRITIQUE = ("Review your solution above critically. Check the reasoning and the arithmetic for "
            "errors. If you find a mistake, correct it. Then give your final answer, ending with "
            "a line exactly of the form:\nANSWER: <number>")
ANS_RE = re.compile(r"ANSWER:\s*\$?(-?[\d,]+)", re.IGNORECASE)


def extract(text, truncated):
    if truncated:
        return None
    hits = ANS_RE.findall(text or "")
    return hits[-1].replace(",", "").lstrip("0") or "0" if hits else None


def chat(messages, seed):
    body = json.dumps({
        "messages": messages, "max_tokens": 6000,
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": seed,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    return ch["message"]["content"], ch.get("finish_reason") == "length"


def trial(prob, seed, out, idx):
    try:
        m1 = [{"role": "user", "content": prob["q"] + FMT}]
        t1, tr1 = chat(m1, seed)
        a1 = extract(t1, tr1)
        m2 = m1 + [{"role": "assistant", "content": t1}, {"role": "user", "content": CRITIQUE}]
        t2, tr2 = chat(m2, seed + 1)
        a2 = extract(t2, tr2)
        out[idx] = (a1, a2)
    except Exception as e:
        out[idx] = (None, None)
        print(f"    trial error: {e}", flush=True)


def main():
    probs = [json.loads(l) for l in open(DATASET, encoding="utf-8")]
    print(f"dataset={DATASET} problems={len(probs)} K={K}", flush=True)
    a1_hits = a2_hits = valid = repairs = damages = both_wrong = invalid = 0
    detail = []
    t0 = time.time()

    for p in probs:
        gold = p["answer"].lstrip("0") or "0"
        outs = [None] * K
        ths = [threading.Thread(target=trial, args=(p, 9000 + p["id"] * 10 + i, outs, i))
               for i in range(K)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()

        rep = dam = 0
        for a1, a2 in outs:
            if a1 is None or a2 is None:
                invalid += 1
                continue
            valid += 1
            c1, c2 = (a1 == gold), (a2 == gold)
            a1_hits += c1
            a2_hits += c2
            if not c1 and c2:
                repairs += 1
                rep += 1
            elif c1 and not c2:
                damages += 1
                dam += 1
            elif not c1 and not c2:
                both_wrong += 1
        detail.append({"id": p["id"], "gold": gold, "pairs": outs, "repairs": rep, "damages": dam})
        print(f"  q{p['id']:>2} gold={gold:<5} pairs={outs} repairs={rep} damages={dam}", flush=True)

    res = {
        "K": K, "problems": len(probs), "wall_sec": round(time.time() - t0, 1),
        "valid_trials": valid, "invalid_trials": invalid,
        "pass1_accuracy_pct": round(100 * a1_hits / valid, 1) if valid else None,
        "pass2_accuracy_pct": round(100 * a2_hits / valid, 1) if valid else None,
        "net_change_points": round(100 * (a2_hits - a1_hits) / valid, 1) if valid else None,
        "repairs": repairs, "damages": damages, "both_wrong": both_wrong,
    }
    print(json.dumps(res))
    with open("benchmarks/e047-detail.json", "w", encoding="utf-8") as f:
        json.dump({"summary": res, "detail": detail}, f, indent=1)


if __name__ == "__main__":
    main()
