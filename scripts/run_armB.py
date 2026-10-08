"""Run arm B on the main table (pre-registered): B-oracle + B-real (top-1).

Writes results/armB.json. One process, GPU-resident model + retriever.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from editbench.datasets_common import match  # noqa: E402
from editbench.evaluate import generate, load_model, peak_vram_gb  # noqa: E402
from editbench.methods_retrieval import build_index, fact_text, retrieve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
import torch  # noqa: E402


def ask(model, tok, context, question):
    prompt = f"{context}\nQ: {question}\nA:" if context else f"Q: {question}\nA:"
    texts, _ = generate(model, tok, [prompt])
    return texts[0]


def score_rows(model, tok, rows, ctx_fn):
    hold = gen = loc = 0
    n_hold = n_gen = n_loc = 0
    t0 = time.time()
    for r in rows:
        if r["new_object"]:
            hq = _hold_q(r)
            n_hold += 1
            hold += match(ask(model, tok, ctx_fn(r, hq), hq), [r["new_object"]])
        for p in r["paraphrases"][:2]:
            n_gen += 1
            gen += match(ask(model, tok, ctx_fn(r, p), p), [r["new_object"]])
        for q in r["ripple_questions"][:3] + r["locality_questions"][:3]:
            n_loc += 1
            loc += match(ask(model, tok, ctx_fn(r, q["question"]), q["question"]),
                         q["answers"] + q["aliases"])
    return {"hold": hold / max(n_hold, 1), "gen": gen / max(n_gen, 1),
            "loc": loc / max(n_loc, 1), "n_hold": n_hold, "n_gen": n_gen,
            "n_loc": n_loc, "seconds": round(time.time() - t0, 1)}


def _hold_q(r):
    if r["ripple_questions"]:
        return r["ripple_questions"][0]["question"]
    return f"{r['subject']} {r['relation']}"


def main():
    recs = [json.loads(l) for l in open(ROOT / "results" / "benchmark_main.jsonl", encoding="utf-8")]
    model, tok = load_model()
    torch.cuda.reset_peak_memory_stats()
    enc, index = build_index(recs)
    # retrieval hit rate: does top-1 return the record's own fact?
    hits = 0
    probes = []
    for r in recs:
        q = (r["ripple_questions"][0]["question"] if r["ripple_questions"]
             else f"{r['subject']} {r['relation']}")
        probes.append(q)
    for r, q in zip(recs, probes):
        got = retrieve(enc, index, recs, q)[0]
        hits += got["edit_id"] == r["edit_id"]
    out = {"retrieval_hit_rate": round(hits / len(recs), 4), "n": len(recs)}
    print(f"retrieval hit rate: {out['retrieval_hit_rate']}", flush=True)
    out["oracle"] = score_rows(model, tok, recs, lambda r, q: f"Fact: {fact_text(r)}.")
    print(f"oracle done: {out['oracle']}", flush=True)
    out["real"] = score_rows(
        model, tok, recs,
        lambda r, q: f"Fact: {fact_text(retrieve(enc, index, recs, q)[0])}.")
    out["peak_vram_gb"] = round(peak_vram_gb(), 2)
    (ROOT / "results" / "armB.json").write_text(json.dumps(out, indent=1))
    print(f"armB done: {out['real']} peak {out['peak_vram_gb']} GB")


if __name__ == "__main__":
    main()
