"""Run arm C on the main table (pre-registered): graph-covered vs uncovered split.

Writes results/armC.json. Run ONLY when the GPU is free.
Scoring identical to arm B; context comes from the triple store instead.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from editbench.datasets_common import match  # noqa: E402
from editbench.evaluate import generate, load_model, peak_vram_gb  # noqa: E402
from editbench.methods_graph import build_store, compile, context  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
import torch  # noqa: E402


def ask(model, tok, ctx, question):
    prompt = f"{ctx}\nQ: {question}\nA:" if ctx else f"Q: {question}\nA:"
    texts, _ = generate(model, tok, [prompt])
    return texts[0]


def main():
    recs = [json.loads(l) for l in open(ROOT / "results" / "benchmark_main.jsonl", encoding="utf-8")]
    store = build_store(recs)
    model, tok = load_model()
    torch.cuda.reset_peak_memory_stats()
    agg = {"covered": {"hold": 0, "gen": 0, "loc": 0, "n_hold": 0, "n_gen": 0, "n_loc": 0},
           "uncovered": {"hold": 0, "gen": 0, "loc": 0, "n_hold": 0, "n_gen": 0, "n_loc": 0}}
    t0 = time.time()
    for i, r in enumerate(recs):
        if r["new_object"]:
            hq = r["ripple_questions"][0]["question"] if r["ripple_questions"] else f"{r['subject']} {r['relation']}"
            triples, cov = compile(store, hq)
            key = "covered" if cov else "uncovered"
            agg[key]["n_hold"] += 1
            agg[key]["hold"] += match(ask(model, tok, context(triples), hq), [r["new_object"]])
        for p in r["paraphrases"][:2]:
            triples, cov = compile(store, p)
            key = "covered" if cov else "uncovered"
            agg[key]["n_gen"] += 1
            agg[key]["gen"] += match(ask(model, tok, context(triples), p), [r["new_object"]])
        for q in r["ripple_questions"][:3] + r["locality_questions"][:3]:
            triples, cov = compile(store, q["question"])
            key = "covered" if cov else "uncovered"
            agg[key]["n_loc"] += 1
            agg[key]["loc"] += match(ask(model, tok, context(triples), q["question"]),
                                     q["answers"] + q["aliases"])
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{len(recs)}", flush=True)
    out = {"seconds": round(time.time() - t0, 1),
           "peak_vram_gb": round(peak_vram_gb(), 2)}
    for key, a in agg.items():
        out[key] = {k: (round(v / max(a[f"n_{k}"], 1), 4) if not k.startswith("n_") else v)
                      for k, v in a.items()}
    (ROOT / "results" / "armC.json").write_text(json.dumps(out, indent=1))
    print(f"armC done: covered={out['covered']} uncovered={out['uncovered']}")


if __name__ == "__main__":
    main()
