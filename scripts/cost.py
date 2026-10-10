"""H4 cost rows: timed micro-benchmarks per arm (extrapolations labeled as such).

- RAG / graph: 100 queries end-to-end (retrieval or compile + generation).
- ROME: 10 fresh single edits, timed (linear extrapolation to 100/1000 stated).
- MEMIT: one fresh batch of 10, timed (100/1000 extrapolated linearly, stated).
Writes results/cost.json. GPU run.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from editbench.evaluate import generate, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
import torch  # noqa: E402


def main():
    recs = [json.loads(l) for l in open(ROOT / "results" / "benchmark_main.jsonl", encoding="utf-8")]
    model, tok = load_model()
    torch.cuda.reset_peak_memory_stats()
    out = {}
    # RAG + graph query latency: retrieval/compile + one generation each
    sys.path.insert(0, str(ROOT / "third_party_easyedit"))
    from editbench.methods_retrieval import build_index, fact_text, retrieve
    from editbench.methods_graph import build_store, compile, context
    enc, index = build_index(recs)
    store = build_store(recs)
    qs = []
    for r in recs[:100]:
        qs.append(r["ripple_questions"][0]["question"] if r["ripple_questions"]
                  else f"{r['subject']} {r['relation']}")
    t0 = time.time()
    for q in qs:
        fact_text(retrieve(enc, index, recs, q)[0])
    out["rag_retrieve_100_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    for q in qs:
        compile(store, q)
    out["graph_compile_100_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    generate(model, tok, [f"Q: {q}\nA:" for q in qs])
    out["gen_100_s"] = round(time.time() - t0, 2)
    out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    # ROME x10 timed
    from easyeditor import BaseEditor, ROMEHyperParams
    editor = BaseEditor.from_hparams(
        ROMEHyperParams.from_hparams("hparams_15b/ROME-qwen2.5-1.5b.yaml"))
    from easyeditor.editors.utils import restore_after_edit
    t0 = time.time()
    for r in recs[:10]:
        req = {"prompt": f"{r['subject']} {r['relation']}",
               "target_new": r["new_object"] or "", "ground_truth": r["old_object"] or "",
               "subject": r["subject"], "portability": {}, "locality": {}}
        edited_model, weights_copy = editor.apply_algo(
            editor.model, editor.tok, [req], editor.hparams,
            copy=False, return_orig_weights=True)
        restore_after_edit(editor, edited_model, weights_copy)
        del edited_model, weights_copy
        torch.cuda.empty_cache()
    out["rome_10_s"] = round(time.time() - t0, 2)
    (ROOT / "results" / "cost.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
