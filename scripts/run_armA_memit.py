"""Arm A (MEMIT half): two cumulative batches of 100 on Qwen2.5-1.5B, then score recs150.

Fresh process on purpose: the ROME editor must not stay resident (two fp16
models plus MEMIT batch caches do not fit 12 GB together). Merges into
results/armA.json (keeps the ROME section). Run ONLY when the GPU is free.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party_easyedit"))
from editbench.datasets_common import match  # noqa: E402

MEMIT_HPARAMS = "hparams_15b/MEMIT-qwen2.5-1.5b.yaml"


def slice_150(recs):
    by_fam = {}
    for r in recs:
        fam = "rippleedits" if r["source"].startswith("rippleedits") else r["source"]
        by_fam.setdefault(fam, []).append(r)
    out = []
    for fam in ("counterfact", "mquake-cf", "rippleedits"):
        out.extend(by_fam[fam][:50])
    return out


def score_one(editor, r):
    hold = gen = loc = 0
    n_hold = n_gen = n_loc = 0
    if r["new_object"]:
        n_hold += 1
        hold += match(_gen(editor, f"{r['subject']} {r['relation']}"), [r["new_object"]])
    for p in r["paraphrases"][:2]:
        n_gen += 1
        gen += match(_gen(editor, p), [r["new_object"]])
    for q in r["ripple_questions"][:3] + r["locality_questions"][:3]:
        n_loc += 1
        loc += match(_gen(editor, q["question"]), q["answers"] + q["aliases"])
    return (hold, gen, loc), (n_hold, n_gen, n_loc)


def _gen(editor, prompt):
    import torch
    from editbench.evaluate import GEN_KWARGS

    tok = editor.tok
    ids = tok(prompt, return_tensors="pt").to(editor.model.device)
    n_in = ids["input_ids"].shape[1]
    with torch.inference_mode():
        gen = editor.model.generate(**ids, **GEN_KWARGS,
                                    pad_token_id=tok.pad_token_id,
                                    eos_token_id=tok.eos_token_id)
    return tok.decode(gen[0][n_in:], skip_special_tokens=True)


def main():
    from easyeditor import BaseEditor, MEMITHyperParams

    recs = [json.loads(l) for l in open(ROOT / "results" / "benchmark_main.jsonl", encoding="utf-8")]
    recs150 = slice_150(recs)
    memit = BaseEditor.from_hparams(MEMITHyperParams.from_hparams(MEMIT_HPARAMS))
    import torch as _tt
    _lm, _emb = memit.model.lm_head.weight, memit.model.model.embed_tokens.weight
    if _lm is _emb:  # Qwen ties them; MEMIT needs a separate lm_head parameter
        memit.model.lm_head.weight = _tt.nn.Parameter(_lm.detach().clone())
        print("untied lm_head for MEMIT", flush=True)
    batch_recs = [r for r in recs if r["new_object"]][:200]
    t1 = time.time()
    for b in (batch_recs[:100], batch_recs[100:]):
        reqs = [{"prompt": f"{r['subject']} {r['relation']}",
                 "target_new": r["new_object"] or "",
                 "ground_truth": r["old_object"] or "",
                 "subject": r["subject"], "portability": {}, "locality": {}}
                for r in b]
        memit.apply_algo(memit.model, memit.tok, reqs, memit.hparams,
                         copy=False, return_orig_weights=False)
        print(f"MEMIT batch done ({len(reqs)} edits)", flush=True)
    out = json.loads((ROOT / "results" / "armA.json").read_text())
    out["memit_edit_seconds"] = round(time.time() - t1, 1)
    agg = [0, 0, 0]
    tot = [0, 0, 0]
    batch_ids = {r["edit_id"] for r in batch_recs}
    overlap = 0
    for r in recs150:
        overlap += r["edit_id"] in batch_ids
        (h, g, lo), (nh, ng, nl) = score_one(memit, r)
        agg[0] += h; agg[1] += g; agg[2] += lo
        tot[0] += nh; tot[1] += ng; tot[2] += nl
    out["memit"] = {"hold": agg[0] / max(tot[0], 1), "gen": agg[1] / max(tot[1], 1),
                    "loc": agg[2] / max(tot[2], 1), "n_hold": tot[0],
                    "n_gen": tot[1], "n_loc": tot[2],
                    "eval_overlap_with_batches": overlap}
    (ROOT / "results" / "armA.json").write_text(json.dumps(out, indent=1))
    print(f"MEMIT done: {out['memit']}")


if __name__ == "__main__":
    main()
