"""Arm A: EasyEdit ROME (150 singles) + MEMIT (2x100 batches) on Qwen2.5-1.5B.

Writes results/armA.json. Run ONLY when the GPU is free (after arm B/C).
Per pre-registration: ROME one independent edit at a time (apply + score +
restore via apply_algo directly — the public edit() restores weights, which
would score the base model); MEMIT two batches of 100. Same alias-match
scoring as every arm.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party_easyedit"))
from editbench.datasets_common import match  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ROME_HPARAMS = "hparams_15b/ROME-qwen2.5-1.5b.yaml"
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
    """Hold/gen/loc probe counts for one record against the CURRENT weights."""
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
    from easyeditor import BaseEditor, MEMITHyperParams, ROMEHyperParams
    from easyeditor.editors.utils import restore_after_edit

    recs = [json.loads(l) for l in open(ROOT / "results" / "benchmark_main.jsonl", encoding="utf-8")]
    recs150 = slice_150(recs)
    out = {"slice_n": len(recs150)}
    editor = BaseEditor.from_hparams(ROMEHyperParams.from_hparams(ROME_HPARAMS))
    import torch as _torch
    partial = ROOT / "results" / "armA_rome_partial.jsonl"
    done_ids = set()
    agg = [0, 0, 0]
    tot = [0, 0, 0]
    if partial.exists():
        for line in open(partial, encoding="utf-8"):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            done_ids.add(e["edit_id"])
            agg[0] += e["s"][0]; agg[1] += e["s"][1]; agg[2] += e["s"][2]
            tot[0] += e["n"][0]; tot[1] += e["n"][1]; tot[2] += e["n"][2]
    print(f"resuming ROME: {len(done_ids)} already scored", flush=True)
    t0 = time.time()
    part_out = open(partial, "a", encoding="utf-8")
    for i, r in enumerate(recs150):
        if r["edit_id"] in done_ids:
            continue
        req = {"prompt": f"{r['subject']} {r['relation']}",
               "target_new": r["new_object"] or "",
               "ground_truth": r["old_object"] or "",
               "subject": r["subject"], "portability": {}, "locality": {}}
        edited_model, weights_copy = editor.apply_algo(
            editor.model, editor.tok, [req], editor.hparams,
            copy=False, return_orig_weights=True)
        (h, g, lo), (nh, ng, nl) = score_one(editor, r)
        agg[0] += h; agg[1] += g; agg[2] += lo
        tot[0] += nh; tot[1] += ng; tot[2] += nl
        restore_after_edit(editor, edited_model, weights_copy)
        del edited_model, weights_copy
        _torch.cuda.empty_cache()
        if r["edit_id"] not in done_ids:
            part_out.write(json.dumps({"edit_id": r["edit_id"], "s": [h, g, lo],
                                       "n": [nh, ng, nl]}) + "\n")
            part_out.flush()
            done_ids.add(r["edit_id"])
        if (i + 1) % 25 == 0:
            print(f"ROME {i + 1}/{len(recs150)}", flush=True)
    out["rome_edit_seconds"] = round(time.time() - t0, 1)
    out["rome"] = {"hold": agg[0] / max(tot[0], 1), "gen": agg[1] / max(tot[1], 1),
                   "loc": agg[2] / max(tot[2], 1), "n_hold": tot[0],
                   "n_gen": tot[1], "n_loc": tot[2]}
    print(f"ROME done: {out['rome']} in {out['rome_edit_seconds']}s", flush=True)
    # MEMIT: two cumulative batches of 100 from the main table, then score recs150.
    from easyeditor import BaseEditor as _BE  # noqa: F401  (same class, new hparams)
    memit = _BE.from_hparams(MEMITHyperParams.from_hparams(MEMIT_HPARAMS))
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
    out["memit_edit_seconds"] = round(time.time() - t1, 1)
    agg = [0, 0, 0]
    tot = [0, 0, 0]
    overlap = 0
    batch_ids = {r["edit_id"] for r in batch_recs}
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
    print(f"ALL DONE. ROME: {out['rome']} MEMIT: {out['memit']}")


if __name__ == "__main__":
    main()
