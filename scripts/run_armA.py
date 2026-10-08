"""Arm A: EasyEdit ROME (150 singles) + MEMIT (2x100 batches) on Qwen2.5-1.5B.

Writes results/armA.json. Run ONLY when the GPU is free (after arm B).
Per pre-registration: ROME one edit at a time on the 150-edit slice;
MEMIT two batches of 100. Same alias-match scoring as every arm.
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


def edit_request(r):
    subj = r["subject"]
    prompt = f"{subj} {r['relation']}"
    return {"prompt": prompt, "subject": subj,
            "target_new": {"str": r["new_object"]},
            "target_true": {"str": r["old_object"]},
            "portability": {}, "locality": {}}


def main():
    from easyeditor import BaseEditor
    recs = [json.loads(l) for l in open(ROOT / "results" / "benchmark_main.jsonl", encoding="utf-8")]
    recs150 = slice_150(recs)
    out = {"slice_n": len(recs150)}
    t0 = time.time()
    rome = BaseEditor.from_hparams(ROME_HPARAMS)
    for r in recs150:
        req = edit_request(r)
        rome.edit(prompts=req["prompt"], target_new=req["target_new"],
                  ground_truth=req["target_true"])
    out["rome_edit_seconds"] = round(time.time() - t0, 1)
    out["rome"] = score_rows_rome(rome, recs150)
    (ROOT / "results" / "armA.json").write_text(json.dumps(out, indent=1))
    print(f"ROME done: {out['rome']} in {out['rome_edit_seconds']}s")


def score_rows_rome(editor, rows):
    hold = gen = loc = 0
    n_hold = n_gen = n_loc = 0
    for r in rows:
        req = edit_request(r)
        if r["new_object"]:
            n_hold += 1
            hold += match(_gen(editor, req["prompt"]), [r["new_object"]])
        for p in r["paraphrases"][:2]:
            n_gen += 1
            gen += match(_gen(editor, p), [r["new_object"]])
        for q in r["ripple_questions"][:3] + r["locality_questions"][:3]:
            n_loc += 1
            loc += match(_gen(editor, q["question"]), q["answers"] + q["aliases"])
    return {"hold": hold / max(n_hold, 1), "gen": gen / max(n_gen, 1),
            "loc": loc / max(n_loc, 1), "n_hold": n_hold, "n_gen": n_gen, "n_loc": n_loc}


def _gen(editor, prompt):
    try:
        out = editor.generate(prompt)
    except TypeError:
        out = editor.model_generate(prompt)
    if isinstance(out, list):
        out = out[0] if out else ""
    return str(out or "")


if __name__ == "__main__":
    main()
