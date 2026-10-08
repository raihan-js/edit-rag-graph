"""CounterFact (Meng et al., NeurIPS 2022; via rome.baulab.info) -> common records.

Paraphrase prompts test the same edit in new words (generalization);
neighborhood prompts must NOT change (locality). CounterFact has no
multi-hop questions, so ripple coverage comes from RippleEdits/MQuAKE.
"""

import json


def convert(path, cap=None):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if cap:
        data = data[:cap]
    records = []
    for c in data:
        rw = c.get("requested_rewrite", {})
        new = rw.get("target_new", {})
        old = rw.get("target_true", {})
        records.append({
            "edit_id": f"counterfact-{c.get('case_id', '?')}",
            "subject": rw.get("subject", ""),
            "relation": rw.get("relation_id", ""),
            "old_object": old.get("str", ""),
            "new_object": new.get("str", ""),
            "paraphrases": list(c.get("paraphrase_prompts", [])),
            "ripple_questions": [],
            "locality_questions": [
                {"question": q, "answers": [], "aliases": [],
                 "criterion": "counterfact-neighborhood"}
                for q in c.get("neighborhood_prompts", [])],
            "source": "counterfact",
        })
    return records
