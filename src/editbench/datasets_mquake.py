"""MQuAKE-CF (Zhong et al., EMNLP 2023) -> common records.

One record per requested rewrite (multi-edit cases share their questions and
carry multi_edit=true). Multi-hop questions are ripple by construction; the
answer + aliases score the ripple-through.
"""

import json


def convert(path):
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    records = []
    for c in cases:
        rewrites = c.get("requested_rewrite", [])
        for j, rw in enumerate(rewrites):
            new = rw.get("target_new", {})
            old = rw.get("target_true", {})
            records.append({
                "edit_id": f"mquake-cf-{c.get('case_id', '?')}-{j}",
                "subject": rw.get("subject", ""),
                "relation": rw.get("relation_id", ""),
                "old_object": old.get("str", ""),
                "new_object": new.get("str", ""),
                "paraphrases": [rw.get("question", "")],
                "ripple_questions": [
                    {"question": q, "answers": [c.get("answer", "")],
                     "aliases": list(c.get("answer_alias", [])),
                     "criterion": "mquake-multihop"}
                    for q in c.get("questions", [])],
                "locality_questions": [],
                "source": "mquake-cf",
                "multi_edit": len(rewrites) > 1,
            })
    return records
