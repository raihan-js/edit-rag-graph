"""RippleEdits (Cohen et al., TACL 2024; MIT) -> common records.

Six criteria split by what they test: Compositionality I/II and Logical
Generalization need the edit (ripple); Relation Specificity, Subject Aliasing
and Forgetfulness must NOT change (locality). Condition queries are recorded
but not scored in v1.
"""

import json

RIPPLE = {"Compositionality_I", "Compositionality_II", "Logical_Generalization"}
LOCALITY = {"Relation_Specifity", "Subject_Aliasing", "Forgetfulness"}


def convert(path, example_type):
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    records = []
    for i, e in enumerate(entries):
        ed = e["edit"]
        subj = _subject(ed)
        ripple, locality = [], []
        for crit in RIPPLE | LOCALITY:
            for block in e.get(crit, []) or []:
                for t in block.get("test_queries", []):
                    q = {"question": t["prompt"],
                         "answers": [a["value"] for a in t.get("answers", [])],
                         "aliases": [al for a in t.get("answers", [])
                                     for al in a.get("aliases", [])],
                         "criterion": crit}
                    (ripple if crit in RIPPLE else locality).append(q)
        records.append({
            "edit_id": f"rippleedits-{example_type}-{i:04d}",
            "subject": subj,
            "relation": ed.get("relation", ""),
            "old_object": (ed.get("original_fact") or {}).get("prompt", ""),
            "new_object": ed.get("prompt", ""),
            "paraphrases": [],
            "ripple_questions": ripple,
            "locality_questions": locality,
            "source": f"rippleedits-{example_type}",
        })
    return records


def _subject(ed):
    p = ed.get("prompt", "")
    # Prompts read "... of <SUBJECT> is <OBJECT>." — take text after the last " of ".
    if " of " in p and " is " in p:
        tail = p.rsplit(" of ", 1)[1]
        subj = tail.split(" is ", 1)[0].strip().rstrip(".")
        if subj:
            return subj
    return ed.get("subject_id", "")
