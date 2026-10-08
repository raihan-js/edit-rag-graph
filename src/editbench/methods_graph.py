"""Arm C: explicit triple store + question-to-query compiler.

Store: (subject, relation, object) from the edit set (new objects). Query:
longest known-subject substring in the question, then walk outgoing edges
whose relation tokens overlap the question (up to 2 hops). Covered = a walk
reaches an object; uncovered questions get no triples. The model answers from
returned triples. Ripple-by-construction only where covered — reported split.
"""

import re


def tokens(s):
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))


def build_store(records):
    store = {}
    for r in records:
        key = r["subject"].lower().strip()
        store.setdefault(key, {"name": r["subject"].strip(), "edges": []})
        store[key]["edges"].append((r["relation"], r["new_object"], r["edit_id"]))
    subjects = sorted(store, key=len, reverse=True)
    return {"edges": store, "subjects": subjects}


def compile(store, question):
    """Return (triples, covered). Triples are (subject, relation, object)."""
    q = question.lower()
    subj = next((s for s in store["subjects"] if s and s in q), None)
    if subj is None:
        return [], False
    out, frontier, seen = [], [(subj, 0)], {subj}
    while frontier:
        node, depth = frontier.pop(0)
        if depth >= 2:
            continue
        for rel, obj, _ in store["edges"][node]["edges"]:
            if tokens(rel) & tokens(question) or depth == 0:
                name = store["edges"][node]["name"]
                out.append((name, rel, obj))
                key = obj.lower().strip()
                if key in store["edges"] and key not in seen:
                    seen.add(key)
                    frontier.append((key, depth + 1))
    return out, bool(out)


def context(triples):
    if not triples:
        return ""
    lines = [f"{s} | {r} | {o}" for s, r, o in triples]
    return "Known facts:\n" + "\n".join(lines) + "."
