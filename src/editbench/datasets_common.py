"""Common record helpers: alias-aware exact match (the only scorer)."""


def norm(s):
    return " ".join(str(s).lower().strip().split())


def match(pred, answers):
    """True if the prediction contains the answer or a listed alias."""
    p = norm(pred)
    for a in answers:
        if isinstance(a, dict):
            cands = [a.get("value", "")] + list(a.get("aliases", []))
        else:
            cands = [a]
        for c in cands:
            c = norm(c)
            if c and c in p:
                return True
    return False
