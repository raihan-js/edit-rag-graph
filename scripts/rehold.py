"""Hold-probe fix (recorded, not tuning): re-derive hold probes from raw sources.

The main runs probed hold with ripple/multi-hop questions (and raw relation
IDs for CounterFact) — garbage probes even with the gold fact in context.
Derived probes: CounterFact = filled template; MQuAKE = rewrite question;
RippleEdits = original-fact prompt minus the old object. Re-runs ONLY hold
for the oracle and real-retriever contexts and patches results/armB.json,
keeping the original numbers under hold_probe_fix.before.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from editbench.datasets_common import match  # noqa: E402
from editbench.evaluate import generate, load_model  # noqa: E402
from editbench.methods_retrieval import build_index, fact_text, retrieve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def derive(recs):
    cf, mq, rp = {}, {}, {}
    for r in json.loads((ROOT / "data" / "raw" / "counterfact.json").read_text(encoding="utf-8")):
        rw = r.get("requested_rewrite", {})
        if rw.get("subject") and rw.get("prompt", "").count("{}") == 1:
            cf[f"counterfact-{r.get('case_id', '?')}"] = rw["prompt"].replace("{}", rw["subject"])
    for c in [json.loads(l) for l in open(
            ROOT / "data" / "raw" / "MQuAKE" / "datasets" / "MQuAKE-CF.json", encoding="utf-8")]:
        for j, rw in enumerate(c.get("requested_rewrite", [])):
            mq[f"mquake-cf-{c.get('case_id', '?')}-{j}"] = rw.get("question", "")
    for name in ("popular", "random", "recent"):
        p = ROOT / "data" / "raw" / "RippleEdits" / "data" / "benchmark" / f"{name}.json"
        if not p.exists():
            continue
        for i, e in enumerate(json.loads(open(p, encoding="utf-8").read())):
            orig = (e.get("edit", {}).get("original_fact") or {}).get("prompt", "")
            # Copula prompts: cut the old object after the last " is ".
            if " is " in orig:
                probe = orig.rsplit(" is ", 1)[0] + " is"
            else:
                probe = ""
            rp[f"rippleedits-{name}-{i:04d}"] = probe
    return {**cf, **mq, **rp}


def main():
    arm = json.loads((ROOT / "results" / "armB.json").read_text(encoding="utf-8"))
    recs = [json.loads(l) for l in (ROOT / "results" / "benchmark_main.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    probes = derive(recs)
    model, tok = load_model()
    enc, index = build_index(recs)
    by_id = {r["edit_id"]: r for r in recs}
    before = {"oracle_hold": arm["oracle"]["hold"], "real_hold": arm["real"]["hold"]}
    for variant in ("oracle", "real"):
        hit = n = 0
        for eid, probe in probes.items():
            r = by_id.get(eid)
            if r is None or not probe or not r["new_object"]:
                continue
            ctx = (f"Fact: {fact_text(r)}." if variant == "oracle"
                   else f"Fact: {fact_text(retrieve(enc, index, recs, probe)[0])}.")
            texts, _ = generate(model, tok, [f"{ctx}\nQ: {probe}\nA:"])
            n += 1
            hit += match(texts[0], [r["new_object"]])
        arm[variant]["hold"] = round(hit / max(n, 1), 4)
        arm[variant]["n_hold"] = n
        print(f"{variant} hold (fixed probes): {arm[variant]['hold']} n={n}", flush=True)
    arm["hold_probe_fix"] = {"before": before, "n_probes": len(probes),
                             "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (ROOT / "results" / "armB.json").write_text(json.dumps(arm, indent=1))
    print("armB.json patched")


if __name__ == "__main__":
    main()
