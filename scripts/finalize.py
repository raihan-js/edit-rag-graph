"""Deduplicate across sources and sample the 500/source main table (milestone 1).

Overlap rule (docs/BENCHMARK.md): same (subject, relation) in two sources is
one record — keep the one with the most ripple questions, tag the dropped
sources in `also_in`. Then stratified sample: 500 per source family
(counterfact, mquake-cf, rippleedits-*), seed fixed. Writes
results/benchmark_main.jsonl + results/benchmark_manifest.json.
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 20261008
CAP = 500
ROOT = Path(__file__).resolve().parent.parent


def load():
    recs = []
    for f in sorted((ROOT / "results").glob("benchmark_*.jsonl")):
        if f.name == "benchmark_main.jsonl":
            continue
        recs.extend(json.loads(l) for l in open(f, encoding="utf-8"))
    return recs


def family(r):
    s = r["source"]
    return "rippleedits" if s.startswith("rippleedits") else s


def main():
    recs = load()
    print(f"input: {len(recs)} records")
    groups = defaultdict(list)
    for r in recs:
        groups[(r["subject"].lower().strip(), r["relation"])].append(r)
    deduped = []
    dropped = 0
    for rs in groups.values():
        if len(rs) == 1:
            deduped.append(rs[0])
            continue
        rs.sort(key=lambda r: -len(r["ripple_questions"]))
        keep = rs[0]
        keep["also_in"] = sorted({x["source"] for x in rs[1:]})
        dropped += len(rs) - 1
        deduped.append(keep)
    print(f"after overlap dedup: {len(deduped)} (dropped {dropped})")
    rng = random.Random(SEED)
    main_rows = []
    by_fam = defaultdict(list)
    for r in deduped:
        by_fam[family(r)].append(r)
    for fam, rs in sorted(by_fam.items()):
        rng.shuffle(rs)
        main_rows.extend(rs[:CAP])
        print(f"  {fam}: {len(rs)} -> {min(CAP, len(rs))} sampled")
    out = ROOT / "results" / "benchmark_main.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in main_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    (ROOT / "results" / "benchmark_manifest.json").write_text(json.dumps(
        {"seed": SEED, "cap_per_family": CAP, "n": len(main_rows),
         "sha256": h,
         "families": {fam: sum(1 for r in main_rows if family(r) == fam)
                      for fam in by_fam}}, indent=1))
    print(f"main table: {len(main_rows)} records sha256 {h[:16]}")


if __name__ == "__main__":
    main()
