"""Convert raw datasets to results/benchmark_<source>.jsonl + manifest (milestone 1)."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from editbench.datasets_counterfact import convert as counterfact  # noqa: E402
from editbench.datasets_mquake import convert as mquake  # noqa: E402
from editbench.datasets_rippleedits import convert as ripple  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results"


def dump(name, records):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"benchmark_{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{name}: {len(records)} records -> {path.name} sha256 {h[:16]}")
    return {"source": name, "n": len(records), "sha256": h}


def main():
    manifest = []
    for ex in ("popular", "random", "recent"):
        p = RAW / "RippleEdits" / "data" / "benchmark" / f"{ex}.json"
        if p.exists():
            manifest.append(dump(f"rippleedits-{ex}", ripple(str(p), ex)))
    for name in ("MQuAKE-CF-3k.json", "MQuAKE-CF.json"):
        p = RAW / "MQuAKE" / "datasets" / name
        if p.exists():
            manifest.append(dump("mquake-cf", mquake(str(p))))
            break
    p = RAW / "counterfact.json"
    if p.exists():
        manifest.append(dump("counterfact", counterfact(str(p))))
    (OUT / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("manifest written")


if __name__ == "__main__":
    main()
