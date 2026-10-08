"""Pilot: base-model generation on 12 records (4/family) to validate the harness.

Writes results/pilot_base.json. No edits, no claims — harness check only.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from editbench.datasets_common import match  # noqa: E402
from editbench.evaluate import generate, load_model, peak_vram_gb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    recs = [json.loads(l) for l in open(ROOT / "results" / "benchmark_main.jsonl", encoding="utf-8")]
    by_fam = {}
    for r in recs:
        fam = r["source"].split("-")[0]
        fam = "rippleedits" if r["source"].startswith("rippleedits") else r["source"]
        by_fam.setdefault(fam, []).append(r)
    pilot = []
    for fam in ("counterfact", "mquake-cf", "rippleedits"):
        pilot.extend(by_fam[fam][:4])
    model, tok = load_model()
    torch = __import__("torch")
    torch.cuda.reset_peak_memory_stats()
    prompts = [r["new_object"] if r["source"] == "counterfact" else r["ripple_questions"][0]["question"]
               if r["ripple_questions"] else r["new_object"] for r in pilot]
    texts, secs = generate(model, tok, prompts)
    rows = []
    for r, p, t in zip(pilot, prompts, texts):
        rows.append({"edit_id": r["edit_id"], "prompt": p,
                     "completion": t[:200], "chars": len(t)})
    out = {"n": len(rows), "seconds": round(secs, 1),
           "peak_vram_gb": round(peak_vram_gb(), 2), "rows": rows,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (ROOT / "results" / "pilot_base.json").write_text(json.dumps(out, indent=1))
    print(f"pilot: {len(rows)} probes in {secs:.1f}s, peak {peak_vram_gb():.2f} GB")


if __name__ == "__main__":
    main()
