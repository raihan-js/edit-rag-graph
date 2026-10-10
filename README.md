# EditBench-3Way: edit the weights, retrieve the fact, or look it up in a graph?

Same facts, same model (Qwen2.5-1.5B-Instruct), same metrics. Three ways to update what a small LLM "knows", compared on whether the update holds, whether it ripples through to related multi-hop questions, whether unrelated facts stay intact, and what each costs at 10, 100 and 1,000 edits.

Status: milestone 1 (unified benchmark). Pre-registration: `docs/PREREGISTRATION.md` (frozen before any edit run).

## Arms

- **A. Weight editing** — ROME (single) and MEMIT (batch) via EasyEdit; fallback is the original ROME/MEMIT code if EasyEdit lacks the model.
- **B. Retrieval** — the fact in the prompt (in-context editing baseline); retriever quality measured separately.
- **C. Graph lookup** — compile the question into a query over an explicit triple store (reuses the GraphProof-QA query DSL); ripples by construction where the graph covers the relation.

## Metrics (per edit set, with paired bootstrap CIs)

Hold rate (edited fact answered with the new object), paraphrase generalization, ripple-through (multi-hop questions needing the edit), locality (unrelated facts unchanged), cost (seconds + peak VRAM at 10/100/1,000 edits).

## Results (main table, n=1,500; alias-match rates, Wilson 95% CI)

| Arm | Hold | Generalization | Locality |
|---|---|---|---|
| B-oracle (gold fact in prompt) | 0.226 [0.204, 0.251] | 0.527 [0.501, 0.552] | 0.287 [0.274, 0.300] |
| B-real (top-1 retrieved, hit rate 0.795) | 0.193 | 0.455 [0.430, 0.481] | 0.285 [0.272, 0.299] |
| C-covered (graph has the relation, 76%) | 0.321 | **0.627 [0.603, 0.651]** | **0.369 [0.348, 0.392]** |
| C-uncovered | 0.011 | — (no gen probes uncovered) | 0.095 [0.084, 0.107] |
| A-ROME (150 singles) | 0.753* | 0.707 [0.629, 0.774] | **0.144 [0.115, 0.180] ⚠** |
| A-MEMIT (2×100 batches) | 0.327* | 0.553 [0.473, 0.631] | **0.124 [0.097, 0.158] ⚠** |

\* A-hold uses the original probes (subject + relation ID); B/C hold uses the fixed probes. Not directly comparable — see the probe note below.

Verdicts: **H1 FAILS** — even the gold fact in context is answered 22.6% of the time; a 1.5B model does not reliably defer to context (this bounds every in-context method, including B). **H2**: C-covered leads B-oracle with non-overlapping intervals, only where covered. **H3 FLAGGED**: both weight-editing arms below 80% locality; non-editing arms hold 29–37%. **H4 (measured, `results/cost.json`)**: per query, retrieval adds 0.005s and graph compile ~0s over 0.63s generation (3.25 GB peak); per edit, ROME 3.7s and MEMIT 8.5s (×10 timed rows) — linear extrapolations give ROME ~6 min / ~62 min and MEMIT ~14 min / ~2.4 h at 100 / 1,000 edits, plus MEMIT's one-time ~7 h stats build on first run.

Probe note (recorded, not tuning): the first hold probes were garbage for CounterFact/MQuAKE (`"subject P103"`, multi-hop as single-hop). Fixed probes (filled template, rewrite question, original-minus-object) re-ran for B-oracle/B-real; A keeps original probes. Both versions are in `results/` (`armB.json` carries before/after).

CI note: pre-registered paired bootstrap needs per-item joint outcomes, which v1 runs did not log. All verdicts are far from every boundary, so Wilson + independent intervals decide nothing differently. Paired re-runs queued as v1.1 rigor, not claims.

## Layout

`src/editbench/{datasets,methods,evaluate}` · `scripts/` (fetch, run, report) · `tests/` · `results/` (committed JSON, never overwritten) · `data/` (gitignored raw downloads).

## Setup (EasyEdit vendored, not tracked)

```bash
git clone --depth 1 https://github.com/zjunlp/EasyEdit third_party_easyedit
.venv/bin/pip install hydra-core higher sentence-transformers nltk
```

Qwen2.5 hparams ship upstream (`hparams/ROME/qwen2.5-7b.yaml`, same family as our 1.5B model).
