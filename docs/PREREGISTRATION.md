# Pre-registration: EditBench-3Way main comparison (written before any edit run)

Frozen: model Qwen2.5-1.5B-Instruct (base, unedited; HF snapshot hash recorded in
`results/run_manifest.json` at run time), benchmark `results/benchmark_main.jsonl`
(1,500 records, sha256 df85f8566b1a706c), decoding greedy (temperature 0,
batch size 1 — batch effects are real, see FlipGate), max 32 new tokens for
answer probes. Seeds fixed in `configs/run.toml`.

## Arms (same facts, same model, same harness)

- **A1 ROME** (EasyEdit, Qwen2.5 hparams): one edit at a time; evaluated on a
  150-edit single-edit slice (50/family).
- **A2 MEMIT** (EasyEdit): batches of 100; two batches (200 edits), same slice families.
- **B Retrieval**: the gold edit fact prepended verbatim (oracle retriever);
  plus a real-retriever variant (sentence-transformers + FAISS over the edit
  set, top-1) whose retrieval hit rate is reported separately. Full 1,500.
- **C Graph lookup**: question compiled to a query over the explicit triple
  store built from the edit set; the model answers from returned triples.
  Full 1,500. Ripple-by-construction only where the graph covers the relation.

## Metrics (alias-aware exact match; paired bootstrap over edits, 4,000 resamples)

- **H1 (hold):** edited fact answered with the new object. Claim: B-oracle ≥ 95%; report A1/A2/C without a superiority claim.
- **H2 (ripple-through):** multi-hop/entailment questions needing the edit.
  Pre-stated expectation from the literature (not a pass/fail): A1/A2 < B-oracle;
  C leads only on covered relations — report covered vs uncovered separately.
- **H3 (locality):** unrelated facts unchanged. CounterFact neighborhoods use
  base-model generations as the fixed target (filled before any arm runs).
  Fail line: any arm below 80% locality is flagged, not averaged away.
- **H4 (cost):** wall seconds + peak VRAM at 10/100/1,000 edits per arm.
  ROME's per-edit causal tracing is expected to dominate at scale.

## Analysis rules (fixed now)

- No LLM judge; scorer is alias containment only.
- zsRE excluded by design (CounterFact subsumes its role; recorded in milestone 1).
- RippleEdits condition queries are recorded, not scored, in v1.
- Multi-edit MQuAKE cases: each rewrite scores its own questions; reported
  separately from single-edit cases.
- Anything tuned after seeing results gets a `_posthoc` suffix and no claim.
