# EditBench-3Way: edit the weights, retrieve the fact, or look it up in a graph?

Same facts, same model (Qwen2.5-1.5B-Instruct), same metrics. Three ways to update what a small LLM "knows", compared on whether the update holds, whether it ripples through to related multi-hop questions, whether unrelated facts stay intact, and what each costs at 10, 100 and 1,000 edits.

Status: milestone 1 (unified benchmark). Pre-registration: `docs/PREREGISTRATION.md` (frozen before any edit run).

## Arms

- **A. Weight editing** — ROME (single) and MEMIT (batch) via EasyEdit; fallback is the original ROME/MEMIT code if EasyEdit lacks the model.
- **B. Retrieval** — the fact in the prompt (in-context editing baseline); retriever quality measured separately.
- **C. Graph lookup** — compile the question into a query over an explicit triple store (reuses the GraphProof-QA query DSL); ripples by construction where the graph covers the relation.

## Metrics (per edit set, with paired bootstrap CIs)

Hold rate (edited fact answered with the new object), paraphrase generalization, ripple-through (multi-hop questions needing the edit), locality (unrelated facts unchanged), cost (seconds + peak VRAM at 10/100/1,000 edits).

## Layout

`src/editbench/{datasets,methods,evaluate}` · `scripts/` (fetch, run, report) · `tests/` · `results/` (committed JSON, never overwritten) · `data/` (gitignored raw downloads).

## Setup (EasyEdit vendored, not tracked)

```bash
git clone --depth 1 https://github.com/zjunlp/EasyEdit third_party_easyedit
.venv/bin/pip install hydra-core higher sentence-transformers nltk
```

Qwen2.5 hparams ship upstream (`hparams/ROME/qwen2.5-7b.yaml`, same family as our 1.5B model).
