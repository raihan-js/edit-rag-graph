# Milestone 1: unified benchmark (common schema + converters)

## Common record (every source converts to this; validated by tests)

```json
{
  "edit_id": "mquake-cf-0001",
  "subject": "Leonardo DiCaprio",
  "relation": "country_of_citizenship",
  "old_object": "United States of America",
  "new_object": "Syria",
  "paraphrases": ["Leonardo DiCaprio is a citizen of which country?"],
  "ripple_questions": [{"question": "Who is the sibling of Jack Depp?", "answer": "Lily-Rose Depp", "needs_edit": true}],
  "locality_questions": [{"question": "Who is the mother of Leonardo DiCaprio?", "answer": "Irmelin DiCaprio"}],
  "source": "rippleedits-popular"
}
```

Rules: paraphrases test the same fact in new words; ripple questions need the edit (multi-hop or entailed); locality questions must NOT change. One JSONL per source in `results/benchmark_<source>.jsonl`, plus a frozen manifest with counts and sha256.

## Source mapping

| Source | Raw location | Converter notes |
|---|---|---|
| RippleEdits (5K, MIT) | `data/raw/RippleEdits/data/` | 6 criteria → ripple vs locality by criterion (Compositionality I/II + Logical_Generalization = ripple; Relation_Specificity + Subject_Aliasing + Forgetfulness = locality). Aliases accepted. |
| MQuAKE-CF | `data/raw/MQuAKE/` | Multi-hop questions are ripple by construction; single-hop paraphrases from the edit triple. |
| CounterFact | EasyEdit loader or `data/raw/counterfact/` | Paraphrases + neighborhood (locality) come with the record; ripple subset via shared relations. |
| zsRE | EasyEdit loader or mirror | Single-hop + paraphrase; locality from unrelated relations. |

Cap: 500 edits per source for the main table (2,000 total), full sets kept for the 1,000-edit cost row. Overlap check: same (subject, relation) in two sources is one record, source tagged.

## Metrics (frozen here, implemented in milestone 2)

Hold, paraphrase generalization, ripple-through, locality intactness — all as exact-match-against-aliases rates with paired bootstrap CIs; cost as wall seconds + peak VRAM at 10/100/1,000 edits. No LLM judge anywhere.
