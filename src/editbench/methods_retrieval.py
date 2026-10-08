"""Arm B: retrieval-augmented answering.

B-oracle: the gold edit fact prepended verbatim (upper bound for in-context
editing). B-real: top-1 fact from a sentence-transformers + FAISS index over
the edit set; retrieval hit rate reported separately.
"""

from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def fact_text(r):
    return f"{r['subject']} {r['relation']} {r['new_object']}".strip()


def build_index(records):
    import faiss
    import numpy as np

    enc = SentenceTransformer(MODEL)
    texts = [fact_text(r) for r in records]
    emb = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.asarray(emb, dtype="float32"))
    return enc, index


def retrieve(enc, index, records, question, k=1):
    import numpy as np

    q = enc.encode([question], normalize_embeddings=True)
    _, idx = index.search(np.asarray(q, dtype="float32"), k)
    return [records[i] for i in idx[0]]
