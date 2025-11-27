# retriever_simple.py
"""
Simple in-memory retriever using cosine similarity on precomputed embeddings.
Loads:
- data/events_embeddings.npy  (float32, L2-normalized, shape [N, D])
- data/events_metadata.json   (list of {id, doc_id, chunk})
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict
from utils.utils import embed_text

EMB_PATH = Path("data/events_embeddings.npy")
META_PATH = Path("data/events_metadata.json")


class Retriever:
    def __init__(self):
        if not EMB_PATH.exists() or not META_PATH.exists():
            raise RuntimeError(
                "Embeddings or metadata missing. Run indexer_simple.py first "
                f"(expected {EMB_PATH} and {META_PATH})."
            )

        # Load embeddings
        X = np.load(str(EMB_PATH))
        if not isinstance(X, np.ndarray) or X.ndim != 2 or X.size == 0:
            raise RuntimeError(f"Bad embeddings array in {EMB_PATH}: expected 2D non-empty array.")

        # Ensure float32 and L2-normalized row-wise
        X = X.astype("float32", copy=False)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.X = X / norms

        # Load metadata
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, list) or len(meta) != self.X.shape[0]:
            raise RuntimeError(
                f"Metadata count ({len(meta) if isinstance(meta, list) else 'invalid'}) "
                f"does not match embeddings rows ({self.X.shape[0]}). "
                "Rebuild with indexer_simple.py."
            )
        self.meta = meta

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if not query or not query.strip():
            return []

        # Embed and normalize query
        qvec = embed_text(query).astype("float32")
        qn = np.linalg.norm(qvec)
        if qn == 0.0:
            # Degenerate vector; no signal
            return []
        qvec_norm = qvec / qn

        # Cosine similarities via dot product (rows are already L2-normalized)
        sims = (self.X @ qvec_norm).astype("float32")

        # Top-k indices (clamped)
        k = int(max(0, min(top_k, sims.shape[0])))
        if k == 0:
            return []
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        # sort those top-k by score desc
        idx = idx[np.argsort(-sims[idx])]

        results = []
        for i in idx:
            m = self.meta[int(i)]
            results.append(
                {
                    "text": m.get("chunk", ""),
                    "meta": {
                        "doc_id": m.get("doc_id", ""),
                        "chunk_id": m.get("id", ""),
                    },
                    "score": float(sims[int(i)]),
                }
            )
        return results


if __name__ == "__main__":
    r = Retriever()
    for hit in r.retrieve("january wine demand moderation", top_k=3):
        print(f"{hit['score']:.3f} | {hit['meta']['doc_id']} | {hit['text'][:80]}")
