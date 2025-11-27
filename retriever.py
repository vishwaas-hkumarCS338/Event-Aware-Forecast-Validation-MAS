# retriever.py
import faiss
import json
import numpy as np
from pathlib import Path
from utils.utils import embed_text

INDEX_PATH = Path("data/events_index.faiss")
META_PATH = Path("data/events_metadata.json")

class Retriever:
    def __init__(self):
        if not INDEX_PATH.exists() or not META_PATH.exists():
            raise RuntimeError("Index or metadata missing. Run indexer.py first.")
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def retrieve(self, query: str, top_k: int = 5):
        qvec = embed_text(query)
        qvec = qvec.astype('float32')
        faiss.normalize_L2(np.expand_dims(qvec, axis=0))
        D, I = self.index.search(np.expand_dims(qvec, axis=0), top_k)
        results = []
        for idx in I[0]:
            if idx < 0 or idx >= len(self.meta):
                continue
            results.append(self.meta[idx])
        return results

if __name__ == "__main__":
    r = Retriever()
    print(r.retrieve("holiday wine sales in December", top_k=3))
