# indexer.py
import json
import numpy as np
from pathlib import Path
from utils.utils import embed_texts
from tqdm import tqdm
import faiss

DATA_EVENTS_DIR = Path("data/events")
INDEX_PATH = Path("data/events_index.faiss")
META_PATH = Path("data/events_metadata.json")

def build_index():
    docs = []
    for p in DATA_EVENTS_DIR.glob("*.txt"):
        text = p.read_text(encoding="utf-8").strip()
        raw_chunks = [line.strip() for line in text.split("\n") if line.strip()]
        if not raw_chunks:
            raw_chunks = [text]
        for i, ch in enumerate(raw_chunks):
            docs.append({"doc_id": p.name, "chunk_id": f"{p.name}#{i}", "text": ch})
    if not docs:
        raise RuntimeError("No .txt files found in data/events")
    texts = [d["text"] for d in docs]
    print(f"Embedding {len(texts)} chunks...")
    vectors = embed_texts(texts)
    dim = vectors[0].shape[0]
    xb = np.vstack(vectors).astype('float32')
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(dim)
    index.add(xb)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"Built index: {INDEX_PATH}, metadata: {META_PATH}")

if __name__ == "__main__":
    build_index()
