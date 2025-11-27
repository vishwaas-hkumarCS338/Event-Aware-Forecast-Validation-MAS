# indexer_simple.py
"""
Simple file-based indexer:
- Reads all .txt files in data/events
- Splits by newline into chunks (empty lines ignored; if file has no newline, whole file is one chunk)
- Embeds every chunk with utils.utils.embed_text
- L2-normalizes embeddings and saves:
    - data/events_embeddings.npy  (float32, shape [N, D])
    - data/events_metadata.json   (list of {id, doc_id, chunk})
"""

from pathlib import Path
import json
import numpy as np
from utils.utils import embed_text
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data/events")
EMB_PATH = Path("data/events_embeddings.npy")
META_PATH = Path("data/events_metadata.json")
BATCH = 32


def build_index():
    if not DATA_DIR.exists():
        raise SystemExit(f"Missing folder: {DATA_DIR.resolve()}")

    files = sorted(DATA_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {DATA_DIR.resolve()}")

    docs = []
    total_files = 0
    for p in files:
        total_files += 1
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            # skip empty files
            continue
        # split by newline → multiple chunks per file (ignored if no newline)
        raw_chunks = [line.strip() for line in text.split("\n")]
        chunks = [c for c in raw_chunks if c] or [text]
        for i, ch in enumerate(chunks):
            docs.append({"id": f"{p.name}#{i}", "doc_id": p.name, "chunk": ch})

    if not docs:
        raise SystemExit(
            "No non-empty chunks found. Ensure your .txt files contain text (one chunk per line "
            "or a single paragraph)."
        )

    texts = [d["chunk"] for d in docs]
    vectors = []
    print(f"Embedding {len(texts)} chunks from {total_files} files in batches of {BATCH}...")
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        # embed_text must return a 1D numpy array (same dimension for all)
        embs = [embed_text(t) for t in batch]
        # Safety: ensure float32 & consistent shapes
        embs = [np.asarray(e, dtype="float32") for e in embs]
        vectors.extend(embs)

    # Stack and L2-normalize row-wise
    X = np.vstack(vectors).astype("float32")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    X = X / norms

    # Save artifacts
    EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, X)

    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved embeddings to {EMB_PATH} with shape {X.shape}")
    print(f"Saved metadata to  {META_PATH} with {len(docs)} chunks")


if __name__ == "__main__":
    build_index()
