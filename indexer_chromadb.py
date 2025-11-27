# indexer_chromadb.py
import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
from utils.utils import embed_text

DATA_DIR = Path("data/events")
COLLECTION_NAME = "event_docs"
PERSIST_DIR = "data/chroma_persist"

def build_index():
    os.makedirs(PERSIST_DIR, exist_ok=True)
    settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIR)
    client = chromadb.Client(settings=settings)

    # recreate collection to ensure clean index
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    ids = []
    metadatas = []
    docs = []
    for p in DATA_DIR.glob("*.txt"):
        text = p.read_text(encoding="utf-8").strip()
        chunks = [line.strip() for line in text.split("\n") if line.strip()]
        if not chunks:
            chunks = [text]
        for i, ch in enumerate(chunks):
            doc_id = f"{p.name}#{i}"
            ids.append(doc_id)
            metadatas.append({"doc_id": p.name, "chunk_id": doc_id})
            docs.append(ch)

    if not docs:
        raise RuntimeError("No event .txt files found in data/events.")

    # embed in batches
    vectors = []
    B = 32
    for i in range(0, len(docs), B):
        batch = docs[i:i+B]
        embs = [embed_text(t).tolist() for t in batch]
        vectors.extend(embs)

    # add and persist
    collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=vectors)
    client.persist()
    print(f"Indexed {len(docs)} event chunks into Chromadb at {PERSIST_DIR}")

if __name__ == "__main__":
    build_index()
