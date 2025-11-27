# retriever_chromadb.py
import os
from typing import List
import chromadb
from chromadb.config import Settings
from utils.utils import embed_text

PERSIST_DIR = "data/chroma_persist"
COLLECTION_NAME = "event_docs"

class ChromaRetriever:
    def __init__(self):
        # ensure persist dir exists
        os.makedirs(PERSIST_DIR, exist_ok=True)

        # New client construction (explicit keyword argument)
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=PERSIST_DIR
        )
        # pass settings by keyword to avoid legacy-config detection
        self.client = chromadb.Client(settings=settings)
        # create collection if missing
        try:
            self.collection = self.client.get_collection(name=COLLECTION_NAME)
        except Exception:
            self.collection = self.client.create_collection(name=COLLECTION_NAME)

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        # embed with OpenAI helper
        q_emb = embed_text(query).tolist()
        results = self.collection.query(queries=[q_emb], n_results=top_k, include=["documents","metadatas","distances"])
        docs = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            docs.append({"text": doc, "meta": meta, "distance": float(dist)})
        return docs

if __name__ == "__main__":
    r = ChromaRetriever()
    print("ChromaRetriever initialized. Collections:", [c.name for c in r.client.list_collections()])
