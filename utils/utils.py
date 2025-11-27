"""
utils/utils.py
-----------------------------------------------
Utility helpers for embeddings and retrieval (RAG)
Works with prebuilt events_embeddings.npy + metadata.json
"""

import os
import numpy as np
import json
from openai import OpenAI
from numpy.linalg import norm

# Load environment
# utils/utils.py
from dotenv import load_dotenv

# --- force load .env even if running via uvicorn subprocess ---
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY not loaded. Check your .env file!")

client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------------------------------------------
# Embedding utilities (for indexing and retrieval)
# ----------------------------------------------------
def embed_texts(texts, model="text-embedding-3-small"):
    """Batch embed multiple texts using OpenAI embeddings API."""
    resp = client.embeddings.create(model=model, input=texts)
    return [np.array(item.embedding, dtype="float32") for item in resp.data]


def embed_text(text, model="text-embedding-3-small"):
    """Single text embed."""
    return embed_texts([text], model=model)[0]


# ----------------------------------------------------
# Load event embeddings + metadata (used by RAG)
# ----------------------------------------------------
def load_event_index():
    emb_path = "data/events_embeddings.npy"
    meta_path = "data/events_metadata.json"
    if not os.path.exists(emb_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("❌ events index not found. Run indexer_simple.py first.")
    embeddings = np.load(emb_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    return embeddings, metadata


# ----------------------------------------------------
# Core retrieval (cosine similarity)
# ----------------------------------------------------
def get_top_k_chunks(query, k=2):
    """Return top-k most relevant event chunks using cosine similarity."""
    embeddings, metadata = load_event_index()
    query_emb = embed_text(query)
    similarities = (embeddings @ query_emb) / (norm(embeddings, axis=1) * norm(query_emb) + 1e-8)
    top_idx = np.argsort(similarities)[::-1][:k]

    top_chunks = []
    for i in top_idx:
        top_chunks.append({
            "text": metadata[i]["text"],
            "meta": metadata[i]["meta"],
            "score": float(similarities[i])
        })
    return top_chunks
def cosine_sim(a, b):
    """Compute cosine similarity between two numpy vectors."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))