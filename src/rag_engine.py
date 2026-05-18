# src/rag_engine.py
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "rag_docs"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def load_documents():
    docs = []
    for file in DOCS_DIR.glob("*.txt"):
        docs.append((file.name, file.read_text()))
    return docs

DOCUMENTS = load_documents()
EMBEDDINGS = model.encode([d[1] for d in DOCUMENTS])

def retrieve_context(query: str, k=3):
    q_emb = model.encode([query])[0]
    scores = np.dot(EMBEDDINGS, q_emb)
    top_idx = np.argsort(scores)[-k:][::-1]

    results = []
    for idx in top_idx:
        name, text = DOCUMENTS[idx]
        results.append(f"[{name}]\n{text}\n")

    return "\n".join(results)
