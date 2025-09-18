# rag/retrieve.py
from pathlib import Path
import json
from typing import List, Dict
import numpy as np, faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

DATA = Path("data/processed/rag_index")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_artifacts():
    passages = json.loads((DATA / "passages.json").read_text())
    metas = json.loads((DATA / "metas.json").read_text())
    tokens = json.loads((DATA / "bm25_tokens.json").read_text())
    bm25 = BM25Okapi(tokens)
    idx = faiss.read_index(str(DATA / "faiss.index"))
    model = SentenceTransformer(EMB_MODEL)
    return passages, metas, bm25, idx, model

def search(query: str, k: int = 6, alpha: float = 0.5) -> List[Dict]:
    passages, metas, bm25, idx, model = load_artifacts()
    bm = bm25.get_scores(query.lower().split())
    bm = (bm - bm.min()) / (np.ptp(bm) + 1e-9)
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = idx.search(qv.astype(np.float32), min(k*6, len(passages)))
    dense = np.zeros_like(bm); dense[I[0]] = D[0]
    s = alpha * bm + (1 - alpha) * dense
    top = s.argsort()[::-1][:k]
    out = []
    for i in top:
        out.append({
            "passage": passages[i][:900] + ("…" if len(passages[i]) > 900 else ""),
            "doc_id": metas[i]["doc_id"],
            "pub_hint": metas[i]["pub_hint"],
            "score": float(s[i])
        })
    return out

def stitched_context(query: str, k: int = 6) -> str:
    hits = search(query, k=k)
    blocks = []
    for h in hits:
        blocks.append(f"[{h['doc_id']}] {h['passage']}")
    return "\n\n".join(blocks)
