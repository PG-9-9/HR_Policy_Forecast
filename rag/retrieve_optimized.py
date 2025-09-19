# rag/retrieve_optimized.py - Optimized version with lazy loading
from pathlib import Path
import json
from typing import List, Dict, Optional
import numpy as np, faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import threading
import time

DATA = Path("data/processed/rag_index")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Global cache for loaded artifacts
_artifacts_cache = None
_loading_lock = threading.Lock()
_is_loading = False

def load_artifacts_lazy():
    """Load artifacts with lazy initialization and caching"""
    global _artifacts_cache, _is_loading
    
    if _artifacts_cache is not None:
        return _artifacts_cache
    
    with _loading_lock:
        # Double-check pattern
        if _artifacts_cache is not None:
            return _artifacts_cache
            
        if _is_loading:
            # If another thread is loading, wait a bit
            time.sleep(0.1)
            return None
            
        _is_loading = True
        
        try:
            print("Loading RAG artifacts (this may take a moment on first use)...")
            passages = json.loads((DATA / "passages.json").read_text())
            metas = json.loads((DATA / "metas.json").read_text())
            tokens = json.loads((DATA / "bm25_tokens.json").read_text())
            bm25 = BM25Okapi(tokens)
            idx = faiss.read_index(str(DATA / "faiss.index"))
            
            # This will download models on first use
            model = SentenceTransformer(EMB_MODEL)
            
            _artifacts_cache = (passages, metas, bm25, idx, model)
            print("RAG artifacts loaded successfully!")
            return _artifacts_cache
            
        except Exception as e:
            print(f"Error loading RAG artifacts: {e}")
            return None
        finally:
            _is_loading = False

def search_optimized(query: str, k: int = 6, alpha: float = 0.5) -> Optional[List[Dict]]:
    """Search with error handling for model loading"""
    artifacts = load_artifacts_lazy()
    
    if artifacts is None:
        return None
        
    passages, metas, bm25, idx, model = artifacts
    
    try:
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
    except Exception as e:
        print(f"Error in search: {e}")
        return None

def stitched_context_optimized(query: str, k: int = 6) -> str:
    """Context retrieval with graceful fallback"""
    hits = search_optimized(query, k=k)
    
    if hits is None:
        return ""  # Return empty context if RAG is not ready
        
    blocks = []
    for h in hits:
        blocks.append(f"[{h['doc_id']}] {h['passage']}")
    return "\n\n".join(blocks)

# Legacy compatibility
def stitched_context(query: str, k: int = 6) -> str:
    return stitched_context_optimized(query, k)