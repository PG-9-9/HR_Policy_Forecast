# rag/build_index.py
from pathlib import Path
import json, re, glob
from typing import List, Dict
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text  # Still used, but requires pdfminer.six
from sentence_transformers import SentenceTransformer
import numpy as np, faiss
from rank_bm25 import BM25Okapi

DATA = Path("data")
CORPUS = DATA / "external" / "uk_corpus"
OUT = DATA / "processed" / "rag_index"
OUT.mkdir(parents=True, exist_ok=True)

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

def clean_text(txt: str) -> str:
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def chunk(text: str) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        j = min(i + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[i:j]))
        i = j - CHUNK_OVERLAP if j - CHUNK_OVERLAP > i else j
    return chunks

def parse_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    # keep visible text
    for s in soup(["script", "style", "nav", "footer", "header"]):
        s.decompose()
    txt = soup.get_text(" ")
    return clean_text(txt)

def parse_pdf(path: Path) -> str:
    try:
        txt = extract_text(str(path))
        return clean_text(txt)
    except Exception as e:
        print(f"[WARN] Failed to parse PDF {path}: {e}")
        return ""

def load_docs() -> List[Dict]:
    docs = []
    for f in glob.glob(str(CORPUS / "*")):
        p = Path(f)
        if p.suffix.lower() == ".html":
            text = parse_html(p)
        elif p.suffix.lower() == ".pdf":
            text = parse_pdf(p)
        else:
            continue
        # extract publication-ish date if present
        m = re.search(r"(\b\d{1,2}\s\w+\s20\d{2}\b)|(\b20\d{2}-\d{2}-\d{2}\b)", text)
        pub = m.group(0) if m else ""
        docs.append({"doc_id": p.name, "text": text, "pub_hint": pub})
    return docs

def build_hybrid(docs: List[Dict]):
    # Chunk & collect metadata
    passages, metas = [], []
    for d in docs:
        for ch in chunk(d["text"]):
            passages.append(ch)
            metas.append({"doc_id": d["doc_id"], "pub_hint": d["pub_hint"]})

    # BM25
    tokens = [p.lower().split() for p in passages]
    bm25 = BM25Okapi(tokens)

    # Dense
    model = SentenceTransformer(EMB_MODEL)
    emb = model.encode(passages, convert_to_numpy=True, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb.astype(np.float32))

    # Save
    OUT.mkdir(exist_ok=True, parents=True)
    np.save(OUT / "emb.npy", emb)
    faiss.write_index(idx, str(OUT / "faiss.index"))
    (OUT / "passages.json").write_text(json.dumps(passages), encoding="utf-8")
    (OUT / "metas.json").write_text(json.dumps(metas), encoding="utf-8")
    # BM25 corpus for re-build (store lowercased tokens)
    (OUT / "bm25_tokens.json").write_text(json.dumps(tokens), encoding="utf-8")
    (OUT / "meta_info.json").write_text(json.dumps({"emb_model": EMB_MODEL, "n": len(passages)}), encoding="utf-8")
    print(f"[RAG] {len(passages)} passages indexed.")

if __name__ == "__main__":
    docs = load_docs()
    if not docs:
        raise SystemExit("No UK corpus files. Run: python scripts/download_uk_data.py")
    build_hybrid(docs)