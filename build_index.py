import json
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 40

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR
KB_DIR = PROJECT_ROOT / "knowledge_base"
OUT_DIR = BASE_DIR / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def list_docs():
    return sorted([p for p in KB_DIR.glob("*.md")])


def split_words(text: str):
    return re.findall(r"\S+", text)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS):
    words = split_words(text)
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "text": chunk_text,
            "start_word": start,
            "end_word": end,
            "word_count": len(chunk_words),
        })
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_chunks():
    docs = []
    for path in list_docs():
        text = path.read_text(encoding="utf-8").strip()
        title = path.stem
        for idx, chunk in enumerate(chunk_text(text)):
            docs.append({
                "chunk_id": f"{title}__{idx}",
                "title": title,
                "source_path": str(path.relative_to(PROJECT_ROOT)),
                "chunk_index": idx,
                **chunk,
            })
    return docs


def try_sentence_transformers(texts):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32"), {
            "embedding_backend": "sentence-transformers",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dim": int(vectors.shape[1]),
        }
    except Exception as e:
        print(f"[WARN] Не удалось загрузить SentenceTransformer: {e}")
        print("[WARN] Используется локальный fallback на HashingVectorizer (только для офлайн-демо).")
        vectorizer = HashingVectorizer(n_features=EMBEDDING_DIM, alternate_sign=False, norm="l2")
        matrix = vectorizer.transform(texts)
        vectors = matrix.astype("float32").toarray()
        return vectors, {
            "embedding_backend": "hashing-fallback",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dim": EMBEDDING_DIM,
        }


def save_faiss_or_fallback(vectors, chunks):
    metadata_path = OUT_DIR / "chunks_metadata.json"
    metadata_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import faiss  # type: ignore
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(OUT_DIR / "faiss.index"))
        return "faiss.index"
    except Exception as e:
        print(f"[WARN] FAISS недоступен: {e}")
        with open(OUT_DIR / "vectors.pkl", "wb") as f:
            pickle.dump(vectors, f)
        return "vectors.pkl"


def main():
    started = time.time()
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    vectors, emb_meta = try_sentence_transformers(texts)
    artifact_name = save_faiss_or_fallback(vectors, chunks)

    summary = {
        "knowledge_base": "knowledge_base",
        "documents_count": len(list_docs()),
        "chunks_count": len(chunks),
        "chunk_size_words": CHUNK_SIZE_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        "index_artifact": artifact_name,
        **emb_meta,
        "generation_time_seconds": round(time.time() - started, 3),
    }
    (OUT_DIR / "index_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
