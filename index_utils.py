import json
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 40


def list_docs(kb_dir: Path):
    return sorted([p for p in kb_dir.glob("*.md")])


def split_words(text: str):
    return re.findall(r"\S+", text)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS):
    words = split_words(text)
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_value = " ".join(chunk_words)

        chunks.append({
            "text": chunk_value,
            "start_word": start,
            "end_word": end,
            "word_count": len(chunk_words),
        })

        if end == len(words):
            break

        start = max(end - overlap, start + 1)

    return chunks


def load_chunks_from_dir(kb_dir: Path, project_root: Path):
    docs = []

    for path in list_docs(kb_dir):
        text = path.read_text(encoding="utf-8").strip()
        title = path.stem

        for idx, chunk in enumerate(chunk_text(text)):
            docs.append({
                "chunk_id": f"{title}__{idx}",
                "title": title,
                "source_path": str(path.relative_to(project_root)),
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

        vectorizer = HashingVectorizer(
            n_features=EMBEDDING_DIM,
            alternate_sign=False,
            norm="l2",
        )
        matrix = vectorizer.transform(texts)
        vectors = matrix.astype("float32").toarray()

        return vectors, {
            "embedding_backend": "hashing-fallback",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dim": EMBEDDING_DIM,
        }


def save_faiss_or_fallback(vectors, chunks, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = out_dir / "chunks_metadata.json"
    metadata_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(out_dir / "faiss.index"))
        return "faiss.index"
    except Exception as e:
        print(f"[WARN] FAISS недоступен: {e}")
        with open(out_dir / "vectors.pkl", "wb") as f:
            pickle.dump(vectors, f)
        return "vectors.pkl"