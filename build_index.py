import json
import time
from pathlib import Path

from index_utils import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    load_chunks_from_dir,
    list_docs,
    save_faiss_or_fallback,
    try_sentence_transformers,
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR
KB_DIR = PROJECT_ROOT / "knowledge_base"
OUT_DIR = BASE_DIR / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    started = time.time()

    chunks = load_chunks_from_dir(KB_DIR, PROJECT_ROOT)
    texts = [c["text"] for c in chunks]
    vectors, emb_meta = try_sentence_transformers(texts)
    artifact_name = save_faiss_or_fallback(vectors, chunks, OUT_DIR)

    summary = {
        "knowledge_base": "knowledge_base",
        "documents_count": len(list_docs(KB_DIR)),
        "chunks_count": len(chunks),
        "chunk_size_words": CHUNK_SIZE_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        "index_artifact": artifact_name,
        **emb_meta,
        "generation_time_seconds": round(time.time() - started, 3),
    }

    (OUT_DIR / "index_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()