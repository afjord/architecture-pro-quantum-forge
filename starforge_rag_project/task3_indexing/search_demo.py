import faiss
import json
import os
import time
from sentence_transformers import SentenceTransformer

ARTIFACTS_DIR = "artifacts"
FAISS_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "chunks_metadata.json")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


def load_metadata(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index(path: str):
    return faiss.read_index(path)


def load_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


model = load_model()


def embed_query(query: str):
    query_for_embedding = f"Represent this sentence for searching relevant passages: {query}"
    vector = model.encode([query_for_embedding], normalize_embeddings=True)
    return vector


def search(query: str, top_k: int = TOP_K):
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"Index not found: {FAISS_INDEX_PATH}")

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    metadata = load_metadata(METADATA_PATH)
    index = load_index(FAISS_INDEX_PATH)

    query_vector = embed_query(query)
    distances, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        item = metadata[idx]
        results.append(
            {
                "score": float(score),
                "chunk_id": item.get("chunk_id"),
                "source": item.get("source_path"),
                "title": item.get("title"),
                "text": item.get("text"),
            }
        )
    return results


if __name__ == "__main__":
    queries = [
        "Who is Xarn Velgor?",
        "What do you know about Kharos IV?",
        "What weapon do Aether Knights use?",
        "Who is Darth Vader?",
    ]
    for q in queries:
        print(f"\n=== QUERY: {q} ===")
        for item in search(q, top_k=3):
            print(json.dumps(item, ensure_ascii=False, indent=2))
