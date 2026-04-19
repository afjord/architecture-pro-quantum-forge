import faiss
import json
import os
import requests
from sentence_transformers import SentenceTransformer

ARTIFACTS_DIR = "artifacts"
FAISS_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "chunks_metadata.json")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
MIN_SCORE = 0.35

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"


def load_metadata(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index(path: str):
    return faiss.read_index(path)


def load_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


model = load_model()


def embed_query(query: str):
    vector = model.encode([query], normalize_embeddings=True)
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
                "source": item.get("source_path") or item.get("source"),
                "title": item.get("title"),
                "text": item.get("text"),
            }
        )
    return results


def should_answer(search_results, min_score: float = MIN_SCORE):
    if not search_results:
        return False
    return search_results[0]["score"] >= min_score


def build_context(search_results, top_k: int = TOP_K):
    parts = []
    for i, item in enumerate(search_results[:top_k], start=1):
        parts.append(
            f"[Chunk {i}]\n"
            f"Title: {item.get('title')}\n"
            f"Source: {item.get('source')}\n"
            f"Text: {item.get('text')}"
        )
    return "\n\n".join(parts)


def build_prompt(question: str, context: str):
    return f"""You are a QA assistant for a private fictional knowledge base.

Answer only using the provided context.
If the context does not contain enough information to answer the question, say exactly:
I don't know.

Rules:
- Do not use outside knowledge.
- Do not guess.
- Keep the answer concise.
- If possible, mention the source title.

Question:
{question}

Context:
{context}
"""


def call_ollama(prompt: str):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["response"].strip()


def answer_with_rag(question: str):
    search_results = search(question, top_k=TOP_K)

    if not should_answer(search_results, min_score=MIN_SCORE):
        return {
            "answer": "I don't know.",
            "used_context": False,
            "top_score": search_results[0]["score"] if search_results else None,
            "sources": [],
        }

    context = build_context(search_results, top_k=TOP_K)
    prompt = build_prompt(question, context)
    answer = call_ollama(prompt)

    if not answer:
        answer = "I don't know."

    return {
        "answer": answer,
        "used_context": True,
        "top_score": search_results[0]["score"],
        "sources": [
            {
                "title": item.get("title"),
                "source": item.get("source"),
                "chunk_id": item.get("chunk_id"),
                "score": item.get("score"),
            }
            for item in search_results[:TOP_K]
        ],
    }


if __name__ == "__main__":
    queries = [
        "Who is Xarn Velgor?",
        "What do you know about Kharos IV?",
        "What weapon do Aether Knights use?",
        "Who is Darth Vader?",
    ]

    for q in queries:
        print(f"\n=== QUERY: {q} ===")
        result = answer_with_rag(q)
        print(json.dumps(result, ensure_ascii=False, indent=2))