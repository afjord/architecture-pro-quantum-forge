import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import faiss
import requests
from sentence_transformers import SentenceTransformer
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("rag_telegram_bot")

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss.index"
METADATA_PATH = ARTIFACTS_DIR / "chunks_metadata.json"

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
TOP_K = int(os.getenv("TOP_K", "4"))
CONTEXT_K = int(os.getenv("CONTEXT_K", "3"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.35"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "3200"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

FEW_SHOT_EXAMPLES = [
    {
        "question": "Who is Xarn Velgor?",
        "answer": "Xarn Velgor was once Aren Veyra, a gifted Aether Knight who fell under Emperor Malrec's influence and became the chief enforcer of the Helion Dominion.",
        "reasoning_steps": [
            "I look for the chunk that directly defines Xarn Velgor.",
            "The retrieved chunk says he was once Aren Veyra and later became the chief enforcer of the Helion Dominion.",
            "So I answer with that identity and role."
        ],
    },
    {
        "question": "What do you know about Kharos IV?",
        "answer": "Kharos IV is an arid frontier world with scarcity, isolated homesteads and a harsh economy; it is also Kael Veyra's childhood home.",
        "reasoning_steps": [
            "I identify the chunk dedicated to Kharos IV.",
            "The chunk describes it as an arid frontier world with scarcity and small-scale trade.",
            "It also states that it is Kael Veyra's childhood home, so I include both points."
        ],
    },
]

@dataclass
class SearchResult:
    score: float
    chunk_id: str | None
    source: str | None
    title: str | None
    text: str | None


class RagEngine:
    def __init__(self) -> None:
        self.metadata = self._load_metadata(METADATA_PATH)
        self.index = self._load_index(FAISS_INDEX_PATH)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    @staticmethod
    def _load_metadata(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_index(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Index not found: {path}")
        return faiss.read_index(str(path))

    def embed_query(self, query: str):
        return self.embedding_model.encode([query], normalize_embeddings=True)

    def search(self, query: str, top_k: int = TOP_K) -> List[SearchResult]:
        query_vector = self.embed_query(query)
        distances, indices = self.index.search(query_vector, top_k)

        results: List[SearchResult] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            item = self.metadata[idx]
            results.append(
                SearchResult(
                    score=float(score),
                    chunk_id=item.get("chunk_id"),
                    source=item.get("source_path") or item.get("source"),
                    title=item.get("title"),
                    text=item.get("text"),
                )
            )
        return results

    @staticmethod
    def should_answer(results: List[SearchResult], min_score: float = MIN_SCORE) -> bool:
        if not results:
            return False
        return results[0].score >= min_score

    @staticmethod
    def build_context(results: List[SearchResult], top_k: int = CONTEXT_K) -> str:
        parts: List[str] = []
        total_chars = 0
        for i, item in enumerate(results[:top_k], start=1):
            text = (item.text or "").strip()
            chunk = (
                f"[Chunk {i}]\n"
                f"Title: {item.title or 'unknown'}\n"
                f"Source: {item.source or 'unknown'}\n"
                f"Text: {text}"
            )
            if total_chars + len(chunk) > MAX_CONTEXT_CHARS:
                break
            parts.append(chunk)
            total_chars += len(chunk)
        return "\n\n".join(parts)

    @staticmethod
    def build_prompt(question: str, context: str) -> str:
        few_shot_str = []
        for example in FEW_SHOT_EXAMPLES:
            few_shot_str.append(
                "Example:\n"
                f"Question: {example['question']}\n"
                f"Reasoning steps:\n- " + "\n- ".join(example["reasoning_steps"]) + "\n"
                f"Answer: {example['answer']}"
            )
        joined_examples = "\n\n".join(few_shot_str)

        return f"""
You are a Telegram knowledge-base assistant for a private fictional company wiki.

Use only the provided context. If the context does not contain enough information, answer exactly: I don't know.
Do not use outside knowledge. Do not guess.

Return valid JSON with this schema:
{{
  "reasoning_steps": ["short step 1", "short step 2", "short step 3"],
  "answer": "final answer",
  "sources": ["title_1", "title_2"]
}}

Rules:
- reasoning_steps must be short, explicit, and based only on the context.
- answer must be concise.
- sources must contain source titles only.
- If information is insufficient, return:
{{
  "reasoning_steps": ["I could not find enough relevant information in the retrieved context."],
  "answer": "I don't know.",
  "sources": []
}}

{joined_examples}

User question:
{question}

Retrieved context:
{context}
""".strip()

    @staticmethod
    def call_ollama(prompt: str) -> Dict[str, Any]:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        content = data["message"]["content"].strip()
        return json.loads(content)

    def answer(self, question: str) -> Dict[str, Any]:
        results = self.search(question, top_k=TOP_K)
        if not self.should_answer(results):
            return {
                "reasoning_steps": ["I could not find enough relevant information in the retrieved context."],
                "answer": "I don't know.",
                "sources": [],
                "used_context": False,
                "top_score": results[0].score if results else None,
            }

        context = self.build_context(results)
        prompt = self.build_prompt(question, context)
        llm_result = self.call_ollama(prompt)

        answer = llm_result.get("answer", "").strip() or "I don't know."
        reasoning_steps = llm_result.get("reasoning_steps", [])
        if not isinstance(reasoning_steps, list):
            reasoning_steps = [str(reasoning_steps)]
        sources = llm_result.get("sources", [])
        if not isinstance(sources, list):
            sources = []

        return {
            "reasoning_steps": reasoning_steps[:3],
            "answer": answer,
            "sources": sources,
            "used_context": True,
            "top_score": results[0].score,
            "retrieved": [
                {
                    "title": r.title,
                    "source": r.source,
                    "chunk_id": r.chunk_id,
                    "score": r.score,
                }
                for r in results[:CONTEXT_K]
            ],
        }


def render_answer(payload: Dict[str, Any]) -> str:
    if payload["answer"] == "I don't know.":
        return "🤖 I don't know."

    steps = payload.get("reasoning_steps", [])
    steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps)) if steps else ""
    sources = payload.get("sources", [])
    sources_text = ", ".join(sources) if sources else "unknown"

    return (
        f"🧠 Reasoning steps:\n{steps_text}\n\n"
        f"✅ Answer:\n{payload['answer']}\n\n"
        f"📚 Sources: {sources_text}"
    )


ENGINE: RagEngine | None = None


def get_engine() -> RagEngine:
    global ENGINE
    if ENGINE is None:
        ENGINE = RagEngine()
    return ENGINE


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = (
        "Hi! I am your RAG bot.\n\n"
        "Send me a question about the knowledge base, for example:\n"
        "- Who is Xarn Velgor?\n"
        "- What do you know about Kharos IV?\n"
        "- What weapon do Aether Knights use?"
    )
    await update.message.reply_text(message)


async def ask_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    question = update.message.text.strip()
    await update.message.chat.send_action(action=ChatAction.TYPING)

    try:
        result = get_engine().answer(question)
        await update.message.reply_text(render_answer(result))
    except Exception as exc:
        logger.exception("Failed to process question")
        await update.message.reply_text(f"Error: {exc}")


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Commands:\n"
        "/start - start the bot\n"
        "/help - show help\n\n"
        "Then just send any text question."
    )


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("Set TELEGRAM_BOT_TOKEN in environment variables.")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ask_handler))

    logger.info("Telegram bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
