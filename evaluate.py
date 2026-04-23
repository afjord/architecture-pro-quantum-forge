import json
from datetime import datetime, timezone
from pathlib import Path

from rag_telegram_bot import RagEngine

BASE_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = BASE_DIR / "golden_questions.json"
LOG_PATH = BASE_DIR / "logs" / "golden_question_answers.jsonl"
SUMMARY_PATH = BASE_DIR / "evaluation_summary.json"


def normalize_answer(answer: str) -> str:
    return (answer or "").strip().lower()


def is_unknown_answer(answer: str) -> bool:
    normalized = normalize_answer(answer)
    return normalized in {"i don't know.", "i don't know"}


def is_successful_answer(answer: str, sources: list[str]) -> bool:
    if is_unknown_answer(answer):
        return False
    if len((answer or "").strip()) < 30:
        return False
    if not sources:
        return False
    return True


def main():
    engine = RagEngine()
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    total = 0
    passed = 0
    failed_cases = []

    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        for item in golden:
            total += 1
            question = item["question"]
            expected_mode = item["expected_mode"]

            result = engine.answer(question)
            answer = result.get("answer", "")
            retrieved = result.get("retrieved", [])
            sources = result.get("sources", [])
            top_score = result.get("top_score")

            actual_mode = "unknown" if is_unknown_answer(answer) else "answer"

            passed_case = actual_mode == expected_mode
            if passed_case:
                passed += 1
            else:
                failed_cases.append({
                    "question": question,
                    "expected_mode": expected_mode,
                    "actual_mode": actual_mode,
                    "answer": answer,
                    "sources": sources,
                })

            log_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question": question,
                "answer": answer,
                "chunks_found": len(retrieved) > 0,
                "retrieved_count": len(retrieved),
                "sources": sources,
                "answer_length": len(answer),
                "successful_answer": is_successful_answer(answer, sources),
                "top_score": top_score,
                "expected_mode": expected_mode,
                "actual_mode": actual_mode,
                "passed": passed_case,
            }
            log_file.write(json.dumps(log_record, ensure_ascii=False) + "\n")

    summary = {
        "total_questions": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": round(passed / total, 3) if total else 0,
        "failed_cases": failed_cases,
    }

    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
