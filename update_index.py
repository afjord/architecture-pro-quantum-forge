import hashlib
import json
import logging
import time
from datetime import datetime
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
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = OUT_DIR / "manifest.json"
LOG_PATH = LOG_DIR / "update_index.log"

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_manifest():
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(manifest: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_current_manifest():
    manifest = {}
    for path in list_docs(KB_DIR):
        rel_path = str(path.relative_to(PROJECT_ROOT))
        manifest[rel_path] = sha256_file(path)
    return manifest


def main():
    started_ts = time.time()
    started_at = datetime.now().isoformat()

    logging.info("update_index started at %s", started_at)

    try:
        old_manifest = load_manifest()
        new_manifest = build_current_manifest()

        changed_files = []
        for path_str, digest in new_manifest.items():
            if old_manifest.get(path_str) != digest:
                changed_files.append(path_str)

        deleted_files = [path_str for path_str in old_manifest if path_str not in new_manifest]

        if not changed_files and not deleted_files:
            logging.info("No changes detected. Index update skipped.")
            print("No changes detected. Index update skipped.")
            return

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
            "changed_files_count": len(changed_files),
            "deleted_files_count": len(deleted_files),
            **emb_meta,
            "generation_time_seconds": round(time.time() - started_ts, 3),
            "updated_at": started_at,
        }

        (OUT_DIR / "index_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        save_manifest(new_manifest)

        logging.info(
            "index updated at %s, %d files changed, %d files deleted, %d chunks, 0 errors",
            started_at,
            len(changed_files),
            len(deleted_files),
            len(chunks),
        )

        print(json.dumps(summary, ensure_ascii=False, indent=2))

    except Exception as e:
        logging.exception("Index update failed: %s", e)
        raise


if __name__ == "__main__":
    main()