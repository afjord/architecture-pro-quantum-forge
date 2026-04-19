import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "raw_texts"
OUTPUT_DIR = ROOT / "knowledge_base_generated"
TERMS_FILE = ROOT / "terms_map.json"

OUTPUT_DIR.mkdir(exist_ok=True)

with open(TERMS_FILE, "r", encoding="utf-8") as f:
    terms_map = json.load(f)

ordered_terms = sorted(terms_map.items(), key=lambda item: len(item[0]), reverse=True)

def replace_terms(text: str) -> str:
    for original, replacement in ordered_terms:
        pattern = re.compile(rf"\b{re.escape(original)}\b", flags=re.IGNORECASE)
        text = pattern.sub(replacement, text)
    return text

for file_path in INPUT_DIR.glob("*.md"):
    text = file_path.read_text(encoding="utf-8")
    transformed = replace_terms(text)
    out_path = OUTPUT_DIR / file_path.name
    out_path.write_text(transformed, encoding="utf-8")

print(f"Done. Generated files in: {OUTPUT_DIR}")
