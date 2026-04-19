import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "knowledge_base"
TERMS_FILE = ROOT / "terms_map.json"

with open(TERMS_FILE, "r", encoding="utf-8") as f:
    terms_map = json.load(f)

original_terms = sorted(terms_map.keys(), key=len, reverse=True)
violations = []

for path in KB_DIR.glob("*.md"):
    text = path.read_text(encoding="utf-8")
    hits = [term for term in original_terms if term in text]
    if hits:
        violations.append((path.name, hits[:10]))

if not violations:
    print("OK: no original terms found in knowledge_base.")
else:
    print("Found remaining original terms:")
    for filename, hits in violations:
        print(filename, "->", hits)
