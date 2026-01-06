import pandas as pd
import json
from pathlib import Path
import re

# ---------- Load CSVs ----------
train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")

all_books = pd.concat([
    train["book_name"],
    test["book_name"]
]).unique()

# ---------- Load novel ----------
def load_novel(book_name):
    path = Path("data/raw/Books") / f"{book_name}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ---------- Chunking ----------
def split_into_sections(text):
    sections = re.split(r"\n{3,}", text)
    return [s.strip() for s in sections if len(s.strip()) > 500]

def chunk_section(section, max_chars=2000, overlap=300):
    chunks = []
    start = 0
    while start < len(section):
        end = start + max_chars
        chunks.append(section[start:end])
        start = end - overlap
    return chunks

def chunk_novel(book_name, text):
    sections = split_into_sections(text)
    chunks = []
    cid = 0

    for sid, sec in enumerate(sections):
        for c in chunk_section(sec):
            chunks.append({
                "book_name": book_name,
                "chunk_id": cid,
                "section_id": sid,
                "text": c
            })
            cid += 1

    total = len(chunks)
    for ch in chunks:
        ch["relative_position"] = ch["chunk_id"] / total

    return chunks

# ---------- Save ----------
def save_chunks(book_name, chunks):
    out = Path("data/processed/chunks")
    out.mkdir(parents=True, exist_ok=True)

    with open(out / f"{book_name}.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

# ---------- Main ----------
if __name__ == "__main__":
    stats = {}

    for book in all_books:
        print(f"Processing: {book}")
        text = load_novel(book)
        chunks = chunk_novel(book, text)
        save_chunks(book, chunks)

        stats[book] = {
            "chars": len(text),
            "chunks": len(chunks)
        }

    Path("data/processed/metadata").mkdir(parents=True, exist_ok=True)
    with open("data/processed/metadata/book_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("✅ Processing complete")
