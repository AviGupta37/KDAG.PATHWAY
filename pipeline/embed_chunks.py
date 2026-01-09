   import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

CHUNK_DIR = Path("data/processed/chunks")
OUT_DIR = Path("data/processed/embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for file in CHUNK_DIR.glob("*.jsonl"):
    texts, meta = [], []

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            meta.append(obj)

    embeddings = MODEL.encode(texts, batch_size=16, show_progress_bar=True)

    np.savez_compressed(
        OUT_DIR / f"{file.stem}.npz",
        embeddings=embeddings,
        metadata=meta
    )

    print(f"Embedded {file.stem}")

