import pandas as pd
from pathlib import Path

from retrieve_chunks import retrieve_chunks
from extract_claims import extract_claims
from verify_claims import verify_claim

EMBED_DIR = Path("data/processed/embeddings")

test = pd.read_csv("data/raw/test.csv")
predictions = []

for _, row in test.iterrows():
    story_id = row["story_id"]
    book = row["book_name"]
    backstory = row["backstory"]

    claims = extract_claims(backstory)
    verdicts = []

    emb_file = EMBED_DIR / f"{book}.npz"

    for claim in claims:
        evidence = retrieve_chunks(claim, emb_file)
        verdicts.append(verify_claim(claim, evidence))

    final = 0 if "contradicted" in verdicts else 1
    predictions.append({"story_id": story_id, "prediction": final})

pd.DataFrame(predictions).to_csv("results.csv", index=False)
print("✅ results.csv generated")

