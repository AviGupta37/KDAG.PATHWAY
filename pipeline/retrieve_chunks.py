import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_chunks(query, book_file, top_k=5):
    data = np.load(book_file, allow_pickle=True)
    embeddings = data["embeddings"]
    metadata = data["metadata"]

    q_emb = MODEL.encode([query])
    scores = cosine_similarity(q_emb, embeddings)[0]

    top_idx = scores.argsort()[-top_k:][::-1]

    return [metadata[i] for i in top_idx]

