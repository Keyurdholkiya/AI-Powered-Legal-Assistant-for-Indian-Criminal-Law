import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# 1️⃣ Load FAISS index + texts

print("Loading FAISS index...")
index = faiss.read_index(
    r"C:\Users\Lenovo\OneDrive\Desktop\6TH SEM\law_agent-main\law_agent-main\ipc-vector-db\scripts\indian_law_faiss.index"
)

with open(
    r"C:\Users\Lenovo\OneDrive\Desktop\6TH SEM\law_agent-main\law_agent-main\ipc-vector-db\scripts\indian_law_texts.pkl",
    "rb"
) as f:
    texts = pickle.load(f)


print("Total sections loaded:", len(texts))


# 2️⃣ Load embedding model


print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3️⃣ Query Loop

while True:
    query = input("\nAsk your legal question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    # Encode + normalize query
    query_embedding = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_embedding)

    # Search FAISS
    k = 3
    distances, indices = index.search(query_embedding, k)

    print("\n🔎 Top Results:\n")

    for i, idx in enumerate(indices[0]):
        print(f"Result {i+1}")
        print(texts[idx][:700])  # Print first 700 chars
        print("-" * 60)
