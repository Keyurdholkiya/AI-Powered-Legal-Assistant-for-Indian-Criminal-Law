import os
import faiss
import pickle
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from django.conf import settings
print("🔥 USING NEW LIGHTWEIGHT RAG")


BASE_DIR = settings.BASE_DIR

SC_INDEX_PATH = os.path.join(BASE_DIR, "sc_judgments_faiss.index")
SC_DOCS_PATH = os.path.join(BASE_DIR, "sc_judgments_texts.pkl")

LAW_INDEX_PATH = os.path.join(BASE_DIR, "indian_law_faiss.index")
LAW_DOCS_PATH = os.path.join(BASE_DIR, "indian_law_texts.pkl")


embedder = None
sc_index = None
sc_documents = None
law_index = None
law_documents = None


def load_models():
    global embedder, sc_index, sc_documents, law_index, law_documents

    if embedder is None:
        print("Loading embedding model...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if sc_index is None:
        print("Loading SC FAISS index...")
        sc_index = faiss.read_index(SC_INDEX_PATH)
        with open(SC_DOCS_PATH, "rb") as f:
            sc_documents = pickle.load(f)

    if law_index is None:
        print("Loading Law FAISS index...")
        law_index = faiss.read_index(LAW_INDEX_PATH)
        with open(LAW_DOCS_PATH, "rb") as f:
            law_documents = pickle.load(f)


def retrieve_context(query, k=3):
    load_models()
    query_lower = query.lower()

    # 🔥 STEP 1 — Explicit Article Detection (Highest Priority)
    article_match = re.search(r"article\s+(\d+)", query_lower)
    if article_match:
        article_number = article_match.group(1)

        for doc in law_documents:
            if f"Article {article_number}" in doc:
                return {
                    "type": "bare_act",
                    "content": doc.strip()
                }

    # 🔥 STEP 2 — Explicit Section Detection
    section_match = re.search(r"section\s+(\d+)", query_lower)
    if section_match:
        section_number = section_match.group(1)

        # IPC priority
        if "ipc" in query_lower:
            for doc in law_documents:
                if "Act: IPC" in doc and f"Section {section_number}" in doc:
                    return {
                        "type": "bare_act",
                        "content": doc.strip()
                    }

        # Any section match
        for doc in law_documents:
            if f"Section {section_number}" in doc:
                return {
                    "type": "bare_act",
                    "content": doc.strip()
                }

    # 🔥 STEP 3 — Smart Number Detection (fallback only)
    number_match = re.search(r"\b(\d{1,4})\b", query_lower)
    if number_match:
        number = number_match.group(1)

        # Prefer Article if constitution mentioned
        if "constitution" in query_lower:
            for doc in law_documents:
                if f"Article {number}" in doc:
                    return {
                        "type": "bare_act",
                        "content": doc.strip()
                    }

        # Try Section
        for doc in law_documents:
            if f"Section {number}" in doc:
                return {
                    "type": "bare_act",
                    "content": doc.strip()
                }

    # 🔥 STEP 4 — Semantic Search Fallback
    query_vec = embedder.encode([query], normalize_embeddings=True)

    D1, I1 = sc_index.search(np.array(query_vec), k)
    D2, I2 = law_index.search(np.array(query_vec), k)

    sc_results = [sc_documents[idx] for idx in I1[0] if idx != -1]
    law_results = [law_documents[idx] for idx in I2[0] if idx != -1]

    context_parts = []

    for text in sc_results[:2]:
        context_parts.append(text[:600])

    for text in law_results[:2]:
        context_parts.append(text[:600])

    context = "\n\n".join(context_parts)

    if not context:
        return ""

    return {
        "type": "semantic",
        "content": context
    }