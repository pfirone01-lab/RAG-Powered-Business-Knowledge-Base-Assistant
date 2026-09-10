"""
TaskFlow RAG — Ingestion Script
Reads .txt documents, splits them into overlapping chunks,
embeds each chunk via Cohere, and stores the result in Qdrant.
"""

import os
import glob
import uuid
import requests
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.environ["COHERE_API_KEY"]
QDRANT_URL = os.environ["QDRANT_URL"].rstrip("/")
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

COLLECTION_NAME = "knowledge-base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DOCUMENTS_DIR = "documents"


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks of roughly chunk_size characters."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap

    # Drop any empty or trivially short chunks (e.g. leftover whitespace)
    return [c for c in chunks if len(c) > 20]


def get_embedding(text):
    """Call Cohere's embed endpoint and return a single embedding vector."""
    response = requests.post(
        "https://api.cohere.com/v1/embed",
        headers={
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "texts": [text],
            "model": "embed-english-v3.0",
            "input_type": "search_document",
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["embeddings"][0]


def upsert_point(vector_id, embedding, chunk_text_value, source_document):
    """Store one embedding + its metadata in Qdrant."""
    response = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
        headers={
            "api-key": QDRANT_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "points": [
                {
                    "id": vector_id,
                    "vector": embedding,
                    "payload": {
                        "text": chunk_text_value,
                        "document_source": source_document,
                    },
                }
            ]
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def ingest_document(filepath):
    """Read, chunk, embed, and store a single document."""
    filename = os.path.basename(filepath)
    print(f"\nProcessing: {filename}")

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    print(f"  Split into {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk)
            point_id = str(uuid.uuid4())
            upsert_point(point_id, embedding, chunk, filename)
            print(f"  Chunk {i + 1}/{len(chunks)} stored (id: {point_id[:8]}...)")
        except requests.exceptions.RequestException as e:
            print(f"  FAILED on chunk {i + 1}: {e}")


def main():
    txt_files = glob.glob(os.path.join(DOCUMENTS_DIR, "*.txt"))

    if not txt_files:
        print(f"No .txt files found in '{DOCUMENTS_DIR}/'. Check your folder path.")
        return

    print(f"Found {len(txt_files)} document(s) to ingest.")

    for filepath in txt_files:
        ingest_document(filepath)

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
