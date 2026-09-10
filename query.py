"""
TaskFlow RAG — Query Script
Embeds a question, retrieves the most relevant stored chunks from Qdrant,
and generates an answer via Groq that is grounded only in that retrieved context.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.environ["COHERE_API_KEY"]
QDRANT_URL = os.environ["QDRANT_URL"].rstrip("/")
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

COLLECTION_NAME = "knowledge-base"
TOP_K = 5

SYSTEM_PROMPT = """You are a support assistant that answers questions using ONLY the context provided below.

Rules:
- If the answer is fully or partially contained in the context, answer clearly and cite which document(s) it came from.
- If the context does not contain the answer, say exactly: "I don't have that information in the knowledge base."
- Never use outside knowledge or make assumptions beyond what is in the context.
- Keep answers concise and direct.
"""


def embed_question(question):
    """Embed the user's question using Cohere. Note: input_type differs from ingestion."""
    response = requests.post(
        "https://api.cohere.com/v1/embed",
        headers={
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "texts": [question],
            "model": "embed-english-v3.0",
            "input_type": "search_query",  # different from "search_document" used in ingestion
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def search_qdrant(query_vector, top_k=TOP_K):
    """Find the most similar stored chunks for this query vector."""
    response = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
        headers={
            "api-key": QDRANT_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "vector": query_vector,
            "limit": top_k,
            "with_payload": True,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["result"]


def build_context(search_results):
    """Combine retrieved chunks into a single labeled context block."""
    blocks = []
    for i, result in enumerate(search_results):
        source = result["payload"]["document_source"]
        text = result["payload"]["text"]
        score = result["score"]
        blocks.append(f"[Source: {source} | relevance: {score:.3f}]\n{text}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(question, context):
    """Send the question and retrieved context to Groq for a grounded answer."""
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openai/gpt-oss-20b",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n\n{context}\n\nQuestion: {question}",
                },
            ],
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def ask(question):
    """Full pipeline: embed -> retrieve -> generate."""
    print(f"\nQuestion: {question}")

    query_vector = embed_question(question)
    search_results = search_qdrant(query_vector)

    if not search_results:
        print("No relevant chunks found in the knowledge base.")
        return

    print(f"Retrieved {len(search_results)} chunks:")
    for r in search_results:
        print(f"  - {r['payload']['document_source']} (score: {r['score']:.3f})")

    context = build_context(search_results)
    answer = generate_answer(question, context)

    print(f"\nAnswer: {answer}")
    return answer


if __name__ == "__main__":
    ask("How many automations can I run and how do I get more?")
    ask("Does TaskFlow support Portuguese language?")