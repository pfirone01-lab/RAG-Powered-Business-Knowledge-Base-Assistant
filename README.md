# TaskFlow Knowledge Base — RAG-Powered Support Assistant

A retrieval-augmented generation (RAG) system that answers questions about a fictional SaaS product (TaskFlow) using only the content of its own documentation — grounded, cited, and honest when the answer isn't in scope.

Built as Project 4 of a 5-project automation skill roadmap, focused on vector databases, semantic retrieval, and constraining an LLM to stay truthful.

---

## What it does

A user asks a question. The system:

1. Embeds the question into a 1024-dimensional vector (Cohere)
2. Searches a vector database for the 5 most semantically similar chunks of the source documents (Qdrant)
3. Feeds those chunks, plus the question, to an LLM with an instruction to answer only from that context (Groq)
4. Returns a grounded answer — or an explicit "I don't have that information" if the answer isn't in the documents

The same logic is implemented two ways: a Python script and a no-code n8n workflow, to demonstrate the pipeline at both the code level and the automation-platform level.

---

## Architecture

**Ingestion (one-time, run per document set):**
Source documents → chunked into ~500-character segments with 50-character overlap → each chunk embedded → stored in Qdrant with metadata (source filename)

**Query (every question):**
Question → embedded → Qdrant searched for top 5 similar chunks → chunks combined into a context block → sent to the LLM with a grounding system prompt → answer returned

Both phases share the same Qdrant collection (`knowledge-base`), which is what actually ties ingestion and retrieval together.

*(See the architecture diagram in the portfolio writeup / screenshots folder.)*

---

## Tech stack

| Component | Tool | Notes |
|---|---|---|
| Embeddings | Cohere (`embed-english-v3.0`) | 1024-dim vectors |
| Vector database | Qdrant Cloud | Free tier, cosine distance |
| Generation | Groq (`openai/gpt-oss-20b`) | Free, fast inference |
| Orchestration (script) | Python (`requests`, `python-dotenv`) | No heavy SDKs |
| Orchestration (no-code) | n8n | Webhook → embed → search → build context → generate → respond |
| Interface | Standalone HTML/JS | Dark-themed chat UI, calls the n8n webhook directly |

**Test corpus:** three fictional documents for "TaskFlow," a mock project-management SaaS product — an FAQ, a product manual, and a pricing sheet. The documents deliberately overlap (e.g. automation limits appear in both the manual and pricing doc) to test cross-document retrieval, and each contains information unique to it, to test retrieval precision.

---

## Why these specific tools

The original project plan called for a fully local stack (Ollama for both embeddings and generation). That was swapped for a fully hosted stack for two practical reasons:

- **Ollama's install and model downloads run 6-7GB combined**, which isn't worth it on a constrained connection when the entire point of the exercise is the retrieval architecture, not where inference happens.
- **Hugging Face's free serverless Inference API**, originally planned as the embeddings provider, has been restructured into "Inference Providers" and no longer reliably serves raw HTTP calls to the old-style embedding endpoint — it's now oriented around dedicated client libraries.

Cohere was chosen as the replacement for embeddings because it has a stable, dedicated REST endpoint that works predictably from plain HTTP calls — important for both the Python script and n8n's HTTP Request node.

Worth noting for anyone rebuilding this: Groq deprecated the originally-planned `llama-3.1-8b-instant` model shortly before this build (June 2026), which surfaced as a live 404 error mid-project and was resolved by switching to `openai/gpt-oss-20b`. Model names on fast-moving LLM APIs are not a "set once" detail — checking `GET /v1/models` before hardcoding a model name is worth doing on any new build.

---

## Grounding behavior (the actual point of RAG)

The system prompt instructs the model to answer only from retrieved context, and to say plainly when it doesn't know. Validated with three test questions:

| Question type | Example | Result |
|---|---|---|
| Single-document | "What's the file upload limit on the Growth plan?" | Correct answer (25MB), cited source |
| Cross-document | "How many automations can I run and how do I get more?" | Correctly combined the Product Manual's automation limits with Pricing's upgrade tiers |
| Out-of-scope | "Does TaskFlow support Portuguese language?" | Correctly refused: "I don't have that information in the knowledge base" |

The out-of-scope case is the one that actually proves this is RAG and not a wrapper that always sounds confident — the retrieved chunks for that question scored noticeably lower (0.44–0.49) than the in-scope questions' top matches (0.56+), and the model declined rather than filling the gap with outside knowledge.

---

## Repository contents

```
taskflow-rag/
├── documents/
│   ├── taskflow_faq.txt
│   ├── taskflow_product_manual.txt
│   └── taskflow_pricing.txt
├── ingest.py              # chunks, embeds, and stores documents in Qdrant
├── query.py               # embeds a question, retrieves context, generates a grounded answer
├── taskflow-chat.html     # standalone dark-themed chat interface (calls the n8n webhook)
├── .env                   # API keys (not committed — see setup below)
└── README.md
```

The n8n workflow itself is exported separately as JSON (see `n8n-workflow-export.json` if included, or rebuild from the node list below).

---

## Setup

### 1. Environment variables

Create a `.env` file in the project root:

```
COHERE_API_KEY=your_cohere_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_key
GROQ_API_KEY=your_groq_key
```

### 2. Python environment

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
.\venv\Scripts\Activate.ps1   # Windows PowerShell

pip install requests python-dotenv
```

### 3. Ingest the documents

```bash
python ingest.py
```

Reads every `.txt` file in `documents/`, chunks it, embeds each chunk via Cohere, and stores it in the `knowledge-base` Qdrant collection.

### 4. Query it

```bash
python query.py
```

Edit the `ask(...)` call at the bottom of the file to try different questions.

### 5. n8n workflow (optional, mirrors the Python logic)

Nodes: **Webhook → Embed the question (Cohere) → Search Qdrant → Build the context block (Code node) → Generate answer (AI Agent, Groq model) → Respond to Webhook**.

Publish the workflow to get a permanent production webhook URL, then paste it into `taskflow-chat.html`'s `N8N_WEBHOOK_URL` constant.

### 6. Chat interface

Open `taskflow-chat.html` directly in a browser, or host it via GitHub Pages. Requires the n8n webhook URL to be set and the workflow to be published (active), not just running in test mode.

---

## What this project demonstrates

- Vector database setup and configuration (collection schema, distance metric, dimensionality matching across providers)
- A full document ingestion pipeline: chunking strategy, embedding, metadata-tagged storage
- Semantic retrieval: finding relevant content by meaning, not keyword matching
- Prompt-based grounding: constraining an LLM to avoid hallucination and cite when it doesn't know
- The same architecture built two ways — script and no-code workflow — proving the underlying mechanics are understood at the platform-agnostic level
- Adapting mid-build to real provider changes (a model deprecation, an API restructuring) without derailing the project
