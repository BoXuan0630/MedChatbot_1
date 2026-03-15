# MedBot - Malaysian Medical RAG Chatbot

FastAPI backend for a Malaysian medical chatbot. Part of a group FYP project — this is one feature inside an Android app.

The chatbot answers medical questions in **English, Malay, and Manglish** using Malaysian MOH Clinical Practice Guidelines (CPGs) and the FUKKM Drug Formulary, powered by an advanced 7-technique RAG pipeline.

---

## Quick Start (Local)

### 1. Prerequisites

| Requirement | How to check | Install guide |
|-------------|-------------|---------------|
| **Python 3.11+** | `python --version` | [python.org](https://www.python.org/downloads/) |
| **PostgreSQL** | `psql --version` | [postgresql.org](https://www.postgresql.org/download/) |
| **Redis** (optional) | `redis-cli ping` | [redis.io](https://redis.io/download/) or skip (backend works without it) |

### 2. Create the PostgreSQL database

```bash
psql -U postgres
```

Then inside the psql shell:

```sql
CREATE DATABASE medbot;
\q
```

### 3. Set up the backend

```bash
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Seed parent chunks into database (one-time only)
python -m app.scripts.seed_parents
```

### 4. Configure `.env`

Edit `backend/.env` with your actual keys:

```env
# Pinecone (from app.pinecone.io)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=medical-knowledge

# Google (from aistudio.google.com)
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash

# MedGemma (leave empty until GPU VM is deployed)
MEDGEMMA_URL=

# PostgreSQL
DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost:5432/medbot

# Redis (skip if not installed — caching will be disabled)
REDIS_URL=redis://localhost:6379/0
```

### 5. Start the server

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

You should see:

```
INFO:     PostgreSQL connected, tables created
INFO:     Redis connected
INFO:     SentenceTransformer loaded
INFO:     CrossEncoder loaded
INFO:     BM25 index loaded
INFO:     Pinecone connected
INFO:     Gemini configured
INFO:     Startup complete!
INFO:     Uvicorn running on http://127.0.0.1:8000
```

> First startup takes 1-2 minutes to download ML models. Subsequent starts are faster.

### 6. Test it

Open your browser: **http://localhost:8000/docs** (Swagger UI)

Or use curl:

```bash
# Health check
curl http://localhost:8000/health

# Ask a medical question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the treatment for diabetes?"}'

# Try Malay
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Apakah rawatan untuk kencing manis?"}'

# Try the streaming endpoint
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is diabetes?"}'

# Submit feedback
curl -X POST http://localhost:8000/chat/feedback \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-123", "question": "What is diabetes?", "rating": "up"}'
```

### 7. Run tests

```bash
cd backend
python -m pytest tests/ -v
```

---

## How It Works

```
User Question (en/ms/manglish)
    |
    v
[Intent Classification]
    |
    ├── Emergency ("chest pain right now") --> 999 emergency message
    ├── Greeting ("hi", "hello") --> Friendly greeting response
    ├── Off-topic ("what's the weather") --> Redirect to medical topics
    └── Medical question --> continues below
        |
        v
    [Language Detection] --> [Translation to English] (if Malay/Manglish)
        |
        v
    [Query Reformulation] (uses chat history for follow-ups)
        |
        v
    [Hybrid Search] = Dense (Pinecone) + BM25 (local) --> RRF Fusion
        |
        v
    [CrossEncoder Reranking] --> Top 5 chunks scored
        |
        v
    [Threshold Check] >= 0.70?
       / \
      Yes  No
      |     |
      v     v
    Grounded Mode    Knowledge Mode
    (CPG context +   (General medical
     citations)       knowledge + disclaimer)
        |                 |
        v                 v
    [Refusal Detection] -- if LLM says "not found" --> auto-fallback to Knowledge Mode
        |
        v
    [Gemini Flash] --> Answer + Follow-up suggestions
        |
        v
    [Save to DB] + [Cache in Redis] + [Log Q&A]
```

### RAG Techniques (7)

| # | Technique | Purpose |
|---|-----------|---------|
| 1 | **Parent-Child Chunking** | Small chunks (800 chars) for precise matching, expand to parents (3000 chars) for rich context |
| 2 | **Hybrid Search** | Dense (S-PubMedBert) + BM25 (keyword) cover each other's weaknesses |
| 3 | **Reciprocal Rank Fusion** | Score-agnostic merging of dense + BM25 results |
| 4 | **Cross-Encoder Reranking** | Accurate relevance scoring (query + document seen together) |
| 5 | **Threshold-Based Modes** | Score >= 0.70 → grounded with citations; below → knowledge with disclaimer |
| 6 | **Query Reformulation** | Rewrites follow-up questions as standalone using chat history |
| 7 | **Multilingual Translation** | Malay/Manglish → English before retrieval, answer in user's language |

### Answer Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| **Emergency** | "chest pain right now", "can't breathe", etc. | Returns 999 emergency number |
| **Greeting** | "hi", "hello", "selamat pagi", etc. | Friendly welcome message |
| **Off-topic** | Non-medical short queries | Redirects to medical questions |
| **Grounded** | Top rerank score >= 0.70 | Answer cites CPG/drug sources |
| **Knowledge** | Top rerank score < 0.70, or grounded refusal | General medical answer + disclaimer |

### Safety Features

- **Emergency detection** — queries about chest pain, breathing difficulty, overdose, suicidal ideation trigger immediate emergency response with Malaysian 999 number
- **Dosage change warnings** — "should I stop taking..." appends "Always consult your doctor before changing medication"
- **Grounded fallback** — if LLM says "context does not contain..." it auto-regenerates in knowledge mode instead of giving a useless answer
- **Input validation** — questions capped at 1000 characters, session IDs at 100 characters

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server status + DB/Redis connectivity |
| `POST` | `/chat` | Ask a medical question (full response) |
| `POST` | `/chat/stream` | Ask a medical question (SSE streaming) |
| `GET` | `/chat/history/{session_id}` | Get chat history for a session |
| `POST` | `/chat/feedback` | Submit thumbs up/down rating |
| `POST` | `/evaluate` | Run evaluation with test cases |

### POST /chat

**Request:**
```json
{
  "question": "What is the treatment for diabetes?",
  "session_id": "",
  "use_reranking": true
}
```

- `question` (required, max 1000 chars) — the user's question
- `session_id` (optional, max 100 chars) — send back the `session_id` from a previous response to maintain conversation context
- `use_reranking` (optional, default `true`) — enable/disable CrossEncoder reranking

**Response:**
```json
{
  "question": "What is the treatment for diabetes?",
  "answer": "According to the CPG on Management of Type 2 Diabetes...",
  "sources": [
    {"title": "CPG_T2DM_6th_Edition_2020", "source": "MOH_Malaysia_CPG"}
  ],
  "detected_lang": "en",
  "translated_query": "",
  "answer_mode": "grounded",
  "model_used": "gemini",
  "context_found": true,
  "top_score": 0.84,
  "session_id": "abc-123",
  "cached": false,
  "follow_ups": [
    "What are the side effects of metformin?",
    "What lifestyle changes help manage diabetes?"
  ]
}
```

### POST /chat/stream

Same request body as `/chat`. Returns Server-Sent Events:

```
data: {"token": "According to ", "done": false}
data: {"token": "the CPG on ", "done": false}
data: {"token": "Management...", "done": false}
data: {"token": "", "done": true, "answer_mode": "grounded", "sources": [...], "session_id": "abc-123"}
```

### POST /chat/feedback

```json
{
  "session_id": "abc-123",
  "question": "What is the treatment for diabetes?",
  "rating": "up",
  "comment": ""
}
```

- `rating` — `"up"` or `"down"`
- `comment` (optional) — free-text feedback

---

## Project Structure

```
backend/
├── .env
├── requirements.txt
├── Dockerfile
├── bm25_index.pkl               ← pre-built BM25 index (from Colab)
├── parents.json                  ← seed file (imported to DB once)
├── alembic.ini
├── alembic/
│   └── versions/
├── evaluation/
│   ├── test_cases.json           ← 25 test cases
│   ├── run_eval.py               ← evaluation runner
│   └── results.json              ← evaluation output
├── tests/
│   └── test_query_processor.py   ← 33 unit tests
└── app/
    ├── main.py                   ← FastAPI app + all endpoints
    ├── config.py                 ← environment settings
    ├── database.py               ← PostgreSQL + Redis setup
    ├── models/
    │   ├── schemas.py            ← Pydantic request/response models
    │   └── db_models.py          ← SQLAlchemy ORM (4 tables)
    ├── retrieval/
    │   ├── embedder.py           ← S-PubMedBert query embedding
    │   ├── query_processor.py    ← intent, language, translation, safety
    │   ├── hybrid_search.py      ← dense + BM25 + RRF fusion
    │   ├── reranker.py           ← CrossEncoder reranking
    │   └── retriever.py          ← full retrieval pipeline
    ├── generation/
    │   └── llm.py                ← answer generation + refusal detection + follow-ups
    ├── services/
    │   ├── cache_service.py      ← Redis get/set helpers
    │   ├── chat_service.py       ← save/load chat history
    │   └── log_service.py        ← log Q&A to qa_logs
    └── scripts/
        └── seed_parents.py       ← one-time: load parents.json → PostgreSQL
```

## Database Tables

| Table | Purpose |
|-------|---------|
| `parent_chunks` | Full parent text for context expansion (seeded from parents.json) |
| `chat_history` | Conversation messages per session_id |
| `qa_logs` | Every Q&A + metadata for analytics |
| `feedback` | User thumbs up/down ratings |

---

## Data Sources

- **MOH Malaysia CPGs** — ~53 clinical guideline PDFs (pre-embedded in Pinecone)
- **FUKKM Drug Formulary** — 1,675 drug entries (pre-embedded in Pinecone)
- **BM25 Index** — `bm25_index.pkl` built from Colab (loaded at startup)

All data was scraped, chunked, and embedded in a separate Colab notebook. The backend only **reads** from Pinecone — it never writes, embeds, or upserts.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| API | FastAPI (async) |
| Vector DB | Pinecone (read-only) |
| Database | PostgreSQL (asyncpg + SQLAlchemy) |
| Cache | Redis (redis.asyncio) |
| Embedding | S-PubMedBert-MS-MARCO (768 dims) |
| Reranking | CrossEncoder ms-marco-MiniLM-L-6-v2 |
| LLM | Gemini 2.5 Flash / MedGemma 4B (planned) |
| Streaming | Server-Sent Events (SSE) |
| Testing | pytest (33 tests) |

---

## Evaluation

Run the evaluation suite (25 test cases) against a running server:

```bash
cd backend
python evaluation/run_eval.py --url http://localhost:8000
```

Metrics computed:
- **Retrieval**: Precision@5, Recall@5, MRR, NDCG@5
- **Answer quality**: ROUGE-L, BERTScore F1
- **System**: avg/P90 latency, context hit rate, fallback rate

Test case categories: CPG grounded (EN), drug queries (EN), Malay, Manglish, knowledge mode, follow-ups, edge cases.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `models/gemini-2.5-flash... is not found` | Run `pip install --upgrade google-generativeai` |
| `Redis connection failed` | Redis is optional — backend works without it (caching disabled) |
| `bm25_index.pkl not found` | Make sure you're running from the `backend/` directory |
| `Sorry, I cannot answer this question` | Check your `GEMINI_API_KEY` in `.env` is valid |
| Slow first startup | Normal — ML models download on first run (~1-2 min) |
| `Connection refused` on PostgreSQL | Make sure PostgreSQL is running and `medbot` database exists |
| Tests fail with import errors | Run tests from `backend/`: `cd backend && python -m pytest tests/ -v` |

---

## Documentation

| Document | Description |
|----------|-------------|
| [RAG.md](docs/RAG.md) | Complete RAG technical reference — all 7 techniques explained |
| [MODELS.md](docs/MODELS.md) | ML model details, prompts, and API contracts |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Request flow diagrams and database schema |
| [EVALUATION.md](docs/EVALUATION.md) | Evaluation metrics and methodology |
| [API_SPEC.md](docs/API_SPEC.md) | Full endpoint specifications |
| [FRONTEND_INTEGRATION.md](docs/FRONTEND_INTEGRATION.md) | Android app integration guide |
| [PRD.md](docs/PRD.md) | Product requirements |
