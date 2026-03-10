# MedBot - Malaysian Medical Chatbot Backend

FastAPI backend for a Malaysian medical chatbot. Part of a group FYP project — this is one feature inside an Android app.

The chatbot answers medical questions in **English, Malay, and Manglish** using Malaysian MOH Clinical Practice Guidelines (CPGs) and the FUKKM Drug Formulary.

---

## Quick Start (Local)

### 1. Prerequisites

Make sure you have these installed and running:

| Requirement | How to check | Install guide |
|-------------|-------------|---------------|
| **Python 3.11+** | `python --version` | [python.org](https://www.python.org/downloads/) |
| **PostgreSQL** | `psql --version` | [postgresql.org](https://www.postgresql.org/download/) |
| **Redis** (optional) | `redis-cli ping` | [redis.io](https://redis.io/download/) or skip (backend works without it) |

### 2. Create the PostgreSQL database

Open a terminal and run:

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
INFO:     Loading SentenceTransformer...
INFO:     SentenceTransformer loaded
INFO:     Loading CrossEncoder...
INFO:     CrossEncoder loaded
INFO:     Loading BM25 index...
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

# Try a greeting
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "hi"}'

# Try Malay
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Apakah rawatan untuk kencing manis?"}'
```

---

## How It Works

```
User Question (en/ms/manglish)
    |
    v
[Intent Classification]
    |
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
        |
        v
    [Gemini Flash] --> Answer
        |
        v
    [Save to DB] + [Cache in Redis] + [Log Q&A]
```

## Answer Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| **Greeting** | "hi", "hello", "selamat pagi", etc. | Friendly welcome message |
| **Off-topic** | Non-medical short queries | Redirects user to ask medical questions |
| **Grounded** | Top rerank score >= 0.70 | Answer cites CPG sources |
| **Knowledge** | Top rerank score < 0.70 | General medical answer + disclaimer |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server status + DB/Redis connectivity |
| `POST` | `/chat` | Ask a medical question |
| `GET` | `/chat/history/{session_id}` | Get chat history for a session |
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

- `question` (required) — the user's question
- `session_id` (optional) — send back the `session_id` from a previous response to maintain conversation context
- `use_reranking` (optional, default `true`) — enable/disable CrossEncoder reranking

**Response:**
```json
{
  "question": "What is the treatment for diabetes?",
  "answer": "According to the CPG on Management of Type 2 Diabetes...",
  "sources": [
    {"title": "CPG_Management_of_Type_2_Diabetes", "source": "MOH_Malaysia_CPG"}
  ],
  "detected_lang": "en",
  "translated_query": "",
  "answer_mode": "grounded",
  "model_used": "gemini-2.5-flash",
  "context_found": true,
  "top_score": 0.84,
  "session_id": "abc-123",
  "cached": false
}
```

**What the Android app should display:**
- `answer` — the chatbot's reply (show this in the chat bubble)
- `sources` — reference titles (show below the answer for grounded mode)

**What the Android app should store internally:**
- `session_id` — send this back in the next request to maintain conversation
- Other fields are metadata for debugging/analytics

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
| API | FastAPI |
| Vector DB | Pinecone (read-only) |
| Database | PostgreSQL (asyncpg) |
| Cache | Redis |
| Embedding | S-PubMedBert-MS-MARCO (768 dims) |
| Reranking | CrossEncoder ms-marco-MiniLM-L-6-v2 |
| LLM | Gemini 2.5 Flash / MedGemma 4B |
---

## Evaluation

Run the evaluation suite against a running server:

```bash
cd backend
python evaluation/run_eval.py --url http://localhost:8000
```

Metrics: Precision@5, Recall@5, MRR, NDCG@5, ROUGE-L, BERTScore F1, latency.

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
