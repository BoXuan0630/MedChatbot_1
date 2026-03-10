import asyncio
import logging
import pickle
import time
import uuid
from contextlib import asynccontextmanager

import google.generativeai as genai
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder
from sqlalchemy import text as sa_text

from app.config import Settings
from app.database import create_engine, create_session_factory, create_redis_client
from app.models.db_models import Base
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ChatHistoryResponse,
    EvaluationRequest,
    HistoryMessage,
    Source,
)
from app.retrieval.query_processor import (
    classify_intent,
    detect_language,
    translate_query,
    reformulate_query,
    GREETING_RESPONSE_EN,
    GREETING_RESPONSE_MS,
    OFF_TOPIC_RESPONSE_EN,
    OFF_TOPIC_RESPONSE_MS,
)
from app.retrieval.retriever import retrieve
from app.generation.llm import generate_answer
from app.services.cache_service import get_cached_answer, set_cached_answer
from app.services.chat_service import save_message, get_history, get_recent_history
from app.services.log_service import log_qa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and connections at startup."""
    logger.info("Starting up MedBot API...")

    # 1. Settings
    settings = Settings()
    app.state.settings = settings

    # 2. PostgreSQL
    engine = create_engine(settings)
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("PostgreSQL connected, tables created")

    # 3. Redis
    app.state.redis = create_redis_client(settings)
    try:
        await app.state.redis.ping()
        logger.info("Redis connected")
    except Exception:
        logger.warning("Redis connection failed — caching will be disabled")

    # 4. SentenceTransformer (S-PubMedBert)
    logger.info("Loading SentenceTransformer...")
    app.state.embedder = SentenceTransformer(
        "pritamdeka/S-PubMedBert-MS-MARCO", device="cpu"
    )
    logger.info("SentenceTransformer loaded")

    # 5. CrossEncoder (ms-marco-MiniLM)
    logger.info("Loading CrossEncoder...")
    app.state.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    logger.info("CrossEncoder loaded")

    # 6. BM25 index
    logger.info("Loading BM25 index...")
    with open("bm25_index.pkl", "rb") as f:
        app.state.bm25_data = pickle.load(f)
    logger.info("BM25 index loaded")

    # 7. Pinecone
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    app.state.pinecone_index = pc.Index(settings.PINECONE_INDEX)
    logger.info("Pinecone connected")

    # 8. Gemini
    genai.configure(api_key=settings.GEMINI_API_KEY)
    app.state.gemini = genai.GenerativeModel(settings.GEMINI_MODEL)
    logger.info("Gemini configured")

    # 9. Persistent HTTP client for MedGemma (reuse connections, no per-request overhead)
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(float(settings.MEDGEMMA_TIMEOUT)),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    )
    logger.info("HTTP client created")

    # 10. Keepalive ping — prevents Kaggle/localtunnel from going idle
    keepalive_task = None
    if settings.MEDGEMMA_URL:
        async def _keepalive():
            url = settings.MEDGEMMA_URL.rstrip("/") + "/warmup"
            while True:
                await asyncio.sleep(240)  # ping every 4 minutes
                try:
                    r = await app.state.http_client.get(
                        url, headers={"bypass-tunnel-reminder": "true"}
                    )
                    logger.info("Keepalive ping: %s", r.status_code)
                except Exception as e:
                    logger.warning("Keepalive ping failed: %s", e)

        keepalive_task = asyncio.create_task(_keepalive())
        logger.info("MedGemma keepalive task started (ping every 4 min)")

    logger.info("Startup complete!")
    yield

    # Shutdown
    if keepalive_task:
        keepalive_task.cancel()
    await app.state.http_client.aclose()
    await app.state.redis.close()
    await app.state.engine.dispose()
    logger.info("Shutdown complete")


app = FastAPI(title="MedBot API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────
@app.get("/health")
async def health():
    db_status = "connected"
    redis_status = "connected"

    try:
        async with app.state.session_factory() as session:
            await session.execute(sa_text("SELECT 1"))
    except Exception:
        db_status = "disconnected"

    try:
        await app.state.redis.ping()
    except Exception:
        redis_status = "disconnected"

    status = "healthy" if db_status == "connected" and redis_status == "connected" else "degraded"
    return {
        "status": status,
        "message": "MedBot API is running",
        "version": "1.0.0",
        "database": db_status,
        "redis": redis_status,
    }


# ──────────────────────────────────────────────
# POST /chat
# ──────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    start_time = time.time()
    settings = app.state.settings
    redis_client = app.state.redis

    # Generate session_id if not provided
    session_id = req.session_id or str(uuid.uuid4())

    # Step 0: Classify intent — handle greetings and off-topic before pipeline
    detected_lang = detect_language(req.question)
    intent = classify_intent(req.question)

    if intent in ("greeting", "off_topic"):
        if intent == "greeting":
            answer = GREETING_RESPONSE_MS if detected_lang == "ms" else GREETING_RESPONSE_EN
        else:
            answer = OFF_TOPIC_RESPONSE_MS if detected_lang == "ms" else OFF_TOPIC_RESPONSE_EN

        answer_mode = "greeting" if intent == "greeting" else "off_topic"
        response = ChatResponse(
            question=req.question,
            answer=answer,
            sources=[],
            detected_lang=detected_lang,
            translated_query="",
            answer_mode=answer_mode,
            model_used="none",
            context_found=False,
            top_score=0.0,
            session_id=session_id,
            cached=False,
        )

        # Save to chat_history and qa_logs
        try:
            async with app.state.session_factory() as db_session:
                await save_message(session_id, "user", req.question, db_session, commit=False)
                await save_message(session_id, "assistant", answer, db_session, commit=False)
                await log_qa(
                    session_id=session_id,
                    question=req.question,
                    answer=answer,
                    detected_lang=detected_lang,
                    translated_query="",
                    answer_mode=answer_mode,
                    model_used="none",
                    context_found=False,
                    top_score=0.0,
                    sources=[],
                    latency_ms=(time.time() - start_time) * 1000,
                    db_session=db_session,
                    commit=False,
                )
                await db_session.commit()
        except Exception:
            logger.warning("Failed to save greeting/off-topic to chat history / log")

        return response

    # Step 1: Check answer cache
    try:
        cached = await get_cached_answer(req.question, redis_client)
        if cached:
            cached["cached"] = True
            cached["session_id"] = session_id
            return ChatResponse(**cached)
    except Exception:
        logger.warning("Redis cache check failed, continuing without cache")

    async with app.state.session_factory() as db_session:
        # Step 2: Reformulate using chat history (only for existing sessions)
        question = req.question
        if req.session_id:
            history = await get_recent_history(session_id, db_session)
            question = await reformulate_query(question, history, app.state)

        # Step 3: Detect language + translate
        if detected_lang == "ms":
            clean_query = await translate_query(
                question, app.state, redis_client, settings
            )
        else:
            clean_query = question
        translated_query = clean_query if detected_lang == "ms" else ""

        # Steps 4-8: Retrieve (embed → hybrid search → rerank → threshold → parents)
        retrieval_result = await retrieve(
            clean_query, app.state, settings, db_session, use_reranking=req.use_reranking
        )

        # Step 9: Generate answer
        answer, model_used = await generate_answer(
            clean_query, retrieval_result, detected_lang, app.state, settings
        )

        # Build response
        answer_mode = "grounded" if retrieval_result["context_found"] else "knowledge"
        sources = [
            Source(title=s["title"], source=s["source"])
            for s in retrieval_result["sources"]
        ]

        latency_ms = (time.time() - start_time) * 1000

        response = ChatResponse(
            question=req.question,
            answer=answer,
            sources=sources if answer_mode == "grounded" else [],
            detected_lang=detected_lang,
            translated_query=translated_query,
            answer_mode=answer_mode,
            model_used=model_used,
            context_found=retrieval_result["context_found"],
            top_score=round(retrieval_result["top_score"], 4),
            session_id=session_id,
            cached=False,
        )

        # Step 10: Side effects — fire in background so response returns immediately
        response_data = response.model_dump(mode="json")

        async def _side_effects():
            try:
                await set_cached_answer(
                    req.question, response_data, redis_client,
                    ttl=settings.REDIS_ANSWER_TTL,
                )
            except Exception:
                logger.warning("Failed to cache answer")
            try:
                async with app.state.session_factory() as bg_session:
                    await save_message(session_id, "user", req.question, bg_session, commit=False)
                    await save_message(session_id, "assistant", answer, bg_session, commit=False)
                    await log_qa(
                        session_id=session_id,
                        question=req.question,
                        answer=answer,
                        detected_lang=detected_lang,
                        translated_query=translated_query,
                        answer_mode=answer_mode,
                        model_used=model_used,
                        context_found=retrieval_result["context_found"],
                        top_score=retrieval_result["top_score"],
                        sources=[s.model_dump() for s in sources],
                        latency_ms=latency_ms,
                        db_session=bg_session,
                        commit=False,
                    )
                    await bg_session.commit()
            except Exception:
                logger.exception("Failed to save chat history / log Q&A")

        asyncio.create_task(_side_effects())

    return response


# ──────────────────────────────────────────────
# GET /chat/history/{session_id}
# ──────────────────────────────────────────────
@app.get("/chat/history/{session_id}", response_model=ChatHistoryResponse)
async def chat_history(session_id: str):
    async with app.state.session_factory() as db_session:
        messages = await get_history(session_id, db_session)

    return ChatHistoryResponse(
        session_id=session_id,
        messages=[
            HistoryMessage(
                role=m["role"],
                content=m["content"],
                timestamp=m["timestamp"],
            )
            for m in messages
        ],
    )


# ──────────────────────────────────────────────
# POST /evaluate
# ──────────────────────────────────────────────
@app.post("/evaluate")
async def evaluate(req: EvaluationRequest):
    from rouge_score import rouge_scorer

    settings = app.state.settings
    redis_client = app.state.redis

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    if not req.test_cases:
        return {
            "retrieval_metrics": {
                "precision_at_5": 0.0,
                "recall_at_5": 0.0,
                "mrr": 0.0,
                "ndcg_at_5": 0.0,
            },
            "answer_quality_metrics": {
                "rouge_l": 0.0,
                "bert_score_f1": 0.0,
            },
            "system_metrics": {
                "avg_latency_ms": 0.0,
                "p90_latency_ms": 0.0,
                "context_hit_rate": 0.0,
                "fallback_rate": 0.0,
                "cache_hit_rate": 0.0,
                "grounded_mode_count": 0,
                "knowledge_mode_count": 0,
            },
            "per_question": [],
        }

    per_question = []
    latencies = []
    generated_list = []
    reference_list = []
    context_found_count = 0
    gemini_fallback_count = 0

    for tc in req.test_cases:
        start_time = time.time()

        # Run the chat pipeline for this test case
        async with app.state.session_factory() as db_session:
            detected_lang = detect_language(tc.question)
            if detected_lang == "ms":
                clean_query = await translate_query(
                    tc.question, app.state, redis_client, settings
                )
            else:
                clean_query = tc.question

            retrieval_result = await retrieve(
                clean_query, app.state, settings, db_session,
                use_reranking=req.use_reranking,
            )

            answer, model_used = await generate_answer(
                clean_query, retrieval_result, detected_lang, app.state, settings
            )

        latency = (time.time() - start_time) * 1000
        latencies.append(latency)

        context_found = retrieval_result["context_found"]
        if context_found:
            context_found_count += 1
        if model_used == "gemini":
            gemini_fallback_count += 1

        # Retrieval metrics
        retrieved_titles = [
            c.get("title", "") for c in retrieval_result["top_chunks"]
        ]

        # Precision@5
        hits_p = sum(
            1 for t in retrieved_titles[:5]
            if any(rel.lower() in t.lower() for rel in tc.relevant_sources)
        )
        precision_at_5 = hits_p / 5

        # Recall@5
        if tc.relevant_sources:
            hits_r = sum(
                1 for rel in tc.relevant_sources
                if any(rel.lower() in t.lower() for t in retrieved_titles[:5])
            )
            recall_at_5 = hits_r / len(tc.relevant_sources)
        else:
            recall_at_5 = 1.0

        # MRR
        mrr = 0.0
        for rank, title in enumerate(retrieved_titles, 1):
            if any(rel.lower() in title.lower() for rel in tc.relevant_sources):
                mrr = 1.0 / rank
                break

        # NDCG@5
        gains = [
            1.0 if any(rel.lower() in t.lower() for rel in tc.relevant_sources) else 0.0
            for t in retrieved_titles[:5]
        ]
        dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
        idcg = sum(
            1.0 / np.log2(i + 2)
            for i in range(min(len(tc.relevant_sources), 5))
        )
        ndcg_at_5 = dcg / idcg if idcg > 0 else 0.0

        # ROUGE-L
        rouge_result = scorer.score(tc.ground_truth, answer)
        rouge_l = rouge_result["rougeL"].fmeasure

        generated_list.append(answer)
        reference_list.append(tc.ground_truth)

        per_question.append({
            "question": tc.question,
            "answer": answer,
            "detected_lang": detected_lang,
            "model_used": model_used,
            "context_found": context_found,
            "top_score": round(retrieval_result["top_score"], 4),
            "precision_at_5": round(precision_at_5, 4),
            "recall_at_5": round(recall_at_5, 4),
            "mrr": round(mrr, 4),
            "ndcg_at_5": round(ndcg_at_5, 4),
            "rouge_l": round(rouge_l, 4),
            "latency_ms": round(latency, 2),
        })

    # BERTScore (batch)
    try:
        from bert_score import score as bert_score_fn

        _, _, F1 = bert_score_fn(
            generated_list, reference_list,
            lang="en", model_type="distilbert-base-uncased", verbose=False,
        )
        bert_score_f1 = float(F1.mean())
    except Exception:
        logger.warning("BERTScore computation failed")
        bert_score_f1 = 0.0

    total = len(req.test_cases)
    return {
        "retrieval_metrics": {
            "precision_at_5": round(np.mean([q["precision_at_5"] for q in per_question]), 4),
            "recall_at_5": round(np.mean([q["recall_at_5"] for q in per_question]), 4),
            "mrr": round(np.mean([q["mrr"] for q in per_question]), 4),
            "ndcg_at_5": round(np.mean([q["ndcg_at_5"] for q in per_question]), 4),
        },
        "answer_quality_metrics": {
            "rouge_l": round(np.mean([q["rouge_l"] for q in per_question]), 4),
            "bert_score_f1": round(bert_score_f1, 4),
        },
        "system_metrics": {
            "avg_latency_ms": round(np.mean(latencies), 2),
            "p90_latency_ms": round(float(np.percentile(latencies, 90)), 2),
            "context_hit_rate": round(context_found_count / total, 4) if total else 0.0,
            "fallback_rate": round(gemini_fallback_count / total, 4) if total else 0.0,
            "cache_hit_rate": 0.0,  # Evaluate endpoint bypasses cache by design
            "grounded_mode_count": context_found_count,
            "knowledge_mode_count": total - context_found_count,
        },
        "per_question": per_question,
    }
