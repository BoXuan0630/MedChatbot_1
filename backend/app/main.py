import asyncio
import logging
import pickle
import time
import uuid
from contextlib import asynccontextmanager

import google.generativeai as genai
import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
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
    FeedbackRequest,
    HistoryMessage,
    Source,
)
from app.retrieval.query_processor import (
    classify_intent,
    detect_language,
    translate_query,
    reformulate_query,
    has_dosage_change_phrase,
    GREETING_RESPONSE_EN,
    GREETING_RESPONSE_MS,
    OFF_TOPIC_RESPONSE_EN,
    OFF_TOPIC_RESPONSE_MS,
    EMERGENCY_RESPONSE_EN,
    EMERGENCY_RESPONSE_MS,
    DOSAGE_WARNING_EN,
    DOSAGE_WARNING_MS,
)
from app.retrieval.retriever import retrieve
from app.generation.llm import generate_answer, generate_follow_ups, _build_file_context_prompt, _call_gemini
from app.models.db_models import Feedback
from app.services.cache_service import get_cached_answer, set_cached_answer
from app.services.chat_service import save_message, get_history, get_recent_history
from app.services.log_service import log_qa
from app.services.file_service import (
    validate_upload,
    pdf_to_images,
    extract_file_content,
    save_file_context,
    get_latest_file_context,
    get_file_list,
)

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

    if intent in ("greeting", "off_topic", "emergency"):
        if intent == "greeting":
            answer = GREETING_RESPONSE_MS if detected_lang == "ms" else GREETING_RESPONSE_EN
        elif intent == "emergency":
            answer = EMERGENCY_RESPONSE_MS if detected_lang == "ms" else EMERGENCY_RESPONSE_EN
        else:
            answer = OFF_TOPIC_RESPONSE_MS if detected_lang == "ms" else OFF_TOPIC_RESPONSE_EN

        answer_mode = intent
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

    # Step 0b: Check if session has file context (for follow-up questions about uploaded files)
    if req.session_id:
        try:
            async with app.state.session_factory() as db_session:
                file_context = await get_latest_file_context(req.session_id, db_session)
            if file_context:
                prompt = _build_file_context_prompt(req.question, file_context, detected_lang)
                try:
                    answer = await _call_gemini(prompt, app.state)
                except Exception as e:
                    logger.error("Gemini file context generation failed: %s", e)
                    answer = (
                        "Maaf, saya tidak dapat menjawab soalan ini sekarang. Sila cuba lagi."
                        if detected_lang == "ms"
                        else "Sorry, I cannot answer this question right now. Please try again."
                    )

                follow_ups = []
                try:
                    follow_ups = await asyncio.wait_for(
                        generate_follow_ups(req.question, answer, detected_lang, app.state),
                        timeout=5.0,
                    )
                except Exception:
                    pass

                latency_ms = (time.time() - start_time) * 1000
                response = ChatResponse(
                    question=req.question,
                    answer=answer,
                    sources=[],
                    detected_lang=detected_lang,
                    translated_query="",
                    answer_mode="file_summary",
                    model_used="gemini",
                    context_found=True,
                    top_score=0.0,
                    session_id=session_id,
                    cached=False,
                    follow_ups=follow_ups,
                )

                async def _file_side_effects():
                    try:
                        async with app.state.session_factory() as bg_session:
                            await save_message(session_id, "user", req.question, bg_session, commit=False)
                            await save_message(session_id, "assistant", answer, bg_session, commit=False)
                            await log_qa(
                                session_id=session_id, question=req.question, answer=answer,
                                detected_lang=detected_lang, translated_query="",
                                answer_mode="file_summary", model_used="gemini",
                                context_found=True, top_score=0.0, sources=[],
                                latency_ms=latency_ms, db_session=bg_session, commit=False,
                            )
                            await bg_session.commit()
                    except Exception:
                        logger.exception("Failed to save file follow-up side effects")

                asyncio.create_task(_file_side_effects())
                return response
        except Exception:
            logger.warning("File context check failed, continuing with normal pipeline")

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

        # Step 9b: Append dosage safety warning if user asks about changing medication
        if has_dosage_change_phrase(req.question):
            warning = DOSAGE_WARNING_MS if detected_lang == "ms" else DOSAGE_WARNING_EN
            answer += warning

        # Build response
        answer_mode = "grounded" if retrieval_result["context_found"] else "knowledge"
        # If grounded fallback happened, switch answer_mode to knowledge
        if model_used == "gemini_fallback":
            answer_mode = "knowledge"
        sources = [
            Source(title=s["title"], source=s["source"])
            for s in retrieval_result["sources"]
        ]

        # Generate follow-up questions (non-blocking, with timeout)
        follow_ups = []
        try:
            follow_ups = await asyncio.wait_for(
                generate_follow_ups(clean_query, answer, detected_lang, app.state),
                timeout=5.0,
            )
        except Exception:
            logger.warning("Follow-up generation failed or timed out")

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
            follow_ups=follow_ups,
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


# ──────────────────────────────────────────────
# POST /chat/feedback
# ──────────────────────────────────────────────
@app.post("/chat/feedback")
async def chat_feedback(req: FeedbackRequest):
    try:
        async with app.state.session_factory() as db_session:
            entry = Feedback(
                session_id=req.session_id,
                question=req.question,
                rating=req.rating,
                comment=req.comment,
            )
            db_session.add(entry)
            await db_session.commit()
    except Exception:
        logger.exception("Failed to save feedback")
        raise HTTPException(status_code=500, detail="Failed to save feedback")
    return {"status": "ok"}


# ──────────────────────────────────────────────
# POST /chat/upload
# ──────────────────────────────────────────────
@app.post("/chat/upload", response_model=ChatResponse)
async def chat_upload(
    question: str = Form(...),
    session_id: str = Form(""),
    file: UploadFile = File(...),
):
    start_time = time.time()
    settings = app.state.settings

    # Validate question
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question must be 1000 characters or less")

    session_id = session_id.strip() or str(uuid.uuid4())
    detected_lang = detect_language(question)

    # Validate and process file
    file_bytes, file_ext = await validate_upload(file, settings)

    if file_ext == "pdf":
        images = pdf_to_images(file_bytes, settings.MAX_PDF_PAGES)
    else:
        from PIL import Image
        import io
        images = [Image.open(io.BytesIO(file_bytes))]

    page_count = len(images)

    # Extract text from document via Gemini Vision
    extracted_text = await extract_file_content(images, app.state)

    # Save file context to DB
    async with app.state.session_factory() as db_session:
        await save_file_context(
            session_id, file.filename or "uploaded_file", file_ext,
            extracted_text, page_count, db_session,
        )

    # Build prompt and generate answer
    prompt = _build_file_context_prompt(question, extracted_text, detected_lang)
    try:
        answer = await _call_gemini(prompt, app.state)
    except Exception as e:
        logger.error("Gemini generation failed for file upload: %s", e)
        answer = (
            "Maaf, saya tidak dapat menganalisis dokumen ini sekarang. Sila cuba lagi."
            if detected_lang == "ms"
            else "Sorry, I cannot analyze this document right now. Please try again."
        )

    # Generate follow-up questions
    follow_ups = []
    try:
        follow_ups = await asyncio.wait_for(
            generate_follow_ups(question, answer, detected_lang, app.state),
            timeout=5.0,
        )
    except Exception:
        logger.warning("Follow-up generation failed or timed out")

    latency_ms = (time.time() - start_time) * 1000

    response = ChatResponse(
        question=question,
        answer=answer,
        sources=[],
        detected_lang=detected_lang,
        translated_query="",
        answer_mode="file_summary",
        model_used="gemini",
        context_found=True,
        top_score=0.0,
        session_id=session_id,
        cached=False,
        follow_ups=follow_ups,
    )

    # Side effects — save chat history + log
    async def _side_effects():
        try:
            async with app.state.session_factory() as bg_session:
                await save_message(session_id, "user", f"[File: {file.filename}] {question}", bg_session, commit=False)
                await save_message(session_id, "assistant", answer, bg_session, commit=False)
                await log_qa(
                    session_id=session_id,
                    question=question,
                    answer=answer,
                    detected_lang=detected_lang,
                    translated_query="",
                    answer_mode="file_summary",
                    model_used="gemini",
                    context_found=True,
                    top_score=0.0,
                    sources=[],
                    latency_ms=latency_ms,
                    db_session=bg_session,
                    commit=False,
                )
                await bg_session.commit()
        except Exception:
            logger.exception("Failed to save file upload chat history / log")

    asyncio.create_task(_side_effects())
    return response


# ──────────────────────────────────────────────
# POST /chat/upload/stream  (Server-Sent Events)
# ──────────────────────────────────────────────
@app.post("/chat/upload/stream")
async def chat_upload_stream(
    question: str = Form(...),
    session_id: str = Form(""),
    file: UploadFile = File(...),
):
    import json as _json
    from starlette.responses import StreamingResponse

    start_time = time.time()
    settings = app.state.settings

    # Validate question
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question must be 1000 characters or less")

    session_id = session_id.strip() or str(uuid.uuid4())
    detected_lang = detect_language(question)

    # Validate and process file (before streaming starts)
    file_bytes, file_ext = await validate_upload(file, settings)

    if file_ext == "pdf":
        images = pdf_to_images(file_bytes, settings.MAX_PDF_PAGES)
    else:
        from PIL import Image
        import io
        images = [Image.open(io.BytesIO(file_bytes))]

    page_count = len(images)

    # Extract text from document via Gemini Vision
    extracted_text = await extract_file_content(images, app.state)

    # Save file context to DB
    async with app.state.session_factory() as db_session:
        await save_file_context(
            session_id, file.filename or "uploaded_file", file_ext,
            extracted_text, page_count, db_session,
        )

    # Stream the answer
    async def _stream():
        try:
            prompt = _build_file_context_prompt(question, extracted_text, detected_lang)

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: app.state.gemini.generate_content(prompt, stream=True),
            )

            full_answer = ""
            for chunk in response:
                if chunk.text:
                    full_answer += chunk.text
                    yield f"data: {_json.dumps({'token': chunk.text, 'done': False})}\n\n"

            # Final event
            yield f"data: {_json.dumps({'token': '', 'done': True, 'answer_mode': 'file_summary', 'sources': [], 'session_id': session_id, 'detected_lang': detected_lang, 'top_score': 0.0})}\n\n"

            # Side effects
            latency_ms = (time.time() - start_time) * 1000
            try:
                async with app.state.session_factory() as bg_session:
                    await save_message(session_id, "user", f"[File: {file.filename}] {question}", bg_session, commit=False)
                    await save_message(session_id, "assistant", full_answer, bg_session, commit=False)
                    await log_qa(
                        session_id=session_id,
                        question=question,
                        answer=full_answer,
                        detected_lang=detected_lang,
                        translated_query="",
                        answer_mode="file_summary",
                        model_used="gemini",
                        context_found=True,
                        top_score=0.0,
                        sources=[],
                        latency_ms=latency_ms,
                        db_session=bg_session,
                        commit=False,
                    )
                    await bg_session.commit()
            except Exception:
                logger.exception("Stream file upload side effects failed")

        except Exception as e:
            logger.exception("File upload streaming failed")
            yield f"data: {_json.dumps({'token': '', 'done': True, 'error': str(e)})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ──────────────────────────────────────────────
# GET /files/{session_id}
# ──────────────────────────────────────────────
@app.get("/files/{session_id}")
async def list_files(session_id: str):
    async with app.state.session_factory() as db_session:
        files = await get_file_list(session_id, db_session)
    return {
        "session_id": session_id,
        "files": [
            {
                "id": f.id,
                "filename": f.original_filename,
                "file_type": f.file_type,
                "page_count": f.page_count,
                "created_at": f.created_at.isoformat() if f.created_at else None,
            }
            for f in files
        ],
    }


# ──────────────────────────────────────────────
# POST /chat/stream  (Server-Sent Events)
# ──────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    import json as _json
    from starlette.responses import StreamingResponse

    start_time = time.time()
    settings = app.state.settings
    redis_client = app.state.redis

    session_id = req.session_id or str(uuid.uuid4())
    detected_lang = detect_language(req.question)
    intent = classify_intent(req.question)

    # Short-circuit intents — send as single SSE event
    if intent in ("greeting", "off_topic", "emergency"):
        if intent == "greeting":
            answer = GREETING_RESPONSE_MS if detected_lang == "ms" else GREETING_RESPONSE_EN
        elif intent == "emergency":
            answer = EMERGENCY_RESPONSE_MS if detected_lang == "ms" else EMERGENCY_RESPONSE_EN
        else:
            answer = OFF_TOPIC_RESPONSE_MS if detected_lang == "ms" else OFF_TOPIC_RESPONSE_EN

        async def _short_circuit():
            yield f"data: {_json.dumps({'token': answer, 'done': False})}\n\n"
            yield f"data: {_json.dumps({'token': '', 'done': True, 'answer_mode': intent, 'sources': [], 'session_id': session_id})}\n\n"

        return StreamingResponse(_short_circuit(), media_type="text/event-stream")

    # Check file context for follow-up questions
    file_context = None
    if req.session_id:
        try:
            async with app.state.session_factory() as db_session:
                file_context = await get_latest_file_context(req.session_id, db_session)
        except Exception:
            logger.warning("File context check failed in stream, continuing with RAG")

    if file_context:
        async def _file_stream():
            try:
                prompt = _build_file_context_prompt(req.question, file_context, detected_lang)
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: app.state.gemini.generate_content(prompt, stream=True),
                )

                full_answer = ""
                for chunk in response:
                    if chunk.text:
                        full_answer += chunk.text
                        yield f"data: {_json.dumps({'token': chunk.text, 'done': False})}\n\n"

                yield f"data: {_json.dumps({'token': '', 'done': True, 'answer_mode': 'file_summary', 'sources': [], 'session_id': session_id, 'detected_lang': detected_lang, 'top_score': 0.0})}\n\n"

                # Side effects
                latency_ms = (time.time() - start_time) * 1000
                try:
                    async with app.state.session_factory() as bg_session:
                        await save_message(session_id, "user", req.question, bg_session, commit=False)
                        await save_message(session_id, "assistant", full_answer, bg_session, commit=False)
                        await log_qa(
                            session_id=session_id, question=req.question, answer=full_answer,
                            detected_lang=detected_lang, translated_query="",
                            answer_mode="file_summary", model_used="gemini",
                            context_found=True, top_score=0.0, sources=[],
                            latency_ms=latency_ms, db_session=bg_session, commit=False,
                        )
                        await bg_session.commit()
                except Exception:
                    logger.exception("File stream side effects failed")
            except Exception as e:
                logger.exception("File context streaming failed")
                yield f"data: {_json.dumps({'token': '', 'done': True, 'error': str(e)})}\n\n"

        return StreamingResponse(_file_stream(), media_type="text/event-stream")

    # Full pipeline — retrieval is non-streamed, generation is streamed
    async def _stream():
        try:
            async with app.state.session_factory() as db_session:
                question = req.question
                if req.session_id:
                    history = await get_recent_history(session_id, db_session)
                    question = await reformulate_query(question, history, app.state)

                if detected_lang == "ms":
                    clean_query = await translate_query(
                        question, app.state, redis_client, settings
                    )
                else:
                    clean_query = question

                retrieval_result = await retrieve(
                    clean_query, app.state, settings, db_session,
                    use_reranking=req.use_reranking,
                )

            context_found = retrieval_result["context_found"]
            parents = retrieval_result.get("parents", {})

            context = ""
            if context_found and parents:
                context_parts = []
                for i, (pid, data) in enumerate(parents.items(), 1):
                    context_parts.append(f"[Source {i}: {data['title']}]\n{data['text']}")
                context = "\n\n".join(context_parts)

            # Build prompt
            from app.generation.llm import _build_grounded_prompt, _build_knowledge_prompt
            if context_found:
                prompt = _build_grounded_prompt(clean_query, context, detected_lang)
            else:
                prompt = _build_knowledge_prompt(clean_query, detected_lang)

            # Stream Gemini response
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: app.state.gemini.generate_content(prompt, stream=True),
            )

            full_answer = ""
            for chunk in response:
                if chunk.text:
                    full_answer += chunk.text
                    yield f"data: {_json.dumps({'token': chunk.text, 'done': False})}\n\n"

            # Append disclaimer for knowledge mode
            if not context_found:
                from app.generation.llm import DISCLAIMER_MS, DISCLAIMER_EN
                disclaimer = DISCLAIMER_MS if detected_lang == "ms" else DISCLAIMER_EN
                suffix = f"\n\n{disclaimer}"
                full_answer += suffix
                yield f"data: {_json.dumps({'token': suffix, 'done': False})}\n\n"

            # Dosage warning
            if has_dosage_change_phrase(req.question):
                warning = DOSAGE_WARNING_MS if detected_lang == "ms" else DOSAGE_WARNING_EN
                full_answer += warning
                yield f"data: {_json.dumps({'token': warning, 'done': False})}\n\n"

            answer_mode = "grounded" if context_found else "knowledge"
            sources = [
                {"title": s["title"], "source": s["source"]}
                for s in retrieval_result["sources"]
            ] if answer_mode == "grounded" else []

            # Final event with metadata
            yield f"data: {_json.dumps({'token': '', 'done': True, 'answer_mode': answer_mode, 'sources': sources, 'session_id': session_id, 'detected_lang': detected_lang, 'top_score': round(retrieval_result['top_score'], 4)})}\n\n"

            # Side effects
            latency_ms = (time.time() - start_time) * 1000
            try:
                async with app.state.session_factory() as bg_session:
                    await save_message(session_id, "user", req.question, bg_session, commit=False)
                    await save_message(session_id, "assistant", full_answer, bg_session, commit=False)
                    await log_qa(
                        session_id=session_id,
                        question=req.question,
                        answer=full_answer,
                        detected_lang=detected_lang,
                        translated_query=clean_query if detected_lang == "ms" else "",
                        answer_mode=answer_mode,
                        model_used="gemini",
                        context_found=context_found,
                        top_score=retrieval_result["top_score"],
                        sources=sources,
                        latency_ms=latency_ms,
                        db_session=bg_session,
                        commit=False,
                    )
                    await bg_session.commit()
            except Exception:
                logger.exception("Stream side effects failed")

        except Exception as e:
            logger.exception("Streaming generation failed")
            yield f"data: {_json.dumps({'token': '', 'done': True, 'error': str(e)})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")
