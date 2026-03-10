import asyncio
import logging

from sqlalchemy import select

from app.models.db_models import ParentChunk
from app.retrieval.embedder import embed_query
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.reranker import rerank

logger = logging.getLogger(__name__)


async def fetch_parents_from_db(parent_ids: list[str], db_session) -> dict:
    """Fetch parent chunk text from PostgreSQL."""
    result = await db_session.execute(
        select(ParentChunk).where(ParentChunk.parent_id.in_(parent_ids))
    )
    rows = result.scalars().all()
    return {
        row.parent_id: {
            "title": row.title,
            "source": row.source,
            "type": row.type,
            "text": row.text,
        }
        for row in rows
    }


def fetch_parents_from_pinecone(parent_ids: list[str], app_state) -> dict:
    """Fallback: fetch parent chunks from Pinecone parents namespace."""
    try:
        result = app_state.pinecone_index.fetch(ids=parent_ids, namespace="parents")
        return {
            pid: v["metadata"]
            for pid, v in result.get("vectors", {}).items()
        }
    except Exception:
        logger.exception("Pinecone parent fetch failed")
        return {}


async def retrieve(
    clean_query: str, app_state, settings, db_session, use_reranking: bool = True
) -> dict:
    """Full retrieval pipeline: embed → hybrid search → rerank → threshold → parent expansion.

    Returns dict with keys:
        top_chunks: list[dict] — top 5 reranked chunks
        context_found: bool
        top_score: float
        parents: dict — {parent_id: {title, source, type, text}}
        sources: list[dict] — [{title, source, text}] for response
    """
    # Step 1: Embed query
    loop = asyncio.get_running_loop()
    vector = await loop.run_in_executor(None, embed_query, clean_query, app_state)

    # Step 2: Hybrid search (dense + BM25 + RRF)
    candidates = await hybrid_search(clean_query, vector, app_state, settings)

    # Step 3: Rerank
    if use_reranking and candidates:
        top_chunks = await loop.run_in_executor(
            None, rerank, clean_query, candidates, app_state, settings.TOP_K_RERANK
        )
    else:
        top_chunks = candidates[: settings.TOP_K_RERANK]
        # Normalize RRF scores to [0, 1] so they're comparable to CONTEXT_THRESHOLD
        rrf_scores = [c.get("rrf_score", 0.0) for c in top_chunks]
        min_s = min(rrf_scores) if rrf_scores else 0.0
        max_s = max(rrf_scores) if rrf_scores else 0.0
        score_range = max_s - min_s
        for chunk in top_chunks:
            raw = chunk.get("rrf_score", 0.0)
            chunk["rerank_score"] = (raw - min_s) / score_range if score_range > 0 else 0.0

    # Step 4: Threshold check
    top_score = top_chunks[0]["rerank_score"] if top_chunks else 0.0
    context_found = top_score >= settings.CONTEXT_THRESHOLD

    # Step 5: Parent expansion (only if context found)
    parents = {}
    sources = []
    if context_found and top_chunks:
        parent_ids = list({
            c["parent_id"] for c in top_chunks if c.get("parent_id")
        })

        if parent_ids:
            # Primary: fetch from PostgreSQL
            try:
                parents = await fetch_parents_from_db(parent_ids, db_session)
            except Exception:
                logger.exception("DB parent fetch failed, falling back to Pinecone")
                parents = {}

            # Fallback: fetch missing parents from Pinecone
            missing = [pid for pid in parent_ids if pid not in parents]
            if missing:
                pinecone_parents = await loop.run_in_executor(
                    None, fetch_parents_from_pinecone, missing, app_state
                )
                parents.update(pinecone_parents)

        # Build sources list for response
        seen_titles = set()
        for chunk in top_chunks:
            pid = chunk.get("parent_id")
            if pid and pid in parents and parents[pid]["title"] not in seen_titles:
                seen_titles.add(parents[pid]["title"])
                sources.append({
                    "title": parents[pid]["title"],
                    "source": parents[pid]["source"],
                    "text": parents[pid]["text"],
                })

    return {
        "top_chunks": top_chunks,
        "context_found": context_found,
        "top_score": top_score,
        "parents": parents,
        "sources": sources,
    }
