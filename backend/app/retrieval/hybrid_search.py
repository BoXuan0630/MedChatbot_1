import asyncio
import re
from collections import defaultdict

import numpy as np


# BM25 tokenizer — must match Colab exactly
STOP_WORDS = {
    "the", "a", "an", "is", "in", "on", "at", "to", "for", "of", "and", "or",
    "with", "that", "this", "are", "was", "were", "be", "been", "has", "have",
    "it", "as", "by", "from", "not", "but", "also",
}


def tokenize_for_bm25(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", text)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def dense_search(vector: list[float], app_state, top_k: int = 20) -> list[dict]:
    """Query Pinecone children namespace (dense vector search)."""
    results = app_state.pinecone_index.query(
        vector=vector, top_k=top_k, include_metadata=True, namespace=""
    )
    return [
        {
            "id": match["id"],
            "score": float(match["score"]),
            "text": match["metadata"].get("text", ""),
            "parent_id": match["metadata"].get("parent_id", ""),
            "title": match["metadata"].get("title", ""),
            "source": match["metadata"].get("source", ""),
            "type": match["metadata"].get("type", ""),
        }
        for match in results.get("matches", [])
    ]


def bm25_search(query: str, app_state, top_k: int = 20) -> list[dict]:
    """BM25 keyword search over child chunks."""
    bm25_data = app_state.bm25_data
    tokens = tokenize_for_bm25(query)
    if not tokens:
        return []

    scores = bm25_data["bm25"].get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]

    # Pre-compute fallback lists once (these keys may not exist in the pickle)
    num_docs = len(bm25_data["child_ids"])
    parent_ids = bm25_data.get("child_parent_ids") or [""] * num_docs
    titles = bm25_data.get("child_titles") or [""] * num_docs
    sources = bm25_data.get("child_sources") or [""] * num_docs
    types = bm25_data.get("child_types") or [""] * num_docs

    results = []
    for i in top_idx:
        if scores[i] <= 0:
            continue
        entry = {
            "id": bm25_data["child_ids"][i],
            "score": float(scores[i]),
            "text": bm25_data["child_texts"][i],
            "parent_id": parent_ids[i],
            "title": titles[i],
            "source": sources[i],
            "type": types[i],
        }
        results.append(entry)
    return results


def rrf_fusion(
    dense_results: list[dict], bm25_results: list[dict], k: int = 60, top_k: int = 20
) -> list[dict]:
    """Reciprocal Rank Fusion of dense + BM25 results."""
    scores = defaultdict(float)
    doc_map = {}

    for rank, doc in enumerate(dense_results):
        doc_id = doc["id"]
        scores[doc_id] += 1.0 / (k + rank + 1)
        doc_map[doc_id] = doc

    for rank, doc in enumerate(bm25_results):
        doc_id = doc["id"]
        scores[doc_id] += 1.0 / (k + rank + 1)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
    results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = scores[doc_id]
        results.append(doc)
    return results


async def hybrid_search(
    query: str, vector: list[float], app_state, settings
) -> list[dict]:
    """Run dense + BM25 search in parallel, then fuse with RRF."""
    loop = asyncio.get_running_loop()

    dense_task = loop.run_in_executor(
        None, dense_search, vector, app_state, settings.TOP_K_RETRIEVAL
    )
    bm25_task = loop.run_in_executor(
        None, bm25_search, query, app_state, settings.TOP_K_RETRIEVAL
    )

    dense_results, bm25_results = await asyncio.gather(dense_task, bm25_task)

    return rrf_fusion(
        dense_results, bm25_results, k=settings.RRF_K, top_k=settings.TOP_K_RETRIEVAL
    )
