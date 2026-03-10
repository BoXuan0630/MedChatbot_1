import numpy as np


def _sigmoid(x):
    """Map raw logits to [0, 1] probability."""
    return 1.0 / (1.0 + np.exp(-x))


def rerank(query: str, candidates: list[dict], app_state, top_k: int = 5) -> list[dict]:
    """Rerank candidates using CrossEncoder. Returns top_k with normalized rerank_score in [0, 1]."""
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    raw_scores = app_state.reranker.predict(pairs)
    normalized_scores = _sigmoid(np.array(raw_scores))

    ranked = sorted(zip(normalized_scores, candidates), key=lambda x: x[0], reverse=True)

    results = []
    for score, chunk in ranked[:top_k]:
        chunk = chunk.copy()
        chunk["rerank_score"] = float(score)
        results.append(chunk)
    return results
