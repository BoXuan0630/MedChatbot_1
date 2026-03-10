def embed_query(text: str, app_state) -> list[float]:
    """Embed a query string using S-PubMedBert. Returns 768-dim vector."""
    vec = app_state.embedder.encode(
        [text], normalize_embeddings=True, convert_to_numpy=True
    )
    return vec[0].tolist()
