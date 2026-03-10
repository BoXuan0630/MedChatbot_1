import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

DISCLAIMER_EN = "Note: This answer is based on general medical knowledge. Please consult your doctor."
DISCLAIMER_MS = "Nota: Jawapan ini berdasarkan pengetahuan perubatan umum. Sila rujuk doktor anda."


async def call_medgemma(
    query: str,
    context: str,
    language: str,
    context_found: bool,
    url: str,
    client: httpx.AsyncClient,
) -> str:
    """Call the external MedGemma GPU VM. Returns answer string or raises on failure."""
    url = url.rstrip("/")
    r = await client.post(
        f"{url}/generate",
        json={
            "query": query,
            "context": context,
            "language": language,
            "context_found": context_found,
        },
        headers={"ngrok-skip-browser-warning": "true", "bypass-tunnel-reminder": "true"},
    )
    if r.status_code != 200:
        detail = ""
        try:
            detail = r.json().get("error", r.text)
        except Exception:
            detail = r.text
        logger.error("MedGemma returned %s: %s", r.status_code, detail)
        r.raise_for_status()
    return r.json()["answer"]


def _build_grounded_prompt(query: str, context: str, language: str) -> str:
    lang_name = "Bahasa Melayu" if language == "ms" else "English"
    return (
        f"Answer using ONLY this context. Cite as [Source N: Title]. "
        f"Answer in {lang_name}.\n"
        f"Context: {context}\n"
        f"Question: {query}"
    )


def _build_knowledge_prompt(query: str, language: str) -> str:
    lang_name = "Bahasa Melayu" if language == "ms" else "English"
    return (
        f"Answer using medical knowledge. No citations. "
        f"Answer in {lang_name}.\n"
        f"Question: {query}"
    )


async def generate_answer(
    query: str,
    retrieval_result: dict,
    language: str,
    app_state,
    settings,
) -> tuple[str, str]:
    """Generate an answer using MedGemma (if available) or Gemini Flash fallback.

    Returns (answer, model_used) where model_used is "medgemma" or "gemini".
    """
    context_found = retrieval_result["context_found"]
    parents = retrieval_result.get("parents", {})

    # Build context from parent texts
    context = ""
    if context_found and parents:
        context_parts = []
        for i, (pid, data) in enumerate(parents.items(), 1):
            context_parts.append(f"[Source {i}: {data['title']}]\n{data['text']}")
        context = "\n\n".join(context_parts)

    # Try MedGemma first (if URL is configured)
    if settings.MEDGEMMA_URL:
        try:
            answer = await call_medgemma(
                query=query,
                context=context,
                language=language,
                context_found=context_found,
                url=settings.MEDGEMMA_URL,
                client=app_state.http_client,
            )
            if answer:
                if not context_found:
                    disclaimer = DISCLAIMER_MS if language == "ms" else DISCLAIMER_EN
                    answer = f"{answer}\n\n{disclaimer}"
                return answer, "medgemma"
        except Exception:
            logger.exception("MedGemma call failed, falling back to Gemini Flash")

    # Gemini Flash fallback
    if context_found:
        prompt = _build_grounded_prompt(query, context, language)
    else:
        prompt = _build_knowledge_prompt(query, language)

    generation_succeeded = False
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, app_state.gemini.generate_content, prompt
        )
        if not response.candidates or not response.text:
            raise ValueError("Empty response from Gemini")
        answer = response.text.strip()
        generation_succeeded = True
    except Exception as e:
        logger.error("Gemini generation failed: %s", e)
        answer = (
            "Maaf, saya tidak dapat menjawab soalan ini sekarang. Sila cuba lagi."
            if language == "ms"
            else "Sorry, I cannot answer this question right now. Please try again."
        )

    # Append disclaimer for knowledge mode — only if generation actually succeeded
    if not context_found and generation_succeeded:
        disclaimer = DISCLAIMER_MS if language == "ms" else DISCLAIMER_EN
        answer = f"{answer}\n\n{disclaimer}"

    return answer, "gemini"
