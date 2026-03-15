import asyncio
import json
import logging
import re

import httpx

logger = logging.getLogger(__name__)

DISCLAIMER_EN = "Note: This answer is based on general medical knowledge. Please consult your doctor."
DISCLAIMER_MS = "Nota: Jawapan ini berdasarkan pengetahuan perubatan umum. Sila rujuk doktor anda."

# Phrases that indicate the LLM couldn't answer from the provided context
REFUSAL_PHRASES = [
    "does not list", "does not mention", "does not contain",
    "does not provide", "does not include", "does not specify",
    "does not address", "does not discuss", "does not state",
    "not explicitly stated", "not explicitly mentioned",
    "not mentioned in the context", "not found in the context",
    "no information", "no relevant information",
    "cannot be determined from", "cannot answer",
    "not available in the provided", "not covered in",
    "tidak menyebut", "tidak menyatakan", "tidak mengandungi",
    "tiada maklumat",
]


def _is_refusal(answer: str) -> bool:
    """Check if the LLM's answer is a refusal / 'I don't know' from context."""
    lower = answer.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


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
        "You are MedBot, a Malaysian healthcare assistant for elderly patients.\n"
        "Answer the question using ONLY the provided context from official "
        "Malaysian CPG guidelines and drug formulary.\n\n"
        "Rules:\n"
        "- Be clear, concise, and use simple language suitable for elderly patients\n"
        "- Cite sources as [Source N: Title] after each relevant statement\n"
        "- Structure your answer with bullet points for treatments/symptoms\n"
        "- If the context contains partial information, provide what is available\n"
        "- Keep answers under 300 words unless the question requires a comprehensive list\n"
        f"- Answer in {lang_name}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )


def _build_knowledge_prompt(query: str, language: str) -> str:
    lang_name = "Bahasa Melayu" if language == "ms" else "English"
    return (
        "You are MedBot, a Malaysian healthcare assistant for elderly patients.\n"
        "Answer this medical question using your general knowledge.\n\n"
        "Rules:\n"
        "- Be clear, concise, and use simple language suitable for elderly patients\n"
        "- Structure your answer with bullet points where appropriate\n"
        "- Keep answers under 250 words\n"
        "- Do NOT invent specific drug dosages or cite non-existent studies\n"
        "- For serious conditions, recommend consulting a doctor\n"
        f"- Answer in {lang_name}\n\n"
        f"Question: {query}"
    )


def _build_file_context_prompt(question: str, file_text: str, language: str) -> str:
    lang_name = "Bahasa Melayu" if language == "ms" else "English"
    return (
        "You are MedBot, a Malaysian healthcare assistant for elderly patients.\n"
        "The user uploaded a health document. Here is the extracted content:\n\n"
        "---\n"
        f"{file_text}\n"
        "---\n\n"
        f"User's question: {question}\n\n"
        "Rules:\n"
        "- Answer based on the document content above\n"
        "- Use bullet points for clarity\n"
        "- Highlight any abnormal values or concerning findings\n"
        "- Keep answers under 400 words unless the question requires detail\n"
        "- For serious findings, recommend consulting a doctor\n"
        f"- IMPORTANT: You MUST answer in {lang_name} regardless of the document's language\n"
        f"- If the document is in a different language, translate the relevant information to {lang_name}"
    )


async def _call_gemini(prompt: str, app_state) -> str:
    """Call Gemini Flash synchronously via executor. Returns stripped text or raises."""
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None, app_state.gemini.generate_content, prompt
    )
    if not response.candidates or not response.text:
        raise ValueError("Empty response from Gemini")
    return response.text.strip()


async def generate_answer(
    query: str,
    retrieval_result: dict,
    language: str,
    app_state,
    settings,
) -> tuple[str, str]:
    """Generate an answer using MedGemma (if available) or Gemini Flash fallback.

    Includes grounded-mode refusal detection: if the LLM says "context doesn't
    contain the answer", automatically re-generates in knowledge mode.

    Returns (answer, model_used).
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
                # Check for refusal in grounded mode
                if context_found and _is_refusal(answer):
                    logger.info("MedGemma refused in grounded mode, falling back to knowledge mode")
                else:
                    if not context_found:
                        disclaimer = DISCLAIMER_MS if language == "ms" else DISCLAIMER_EN
                        answer = f"{answer}\n\n{disclaimer}"
                    return answer, "medgemma"
        except Exception:
            logger.exception("MedGemma call failed, falling back to Gemini Flash")

    # Gemini Flash generation
    model_used = "gemini"

    if context_found:
        prompt = _build_grounded_prompt(query, context, language)
    else:
        prompt = _build_knowledge_prompt(query, language)

    try:
        answer = await _call_gemini(prompt, app_state)
    except Exception as e:
        logger.error("Gemini generation failed: %s", e)
        answer = (
            "Maaf, saya tidak dapat menjawab soalan ini sekarang. Sila cuba lagi."
            if language == "ms"
            else "Sorry, I cannot answer this question right now. Please try again."
        )
        return answer, model_used

    # Grounded-mode refusal detection: if LLM says "context doesn't answer",
    # re-generate in knowledge mode so the user still gets a useful answer
    if context_found and _is_refusal(answer):
        logger.info("Grounded answer was a refusal, re-generating in knowledge mode")
        try:
            knowledge_prompt = _build_knowledge_prompt(query, language)
            answer = await _call_gemini(knowledge_prompt, app_state)
            model_used = "gemini_fallback"
        except Exception as e:
            logger.error("Knowledge-mode fallback also failed: %s", e)
        # Always add disclaimer for knowledge-mode fallback
        disclaimer = DISCLAIMER_MS if language == "ms" else DISCLAIMER_EN
        answer = f"{answer}\n\n{disclaimer}"
        return answer, model_used

    # Append disclaimer for knowledge mode
    if not context_found:
        disclaimer = DISCLAIMER_MS if language == "ms" else DISCLAIMER_EN
        answer = f"{answer}\n\n{disclaimer}"

    return answer, model_used


async def generate_follow_ups(
    query: str, answer: str, language: str, app_state
) -> list[str]:
    """Generate 2 suggested follow-up questions based on the Q&A.

    Returns a list of question strings, or empty list on failure.
    """
    lang_name = "Bahasa Melayu" if language == "ms" else "English"
    prompt = (
        f"Based on this medical Q&A, suggest 2 brief follow-up questions "
        f"the patient might ask next. Answer in {lang_name}.\n\n"
        f"Q: {query}\n"
        f"A: {answer[:300]}\n\n"
        'Return ONLY a JSON array: ["question1", "question2"]'
    )
    try:
        raw = await _call_gemini(prompt, app_state)
        # Extract JSON array from response (handle markdown code blocks)
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        logger.warning("Failed to generate follow-up questions")
    return []
