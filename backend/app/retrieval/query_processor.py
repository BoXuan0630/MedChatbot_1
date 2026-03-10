import asyncio
import logging

from app.services.cache_service import get_cached_translation, set_cached_translation

logger = logging.getLogger(__name__)

# Malay / Manglish keyword indicators
MALAY_KEYWORDS = {
    "apa", "bagaimana", "kenapa", "mengapa", "adakah", "boleh", "saya", "nak",
    "ubat", "sakit", "doktor", "kencing", "manis", "darah", "tinggi", "rendah",
    "makan", "jantung", "paru", "hati", "ginjal", "perut", "kepala", "demam",
    "batuk", "selsema", "gatal", "bengkak", "luka", "kecederaan", "hamil",
    "mengandung", "bayi", "kanak", "wanita", "lelaki", "tua", "muda",
    "rawatan", "simptom", "penyakit", "hospital", "klinik", "farmasi",
    "la", "lah", "kan", "meh", "ah", "lor", "eh", "hor", "macam", "macamana",
    "tak", "takde", "sikit", "banyak", "sangat", "dah", "sudah", "belum",
    "ini", "itu", "dia", "mereka", "kita", "kami", "awak", "encik", "puan",
}


GREETINGS = {
    "hi", "hello", "hey", "halo", "hai", "good morning", "good afternoon",
    "good evening", "good night", "morning", "afternoon", "evening",
    "whats up", "what's up", "yo", "howdy", "sup",
    "selamat pagi", "selamat tengahari", "selamat petang", "selamat malam",
    "apa khabar", "assalamualaikum", "salam",
}

GREETING_RESPONSE_EN = (
    "Hello! I'm MedBot, your Malaysian healthcare assistant. "
    "Feel free to ask me any health or medical questions!"
)
GREETING_RESPONSE_MS = (
    "Hai! Saya MedBot, pembantu kesihatan anda. "
    "Silakan tanya saya sebarang soalan tentang kesihatan!"
)

OFF_TOPIC_RESPONSE_EN = (
    "I'm a medical chatbot designed to help with healthcare-related questions. "
    "Feel free to ask me about symptoms, treatments, medications, or any health concerns!"
)
OFF_TOPIC_RESPONSE_MS = (
    "Saya adalah chatbot perubatan yang direka untuk membantu soalan berkaitan kesihatan. "
    "Silakan tanya saya tentang simptom, rawatan, ubat, atau sebarang kebimbangan kesihatan!"
)

MEDICAL_KEYWORDS = {
    "symptom", "treatment", "medicine", "drug", "disease", "diagnosis", "doctor",
    "hospital", "pain", "fever", "cough", "diabetes", "hypertension", "blood",
    "heart", "lung", "liver", "kidney", "cancer", "infection", "surgery",
    "pregnant", "pregnancy", "vitamin", "diet", "exercise", "obesity", "asthma",
    "allergy", "headache", "diarrhea", "vomit", "nausea", "rash", "swelling",
    "medication", "dose", "side effect", "health", "medical", "clinical",
    "therapy", "chronic", "acute", "prescription", "vaccine", "cholesterol",
    # Malay medical terms
    "ubat", "sakit", "doktor", "kencing", "manis", "darah", "tinggi",
    "jantung", "paru", "hati", "ginjal", "demam", "batuk", "rawatan",
    "simptom", "penyakit", "hamil", "kecederaan", "bengkak", "gatal",
}


def _has_medical_keyword(text: str) -> bool:
    """Check if text contains any medical keyword (single-word or multi-word)."""
    words = set(text.split())
    single_word_keywords = {kw for kw in MEDICAL_KEYWORDS if " " not in kw}
    multi_word_keywords = {kw for kw in MEDICAL_KEYWORDS if " " in kw}
    if words & single_word_keywords:
        return True
    for kw in multi_word_keywords:
        if kw in text:
            return True
    return False


def classify_intent(text: str) -> str:
    """Classify user intent: 'greeting', 'medical', or 'off_topic'."""
    normalized = text.strip().lower().rstrip("?!.")

    # Check greetings (exact match)
    if normalized in GREETINGS:
        return "greeting"

    # Check greeting prefix — but only if remainder has no medical keywords
    for g in GREETINGS:
        if normalized.startswith(g) and len(normalized) < len(g) + 10:
            remainder = normalized[len(g):].strip()
            if remainder and _has_medical_keyword(remainder):
                return "medical"
            return "greeting"

    # Check medical relevance (handles both single-word and multi-word keywords)
    if _has_medical_keyword(normalized):
        return "medical"

    # Short queries (1-3 words) without medical keywords are likely off-topic
    if len(normalized.split()) <= 3:
        return "off_topic"

    # Longer queries — assume medical intent (let retrieval pipeline decide)
    return "medical"


def detect_language(text: str) -> str:
    """Detect if text is Malay/Manglish ('ms') or English ('en')."""
    words = set(text.lower().split())
    malay_count = len(words & MALAY_KEYWORDS)
    if malay_count >= 2:
        return "ms"
    return "en"


async def translate_query(question: str, app_state, redis_client, settings) -> str:
    """Translate Malay/Manglish question to medical English using Gemini Flash.

    Checks Redis translation cache first. Returns the English translation.
    """
    # Check cache first
    try:
        cached = await get_cached_translation(question, redis_client)
        if cached:
            return cached
    except Exception:
        logger.warning("Redis translation cache check failed")

    prompt = (
        "Translate to medical English. Return only the translated question.\n"
        f"Question: {question}"
    )
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, app_state.gemini.generate_content, prompt
        )
        translation = response.text.strip()
    except Exception:
        logger.exception("Gemini translation failed")
        translation = question  # fallback to original

    # Cache the translation
    try:
        await set_cached_translation(
            question, translation, redis_client, ttl=settings.REDIS_TRANSLATE_TTL
        )
    except Exception:
        logger.warning("Redis translation cache set failed")
    return translation


async def reformulate_query(
    question: str, history: list[dict], app_state
) -> str:
    """Rewrite a follow-up question as a standalone question using chat history.

    Uses Gemini Flash. If history is empty or call fails, returns the original question.
    """
    if not history:
        return question

    history_text = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in history[-10:]
    )
    prompt = (
        "Rewrite as standalone question using history.\n"
        f"History: {history_text}\n"
        f"Question: {question}"
    )
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, app_state.gemini.generate_content, prompt
        )
        result = response.text.strip()
        return result if result else question
    except Exception:
        logger.exception("Gemini reformulation failed")
        return question
