"""Unit tests for query_processor — no external dependencies (no DB/Redis/Pinecone)."""

import pytest

from app.retrieval.query_processor import (
    classify_intent,
    detect_language,
    has_dosage_change_phrase,
    _has_medical_keyword,
    _has_emergency_phrase,
)
from app.retrieval.hybrid_search import tokenize_for_bm25
from app.services.cache_service import cache_key


# ── classify_intent ──────────────────────────────────────────


class TestClassifyIntent:
    def test_greeting_exact(self):
        assert classify_intent("hello") == "greeting"
        assert classify_intent("Hi") == "greeting"
        assert classify_intent("selamat pagi") == "greeting"
        assert classify_intent("assalamualaikum") == "greeting"

    def test_greeting_with_punctuation(self):
        assert classify_intent("hello!") == "greeting"
        assert classify_intent("hey?") == "greeting"

    def test_greeting_prefix_short(self):
        assert classify_intent("hi there") == "greeting"

    def test_greeting_with_medical_is_medical(self):
        assert classify_intent("hi, I have diabetes") == "medical"

    def test_medical_english(self):
        assert classify_intent("What is the treatment for diabetes?") == "medical"
        assert classify_intent("symptoms of hypertension") == "medical"
        assert classify_intent("side effect of metformin") == "medical"

    def test_medical_malay(self):
        assert classify_intent("ubat kencing manis") == "medical"
        assert classify_intent("rawatan demam") == "medical"

    def test_off_topic_short(self):
        assert classify_intent("hello world program") == "off_topic"
        assert classify_intent("what time") == "off_topic"

    def test_longer_non_medical_is_medical(self):
        # Longer queries without medical keywords still get classified as medical
        # to let the retrieval pipeline decide
        assert classify_intent("I have been feeling very unwell for the past week") == "medical"

    def test_emergency(self):
        assert classify_intent("I'm having chest pain right now") == "emergency"
        assert classify_intent("can't breathe") == "emergency"
        assert classify_intent("I want to kill myself") == "emergency"
        assert classify_intent("overdose") == "emergency"
        assert classify_intent("sakit dada sekarang") == "emergency"

    def test_emergency_takes_priority_over_medical(self):
        assert classify_intent("chest pain right now with diabetes") == "emergency"


# ── detect_language ──────────────────────────────────────────


class TestDetectLanguage:
    def test_english(self):
        assert detect_language("What is the treatment for diabetes?") == "en"

    def test_malay(self):
        assert detect_language("Apakah rawatan untuk kencing manis?") == "ms"

    def test_manglish(self):
        assert detect_language("Diabetes tu ubat apa ah?") == "ms"

    def test_single_malay_word_is_english(self):
        # Only 1 match, threshold is 2
        assert detect_language("ubat please") == "en"

    def test_two_malay_words_is_malay(self):
        assert detect_language("ubat sakit") == "ms"


# ── _has_medical_keyword ─────────────────────────────────────


class TestHasMedicalKeyword:
    def test_single_word_match(self):
        assert _has_medical_keyword("diabetes") is True
        assert _has_medical_keyword("fever") is True

    def test_multi_word_match(self):
        assert _has_medical_keyword("what is the side effect") is True

    def test_no_match(self):
        assert _has_medical_keyword("hello world") is False

    def test_malay_medical(self):
        assert _has_medical_keyword("ubat") is True
        assert _has_medical_keyword("rawatan") is True


# ── _has_emergency_phrase ────────────────────────────────────


class TestHasEmergencyPhrase:
    def test_match(self):
        assert _has_emergency_phrase("i am having chest pain right now") is True
        assert _has_emergency_phrase("can't breathe") is True

    def test_no_match(self):
        assert _has_emergency_phrase("what is chest pain") is False


# ── has_dosage_change_phrase ─────────────────────────────────


class TestHasDosageChangePhrase:
    def test_english_match(self):
        assert has_dosage_change_phrase("Should I stop taking metformin?") is True
        assert has_dosage_change_phrase("Can I increase my dose?") is True

    def test_malay_match(self):
        assert has_dosage_change_phrase("Boleh berhenti makan ubat?") is True

    def test_no_match(self):
        assert has_dosage_change_phrase("What is the dosage of metformin?") is False


# ── tokenize_for_bm25 ───────────────────────────────────────


class TestTokenizeForBm25:
    def test_basic(self):
        tokens = tokenize_for_bm25("What is the treatment for diabetes?")
        assert "treatment" in tokens
        assert "diabetes" in tokens

    def test_stop_words_removed(self):
        tokens = tokenize_for_bm25("the a an is in on at to for of")
        assert tokens == []

    def test_single_char_removed(self):
        tokens = tokenize_for_bm25("a b c diabetes")
        assert tokens == ["diabetes"]

    def test_lowercase(self):
        tokens = tokenize_for_bm25("DIABETES Treatment")
        assert "diabetes" in tokens
        assert "treatment" in tokens

    def test_numbers_in_words(self):
        tokens = tokenize_for_bm25("type2 diabetes hba1c")
        assert "type2" in tokens
        assert "hba1c" in tokens


# ── cache_key ────────────────────────────────────────────────


class TestCacheKey:
    def test_deterministic(self):
        k1 = cache_key("answer", "What is diabetes?")
        k2 = cache_key("answer", "What is diabetes?")
        assert k1 == k2

    def test_case_insensitive(self):
        k1 = cache_key("answer", "What is Diabetes?")
        k2 = cache_key("answer", "what is diabetes?")
        assert k1 == k2

    def test_whitespace_normalized(self):
        k1 = cache_key("answer", "  diabetes  ")
        k2 = cache_key("answer", "diabetes")
        assert k1 == k2

    def test_prefix(self):
        k1 = cache_key("answer", "diabetes")
        k2 = cache_key("translate", "diabetes")
        assert k1 != k2
        assert k1.startswith("answer:")
        assert k2.startswith("translate:")
