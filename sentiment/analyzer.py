"""LLM-backed sentiment analysis on top of LangChain."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from pydantic import ValidationError

from .config import Settings
from .lexicon import LexiconAnalyzer
from .parsing import extract_json
from .prompts import analysis_prompt, comparison_prompt
from .schema import SentimentResult, normalize_label

LANGUAGE_NAMES = {"en": "English", "fr": "French", "ar": "Arabic", "es": "Spanish"}


def build_llm(settings: Settings) -> BaseChatModel:
    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.model,
            base_url=settings.base_url,
            temperature=settings.temperature,
            num_predict=settings.max_tokens,
            format="json",  # Ollama constrained decoding: always valid JSON
        )
    if settings.provider == "openai":
        if not settings.api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=settings.timeout,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    raise ValueError(f"Unsupported provider '{settings.provider}'.")


@dataclass
class AnalysisOutcome:
    """Result plus provenance, so the UI can tell the user what happened."""

    result: SentimentResult
    used_fallback: bool = False
    error: str | None = None


class SentimentAnalyzer:
    """Analyse text with a LangChain chain; degrade gracefully to the lexicon."""

    def __init__(
        self,
        llm: BaseChatModel,
        fallback: LexiconAnalyzer | None = None,
        language: str = "en",
    ):
        self.llm = llm
        self.fallback = fallback or LexiconAnalyzer()
        self.language = language
        self.chain = analysis_prompt | llm | StrOutputParser()
        self.compare_chain = comparison_prompt | llm | StrOutputParser()

    # --------------------------------------------------------------- public API
    def analyze(self, text: str, language: str | None = None) -> AnalysisOutcome:
        text = (text or "").strip()
        if not text:
            raise ValueError("Text is empty.")
        lang = language or self.language

        try:
            raw = self.chain.invoke(
                {"text": text, "answer_language": LANGUAGE_NAMES.get(lang, "English")}
            )
        except Exception as exc:  # network/timeout/model crash
            return AnalysisOutcome(
                self.fallback.analyze(text, lang), used_fallback=True, error=str(exc)[:200]
            )

        parsed = extract_json(raw)
        if parsed is None:
            return AnalysisOutcome(
                self.fallback.analyze(text, lang),
                used_fallback=True,
                error="Model did not return parseable JSON.",
            )

        try:
            result = parse_result(parsed)
        except (ValidationError, ValueError) as exc:
            return AnalysisOutcome(
                self.fallback.analyze(text, lang), used_fallback=True, error=str(exc)[:200]
            )
        return AnalysisOutcome(result)

    def analyze_batch(self, texts: list[str], language: str | None = None) -> list[AnalysisOutcome]:
        """Sequential batch analysis.

        Kept sequential on purpose: a single local Ollama instance serves one
        request at a time, so parallel calls only add queueing noise.
        """

        return [self.analyze(text, language) for text in texts]

    def compare(self, text_a: str, label_a: str, text_b: str, label_b: str) -> str:
        return self.compare_chain.invoke(
            {
                "text_a": text_a,
                "label_a": label_a,
                "text_b": text_b,
                "label_b": label_b,
            }
        ).strip()


def parse_result(payload: dict) -> SentimentResult:
    """Normalise the loose JSON dict into the strict schema."""

    if "label" not in payload and "sentiment" in payload:
        payload["label"] = payload["sentiment"]
    payload["label"] = normalize_label(payload.get("label"))

    for key in ("score", "confidence"):
        if key in payload:
            try:
                payload[key] = float(payload[key])
            except (TypeError, ValueError):
                payload.pop(key)
    if "score" in payload:
        payload["score"] = max(-1.0, min(1.0, payload["score"]))
    if "confidence" in payload:
        payload["confidence"] = max(0.0, min(1.0, payload["confidence"]))

    if not payload.get("rationale"):
        payload["rationale"] = f"Classified as {payload['label']}."
    return SentimentResult(**payload)


def build_default_analyzer(settings: Settings | None = None) -> SentimentAnalyzer:
    settings = settings or Settings.from_env()
    return SentimentAnalyzer(build_llm(settings))
