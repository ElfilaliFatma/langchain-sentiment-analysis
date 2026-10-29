from __future__ import annotations

from typing import Any, List, Optional

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from sentiment.analyzer import SentimentAnalyzer, parse_result
from sentiment.lexicon import LexiconAnalyzer
from sentiment.schema import SentimentResult, normalize_label


class ScriptedChatModel(BaseChatModel):
    responses: List[str]
    calls: List[List[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        text = self.responses.pop(0) if self.responses else ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


GOOD = """{"label": "positive", "score": 0.7, "confidence": 0.9,
"emotions": ["joy"], "aspects": [{"aspect": "delivery", "sentiment": "positive",
"evidence": "arrived in one day"}], "keywords": ["fast", "packaging"],
"rationale": "Customer is happy with speed and packaging."}"""


def analyzer_with(*responses: str) -> SentimentAnalyzer:
    return SentimentAnalyzer(ScriptedChatModel(responses=list(responses)))


def test_llm_result_parsed():
    outcome = analyzer_with(GOOD).analyze("Fast delivery, great packaging!")
    assert outcome.result.label == "positive"
    assert outcome.result.score == 0.7
    assert outcome.result.aspects[0].aspect == "delivery"
    assert outcome.used_fallback is False


def test_malformed_json_triggers_fallback():
    outcome = analyzer_with("I think this is quite positive overall.").analyze(
        "Great product, I love it."
    )
    assert outcome.used_fallback is True
    assert outcome.result.engine == "lexicon"
    assert outcome.result.label == "positive"
    assert "JSON" in (outcome.error or "")


def test_model_exception_triggers_fallback():
    class Exploding(ScriptedChatModel):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("ollama is down")

    analyzer = SentimentAnalyzer(Exploding(responses=[]))
    outcome = analyzer.analyze("This is terrible and useless.")
    assert outcome.used_fallback is True
    assert outcome.result.label == "negative"
    assert "ollama is down" in (outcome.error or "")


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        analyzer_with(GOOD).analyze("   ")


def test_prompt_contains_text_and_answer_language():
    model = ScriptedChatModel(responses=[GOOD])
    SentimentAnalyzer(model).analyze("Great!", language="fr")
    human = model.calls[-1][-1].content
    assert "<review>\nGreat!\n</review>" in human
    system = model.calls[-1][0].content
    assert "French" in system


def test_label_aliases_and_case():
    assert normalize_label("POSITIVE") == "positive"
    assert normalize_label("mixed feelings") == "mixed"
    assert normalize_label("somewhat negative") == "negative"
    assert normalize_label(None) == "neutral"
    assert normalize_label("banana") == "neutral"


def test_parse_result_clamps_and_defaults():
    result = parse_result({"sentiment": "neg", "score": -9, "confidence": 3})
    assert result.label == "negative"
    assert result.score == -1.0
    assert result.confidence == 1.0
    assert result.rationale


def test_parse_result_accepts_string_lists_and_bare_aspects():
    result = parse_result({"label": "mixed", "emotions": "joy, trust", "aspects": ["price"]})
    assert result.emotions == ["joy", "trust"]
    assert result.aspects[0].sentiment == "neutral"


def test_aspects_tolerate_label_key():
    result = parse_result(
        {"label": "negative", "aspects": [{"name": "battery", "label": "Negative"}]}
    )
    assert result.aspects[0].aspect == "battery"
    assert result.aspects[0].sentiment == "negative"
