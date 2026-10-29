"""Sentiment analysis with LangChain.

Two complementary engines:

* :class:`SentimentAnalyzer` -- LLM-based, returns structured JSON (label,
  score, confidence, emotions, aspects, rationale).
* :class:`LexiconAnalyzer` -- deterministic rule-based fallback that keeps the
  application usable when the model is unavailable or returns malformed JSON.
"""

from .analyzer import SentimentAnalyzer, build_default_analyzer, build_llm
from .config import Settings
from .lexicon import LexiconAnalyzer
from .schema import AspectSentiment, SentimentResult

__all__ = [
    "AspectSentiment",
    "LexiconAnalyzer",
    "SentimentAnalyzer",
    "SentimentResult",
    "Settings",
    "build_default_analyzer",
    "build_llm",
]

__version__ = "1.0.0"
