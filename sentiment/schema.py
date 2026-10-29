"""Pydantic schema used to normalise and validate model output.

The LLM is asked for JSON in this shape. Even when the model drifts, the
result is passed through this schema so downstream code always sees a valid
object (labels clamped to the known set, scores inside [-1, 1], non-empty
rationale).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

SentimentLabel = Literal["positive", "negative", "neutral", "mixed"]


class AspectSentiment(BaseModel):
    """Sentiment about one specific aspect of a text."""

    aspect: str = Field(description="Short noun phrase, e.g. 'delivery speed'.")
    sentiment: SentimentLabel
    evidence: str = Field(default="", description="Quote supporting the label.")


class SentimentResult(BaseModel):
    label: SentimentLabel
    score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="-1 very negative, 0 neutral, +1 very positive.",
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    emotions: list[str] = Field(default_factory=list)
    aspects: list[AspectSentiment] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    rationale: str = ""
    engine: str = "llm"

    @field_validator("emotions", "keywords", mode="before")
    @classmethod
    def _as_list(cls, value):
        """Coerce 'joy, trust' or None into a clean list of strings."""

        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return [str(v).strip() for v in value if str(v).strip()]

    @field_validator("aspects", mode="before")
    @classmethod
    def _aspects_as_objects(cls, value):
        """Accept [{'aspect': 'price', 'sentiment': 'negative'}] or bare strings."""

        if not value:
            return []
        cleaned = []
        for item in value:
            if isinstance(item, AspectSentiment):
                cleaned.append(item)
            elif isinstance(item, str):
                cleaned.append({"aspect": item, "sentiment": "neutral"})
            elif isinstance(item, dict):
                label = str(item.get("sentiment", item.get("label", "neutral"))).lower()
                cleaned.append(
                    {
                        "aspect": str(item.get("aspect", item.get("name", "unspecified"))),
                        "sentiment": normalize_label(label),
                        "evidence": str(item.get("evidence", "")),
                    }
                )
        return cleaned

    @property
    def emoji(self) -> str:
        return {"positive": "😊", "negative": "😞", "neutral": "😐", "mixed": "🤔"}[self.label]


_ALIASES: dict[str, str] = {
    "positive": "positive",
    "pos": "positive",
    "favorable": "positive",
    "favourable": "positive",
    "happy": "positive",
    "negative": "negative",
    "neg": "negative",
    "unfavorable": "negative",
    "sad": "negative",
    "angry": "negative",
    "neutral": "neutral",
    "neut": "neutral",
    "objective": "neutral",
    "mixed": "mixed",
    "ambivalent": "mixed",
    "both": "mixed",
}


def normalize_label(raw: str | None) -> str:
    """Map free-form model labels onto the four supported classes."""

    if not raw:
        return "neutral"
    key = str(raw).strip().lower().strip(".!\"'")
    if key in _ALIASES:
        return _ALIASES[key]
    for alias, canonical in _ALIASES.items():
        if alias in key:
            return canonical
    return "neutral"
