"""Deterministic fallback analyser.

No network, no model, no external package: a compact valence lexicon with
negation, intensifier and contrast handling. It is used (a) when the LLM fails,
and (b) as a baseline in the evaluation, which makes the value of the LLM
measurable instead of assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schema import AspectSentiment, SentimentResult

POSITIVE: dict[str, float] = {
    "amazing": 3, "awesome": 3, "excellent": 3, "fantastic": 3, "perfect": 3,
    "love": 2.5, "loved": 2.5, "great": 2.5, "wonderful": 2.5, "best": 2.5,
    "good": 2, "nice": 1.6, "happy": 2, "pleased": 2, "satisfied": 2,
    "recommend": 1.8, "helpful": 1.6, "fast": 1.4, "reliable": 1.6,
    "friendly": 1.6, "professional": 1.4, "affordable": 1.4, "worth": 1.4,
    "smooth": 1.3, "clear": 1.1, "works": 1.0, "impressed": 2.0,
    "delicious": 2.2, "comfortable": 1.4, "responsive": 1.4, "okay": 0.6,
    "ok": 0.6, "fine": 0.6, "decent": 0.8, "solid": 1.2, "clean": 1.0,
    "bien": 2, "excellent": 3, "parfait": 3, "génial": 3, "bon": 1.8,
    "satisfait": 2, "rapide": 1.4, "fiable": 1.6, "aimé": 2.2, "agréable": 1.6,
}

NEGATIVE: dict[str, float] = {
    "terrible": 3, "awful": 3, "horrible": 3, "worst": 3, "useless": 3,
    "hate": 2.6, "hated": 2.6, "disappointed": 2.4, "disappointing": 2.4,
    "bad": 2, "poor": 1.9, "broken": 2.2, "defective": 2.2, "waste": 2.4,
    "slow": 1.5, "expensive": 1.4, "rude": 2.2, "unhelpful": 1.9,
    "late": 1.4, "missing": 1.5, "crash": 2.2, "crashes": 2.2, "bug": 1.5,
    "bugs": 1.5, "dirty": 1.8, "noisy": 1.5, "uncomfortable": 1.7,
    "overpriced": 2.0, "annoying": 1.8, "confusing": 1.5, "wrong": 1.8,
    "mauvais": 2, "horrible": 3, "nul": 2.4, "déçu": 2.4, "lent": 1.5,
    "cher": 1.4, "problème": 1.6, "cassé": 2.2, "déteste": 2.6,
}

NEGATORS = {"not", "no", "never", "none", "cannot", "can't", "don't", "doesn't",
            "didn't", "isn't", "wasn't", "aren't", "won't", "hardly", "barely",
            "ne", "pas", "jamais", "plus", "aucun"}

INTENSIFIERS = {"very": 1.5, "really": 1.4, "extremely": 1.8, "absolutely": 1.7,
                "so": 1.3, "too": 1.2, "super": 1.5, "très": 1.5, "vraiment": 1.4}

CONTRAST_MARKERS = {"but", "however", "although", "though", "mais", "cependant"}

WORD = re.compile(r"[a-zA-ZÀ-ÿ']+")


@dataclass
class _Hit:
    word: str
    value: float
    index: int


@dataclass
class _Analysis:
    hits: list[_Hit] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(hit.value for hit in self.hits)


class LexiconAnalyzer:
    """Rule-based sentiment scorer with negation and contrast handling."""

    name = "lexicon"

    def analyze(self, text: str, language: str = "en") -> SentimentResult:
        tokens = [token.lower() for token in WORD.findall(text)]
        hits = self._collect_hits(tokens)
        raw = sum(hit.value for hit in hits)

        # The clause after the last contrast marker dominates in practice.
        text_lower = text.lower()
        last_contrast = max(
            (text_lower.rfind(marker) for marker in CONTRAST_MARKERS), default=-1
        )
        if last_contrast > 0:
            dominance = self._score_tail(text, last_contrast)
            raw = dominance * 1.15 if dominance * raw > 0 else dominance

        label, score, confidence = self._classify(raw, len(hits), len(tokens))
        return SentimentResult(
            label=label,
            score=round(score, 3),
            confidence=round(confidence, 3),
            emotions=self._emotions(label),
            aspects=[
                AspectSentiment(aspect=hit.word, sentiment=self._word_label(hit.value))
                for hit in hits
            ][:5],
            keywords=[hit.word for hit in sorted(hits, key=lambda h: -abs(h.value))][:5],
            rationale=f"Lexicon baseline: {len(hits)} polar word(s), net valence {raw:+.2f}.",
            engine="lexicon",
        )

    # ------------------------------------------------------------------ helpers
    def _collect_hits(self, tokens: list[str]) -> list[_Hit]:
        hits: list[_Hit] = []
        for index, token in enumerate(tokens):
            value = POSITIVE.get(token, 0.0) - NEGATIVE.get(token, 0.0)
            if value == 0:
                continue
            window = tokens[max(0, index - 3) : index]
            if any(word in NEGATORS for word in window):
                value *= -0.8
            for word in window:
                if word in INTENSIFIERS:
                    value *= INTENSIFIERS[word]
            hits.append(_Hit(word=token, value=value, index=index))
        return hits

    def _score_tail(self, text: str, marker_index: int) -> float:
        tail_tokens = [token.lower() for token in WORD.findall(text[marker_index:])]
        return sum(hit.value for hit in self._collect_hits(tail_tokens))

    @staticmethod
    def _classify(raw: float, hit_count: int, token_count: int):
        if hit_count == 0:
            return "neutral", 0.0, 0.4
        density = hit_count / max(token_count, 1)
        score = max(-1.0, min(1.0, raw / (3.0 + hit_count)))
        if abs(score) < 0.15:
            label = "neutral"
        else:
            label = "positive" if score > 0 else "negative"
        confidence = min(0.85, 0.45 + density * 0.4 + min(hit_count, 4) * 0.05)
        return label, score, confidence

    @staticmethod
    def _word_label(value: float) -> str:
        if value > 0.4:
            return "positive"
        if value < -0.4:
            return "negative"
        return "neutral"

    @staticmethod
    def _emotions(label: str) -> list[str]:
        return {
            "positive": ["satisfaction"],
            "negative": ["disappointment"],
            "neutral": [],
            "mixed": ["ambivalence"],
        }[label]
