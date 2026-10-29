from __future__ import annotations

import pytest

from sentiment.lexicon import LexiconAnalyzer


@pytest.fixture()
def analyzer() -> LexiconAnalyzer:
    return LexiconAnalyzer()


@pytest.mark.parametrize(
    "text,label",
    [
        ("This is great, I love it!", "positive"),
        ("Terrible quality and rude support.", "negative"),
        ("The box contains two cables.", "neutral"),
        ("Très bon produit, je suis satisfait.", "positive"),
        ("Vraiment mauvais, je suis déçu.", "negative"),
    ],
)
def test_labels(analyzer, text, label):
    assert analyzer.analyze(text).label == label


def test_negation_flips_polarity(analyzer):
    assert analyzer.analyze("not good").score < 0
    assert analyzer.analyze("good").score > 0


def test_intensifier_increases_magnitude(analyzer):
    plain = analyzer.analyze("bad").score
    boosted = analyzer.analyze("extremely bad").score
    assert boosted < plain


def test_contrast_marker_weights_last_clause(analyzer):
    result = analyzer.analyze("The screen is beautiful but the battery is terrible.")
    assert result.score < 0


def test_scores_are_bounded(analyzer):
    result = analyzer.analyze("terrible horrible awful bad useless " * 5)
    assert -1.0 <= result.score <= -0.5
    assert 0.0 <= result.confidence <= 1.0


def test_keywords_and_aspects_present(analyzer):
    result = analyzer.analyze("great delivery, awful packaging")
    assert result.aspects
    assert set(result.keywords) & {"great", "awful"}
    assert result.engine == "lexicon"
