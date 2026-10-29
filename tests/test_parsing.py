from __future__ import annotations

import pytest

from sentiment.parsing import balanced_objects, extract_json


@pytest.mark.parametrize(
    "raw",
    [
        '{"label": "positive", "score": 0.8}',
        '```json\n{"label": "positive", "score": 0.8}\n```',
        'Here is the analysis:\n{"label": "positive", "score": 0.8}',
        'Sure! ```\n{"label": "positive", "score": 0.8}\n```',
    ],
)
def test_extract_json_common_shapes(raw):
    assert extract_json(raw)["label"] == "positive"


def test_extract_json_repairs_trailing_comma():
    assert extract_json('{"label": "negative", "score": -0.5,}')["score"] == -0.5


def test_extract_json_prefers_balanced_block():
    raw = 'noise {broken} then {"label": "mixed", "score": 0.0} end'
    assert extract_json(raw)["label"] == "mixed"


def test_balanced_objects_ignores_braces_in_strings():
    blocks = balanced_objects('{"note": "a } brace", "ok": 1}')
    assert blocks == ['{"note": "a } brace", "ok": 1}']


@pytest.mark.parametrize("raw", ["", "   ", "no json here", "[1, 2, 3]"])
def test_extract_json_returns_none(raw):
    assert extract_json(raw) is None
