"""Robust JSON extraction for small local models.

Models routinely wrap JSON in ```json fences, add a sentence before it, or
duplicate keys. This module recovers the object without ever raising on a
malformed model answer; callers decide what to do with ``None``.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(raw: str) -> dict[str, Any] | None:
    """Return the first JSON object found in ``raw``, or ``None``."""

    if not raw or not raw.strip():
        return None

    candidates: list[str] = []
    candidates.extend(match.group(1) for match in _FENCE.finditer(raw))
    candidates.append(raw)
    candidates.extend(balanced_objects(raw))

    for candidate in candidates:
        parsed = _try_load(candidate)
        if isinstance(parsed, dict):
            return parsed

    for candidate in candidates:
        repaired = _repair(candidate)
        parsed = _try_load(repaired)
        if isinstance(parsed, dict):
            return parsed
    return None


def _try_load(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None


def balanced_objects(text: str) -> list[str]:
    """Yield every top-level ``{...}`` block, respecting nested braces/strings."""

    blocks: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                blocks.append(text[start : index + 1])
                start = -1
    return blocks


def _repair(text: str) -> str:
    """Best-effort fixes for the most common small-model JSON mistakes."""

    fixed = text.strip()
    fixed = re.sub(r"^[^{]*", "", fixed, count=1)  # drop preface
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)  # trailing commas
    fixed = fixed.replace("“", '"').replace("”", '"')
    fixed = fixed.replace("'", '"') if '"' not in fixed else fixed
    return fixed.strip()
