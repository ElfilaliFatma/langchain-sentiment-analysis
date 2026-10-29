"""Compare the LLM against the lexicon baseline on a labelled dataset.

Metrics: accuracy and macro-F1 (scikit-learn). The same data is scored by both
engines, which is the cleanest way to demonstrate the contribution of the LLM
instead of assuming it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentiment.analyzer import build_default_analyzer  # noqa: E402
from sentiment.lexicon import LexiconAnalyzer  # noqa: E402

DATASET = ROOT / "data" / "labelled_reviews.csv"


def load_rows() -> list[dict]:
    with DATASET.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def scores(gold: list[str], predicted: list[str]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": round(accuracy_score(gold, predicted), 3),
        "macro_f1": round(f1_score(gold, predicted, average="macro", zero_division=0), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    rows = load_rows()
    gold = [row["label"] for row in rows]

    baseline = LexiconAnalyzer()
    base_pred = [baseline.analyze(row["text"]).label for row in rows]
    report = {"samples": len(rows), "lexicon": scores(gold, base_pred)}

    if not args.baseline_only:
        analyzer = build_default_analyzer()
        llm_pred, fallbacks = [], 0
        for row in rows:
            outcome = analyzer.analyze(row["text"])
            llm_pred.append(outcome.result.label)
            fallbacks += int(outcome.used_fallback)
            print(f"{row['label']:<9} -> {outcome.result.label:<9} {row['text'][:55]}")
        report["llm"] = scores(gold, llm_pred)
        report["llm_fallbacks"] = fallbacks

    print(json.dumps(report, indent=2))
    (ROOT / "docs" / "evaluation_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
