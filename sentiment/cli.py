"""CLI: single text, file, or CSV batch."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .analyzer import build_default_analyzer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sentiment", description="Analyse sentiment.")
    parser.add_argument("text", nargs="?", help="Text to analyse.")
    parser.add_argument("--file", type=Path, help="Read text from a file.")
    parser.add_argument("--csv", type=Path, help="Batch mode: CSV with a 'text' column.")
    parser.add_argument("--out", type=Path, help="Write batch results to this CSV.")
    parser.add_argument("--lang", default="en", help="Language of the rationale (en/fr/ar/es).")
    parser.add_argument("--json", action="store_true", help="Print raw JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252; emoji labels and non-Latin rationales
    # raise UnicodeEncodeError without this.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = parse_args(argv)
    analyzer = build_default_analyzer()

    if args.csv:
        rows = list(csv.DictReader(args.csv.open(encoding="utf-8", newline="")))
        if not rows or "text" not in rows[0]:
            print("CSV must contain a 'text' column.")
            return 2
        out_rows = []
        for row in rows:
            outcome = analyzer.analyze(row["text"], args.lang)
            result = outcome.result
            out_rows.append(
                {
                    "text": row["text"],
                    "label": result.label,
                    "score": result.score,
                    "confidence": result.confidence,
                    "emotions": ", ".join(result.emotions),
                    "rationale": result.rationale,
                    "engine": "lexicon-fallback" if outcome.used_fallback else "llm",
                }
            )
            print(f"{result.label:<9} {result.score:+.2f}  {row['text'][:60]}")
        if args.out:
            with args.out.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
                writer.writeheader()
                writer.writerows(out_rows)
            print(f"Wrote {len(out_rows)} rows to {args.out}")
        return 0

    if args.file:
        text = args.file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("Provide text, --file, --csv, or pipe it on stdin.")
        return 2

    outcome = analyzer.analyze(text, args.lang)
    if args.json:
        print(outcome.result.model_dump_json(indent=2))
        return 0

    result = outcome.result
    print(f"{result.emoji} {result.label.upper()}  score={result.score:+.2f}  "
          f"confidence={result.confidence:.2f}")
    if result.emotions:
        print(f"Emotions: {', '.join(result.emotions)}")
    for aspect in result.aspects:
        print(f"  - {aspect.aspect}: {aspect.sentiment} {aspect.evidence}")
    print(f"Rationale: {result.rationale}")
    if outcome.used_fallback:
        print(f"[fallback used: {outcome.error}]", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
