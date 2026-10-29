# Sentiment Analysis with LangChain

An application that reads a review, message or email and returns a **structured
sentiment verdict**: label (`positive`, `negative`, `neutral`, `mixed`), a
score in `[-1, 1]`, confidence, emotions, per-aspect sentiment, keywords and a
one-sentence rationale.

![pipeline](docs/architecture.md)

## Why it is more than one prompt

| Concern | Handling |
| --- | --- |
| Model returns prose instead of JSON | `format="json"` for Ollama, JSON mode for OpenAI, plus a fence/brace/repair parser |
| Model returns a different schema | Pydantic validation with label aliases and clamping |
| Model is offline or broken | Deterministic lexicon fallback (negation, intensifiers, contrast) |
| User asks for French/Arabic rationale | `answer_language` prompt slot |
| "It felt positive" | Measured against a labelled set with accuracy / macro-F1 |

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
ollama pull mistral
copy .env.example .env
```

### CLI

```powershell
python -m sentiment.cli "The camera is excellent but the battery is terrible."
python -m sentiment.cli --csv data\labelled_reviews.csv --out results.csv
python -m sentiment.cli "Produit parfait, livraison rapide." --lang fr
```

### Dashboard

```powershell
streamlit run sentiment/app.py
```

The sidebar can switch the LLM off, which runs the same UI on the lexicon
baseline — useful for a live comparison during the demo.

### Tests and evaluation

```powershell
pytest
python scripts/evaluate.py            # LLM vs baseline
python scripts/evaluate.py --baseline-only
```

## Layout

```
sentiment/
  schema.py    Pydantic contract + label normalisation
  prompts.py   analysis and comparison prompts
  parsing.py   tolerant JSON extraction
  analyzer.py  LangChain chain, fallback logic, batching
  lexicon.py   deterministic baseline
  cli.py       command line
  app.py       Streamlit dashboard
data/
  labelled_reviews.csv   28 human-labelled examples (4 classes)
scripts/
  evaluate.py  accuracy + macro-F1
tests/
```

## How LangChain contributes

`ChatPromptTemplate | ChatModel | StrOutputParser` keeps the prompt, the model
and the parsing separate and swappable. The provider abstraction means the same
analyzer runs on Ollama and OpenAI. Because the chain is a plain object, the
fallback logic and the batch runner wrap it without touching prompt code.

## Limitations

- Positive/negative is a coarse view of opinion; `mixed` detection is the
  weakest class for 7B models.
- Aspect extraction is only as good as the number of aspects requested.
- Sarcasm and code-switched text (French/Arabic/Derja) remain hard.
- The lexicon baseline has no context beyond a 3-token window.

## Improvements

- Fine-tune a small classifier on the labelled set and use the LLM only for
  ambiguous cases (routing with a LangGraph conditional edge).
- Add a `with_structured_output()` variant and compare reliability against the
  manual JSON parser.
- Use `chain.batch()` with `max_concurrency` on a hosted endpoint.
- Store results in SQLite to build a trend dashboard over time.
- Add multilingual lexicons and a Derja-specific list.

See [`docs/REPORT.md`](docs/REPORT.md) for the full write-up.
