# Technical Report — Sentiment Analysis with LangChain

Author: Fatma Ezzahra Elfilali

## 1. Problem and objectives

Businesses receive thousands of reviews, emails and messages. Manual triage
does not scale, while plain keyword counting misses negation, intensity and
mixed opinions. The objective is a sentiment application that:

1. returns a **machine-readable** verdict (not free prose);
2. distinguishes four classes, including `mixed`;
3. explains itself (rationale + aspects + evidence);
4. degrades gracefully when the model is unavailable;
5. is evaluated against a labelled set, with a deterministic baseline.

## 2. Architecture

The pipeline is `ChatPromptTemplate | ChatModel | StrOutputParser`, followed by
a tolerant JSON extractor, a Pydantic normaliser, and a lexicon fallback.
Layers and their tests are listed in `docs/architecture.md`.

Key implementation decisions:

1. **Constrained decoding first.** Ollama is configured with `format="json"`
   and OpenAI with `response_format={"type": "json_object"}`. This makes
   invalid JSON the exception rather than the rule.
2. **Parsing is defensive anyway.** `extract_json()` tries fenced blocks, the
   whole answer, brace-balanced blocks, and a repair pass (trailing commas,
   smart quotes, prefaces). Small models still produce `{"score": 0.7,}`.
3. **Schema as a contract.** `SentimentResult` normalises labels through an
   alias table (`pos`, `favorable`, `somewhat negative` → canonical classes),
   clamps `score`/`confidence`, and coerces string lists and bare aspect
   strings. Downstream code never sees an out-of-range value.
4. **Fallback with provenance.** `AnalysisOutcome` carries `used_fallback` and
   `error`, so the UI can warn the user and the evaluation can count fallbacks
   instead of silently mixing engines.
5. **Aspect-based sentiment.** The prompt asks for up to three aspects with
   their own label and evidence, which turns "good but expensive" into usable
   data instead of a single `mixed` tag.

## 3. Environment

Python 3.12, LangChain 0.3.x, langchain-ollama, Pydantic 2, Streamlit,
scikit-learn (evaluation only). Local model: `mistral` 7B Q4_K_M via Ollama.

## 4. Testing

### 4.1 Unit tests (30 assertions, 25 tests)

- **Parsing:** plain JSON, fenced JSON, JSON after a preface, trailing commas,
  braces inside strings, and four inputs that must return `None`.
- **Analyzer:** valid result; malformed JSON → fallback; exception → fallback;
  empty input → `ValueError`; prompt contains the text and the requested
  answer language; schema normalisation of aliases and string lists.
- **Lexicon:** labels in English and French, negation, intensifiers, contrast
  weighting, bounded outputs, keywords/aspects.

The fake chat model is a pydantic `BaseChatModel` subclass that records the
messages it receives; this is how prompt-level assertions stay deterministic.

### 4.2 Evaluation protocol

`data/labelled_reviews.csv` contains 28 human-labelled reviews (7 per class,
English). `scripts/evaluate.py` scores the lexicon baseline and the LLM on the
same rows with accuracy and macro-F1, and counts fallbacks.

Measured results (28 examples, `mistral` 7B):

| Engine | Accuracy | Macro-F1 |
| --- | --- | --- |
| Lexicon baseline | 0.679 | 0.591 |
| LLM (LangChain) | **0.893** | **0.903** |
| Fallbacks triggered | 0 | — |

Error analysis from the run:

- The baseline is perfect on `neutral` and on explicit praise/complaints, but
  it collapses on `mixed` (any sentence with both poles ends up neutral) and
  mislabels "The food arrived cold and the driver was rude." because "cold" is
  not in its lexicon.
- The LLM's three errors are all `negative → mixed` on sentences such as
  "The screenshots look nice, but the product itself is defective." The model
  reads the contrast as an aspect-level disagreement and returns `mixed`,
  which is defensible but differs from the human label. This is a genuine
  annotation-guideline ambiguity, documented rather than hidden.
- Every one of the 28 calls returned valid JSON, confirming that Ollama's
  `format="json"` plus the Pydantic normaliser removes the parsing failures
  that plague unstructured prompting.

Raw per-row output and the JSON report are reproducible with
`python scripts/evaluate.py`.

## 5. How LangChain contributes

- **Prompt templates** keep the JSON contract and the language slot in one
  place; changing the output language is a variable, not string surgery.
- **Provider abstraction** allows local/cloud switching without touching the
  analyzer.
- **LCEL composition** makes the pipeline readable and lets the code wrap the
  chain (fallback, batching) as ordinary Python.
- **Ecosystem** (`chain.batch`, structured output, caching) provides a clear
  upgrade path.

## 6. Limitations

1. Four classes cannot express intensity gradations or multi-target opinions.
2. Local 7B models are unreliable on sarcasm and on Derja/Arabic dialect.
3. Aspect extraction is limited to three aspects by design.
4. Sequential batch processing is slow on CPU (roughly 3-6 s per review).
5. The lexicon baseline ignores syntax beyond a 3-token window.
6. Self-reported confidence is not calibrated.

## 7. Improvements

1. Calibrate confidence on the labelled set and route low-confidence cases to a
   stronger model or to human review.
2. Use `with_structured_output(SentimentResult)` and measure its reliability
   versus the manual parser.
3. Replace the lexicon with a small fine-tuned classifier for the fast path.
4. Parallelise batches on a hosted endpoint with `chain.batch()`.
5. Add French/Arabic/Derja labelled data and per-language lexicons.
6. Persist results and build a trend dashboard (sentiment over time, per
   product).
