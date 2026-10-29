# Architecture

```
text ──► analysis_prompt ──► ChatModel (JSON mode) ──► StrOutputParser
                                   │
                          ┌────────┴────────┐
                          │                 │
                    extract_json()     exception
                          │                 │
                   parse_result()           │
                   (Pydantic, clamps)       │
                          │                 │
                     SentimentResult ◄──────┘
                          │        LexiconAnalyzer fallback
              ┌───────────┼────────────┐
              ▼           ▼            ▼
          Streamlit      CLI      evaluate.py (F1)
```

## Layers

| Layer | Module | Testable without a model? |
| --- | --- | --- |
| Prompt | `prompts.py` | yes |
| Parsing | `parsing.py`, `schema.py` | yes |
| Orchestration | `analyzer.py` | yes (scripted fake model) |
| Baseline | `lexicon.py` | yes |
| Presentation | `app.py`, `cli.py` | manually |

The fallback is a design decision, not a patch: a sentiment dashboard that
returns HTTP 500 because the model emitted a trailing comma is not usable, so
every failure path returns a valid `SentimentResult` tagged with
`engine="lexicon"`.
