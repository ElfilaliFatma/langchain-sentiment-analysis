"""Streamlit dashboard for sentiment analysis."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from sentiment.analyzer import build_default_analyzer
from sentiment.config import Settings
from sentiment.lexicon import LexiconAnalyzer

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")

EXAMPLES = [
    "The delivery was fast and the packaging was perfect. I will order again!",
    "The food arrived cold and the driver was rude. Very disappointing.",
    "The package arrived on Tuesday. It contains two items.",
    "The camera is excellent but the battery life is terrible.",
]


@st.cache_resource(show_spinner=False)
def get_analyzer(provider: str, model: str, use_llm: bool):
    if not use_llm:
        return None
    settings = Settings.from_env()
    return build_default_analyzer(
        Settings(
            provider=provider,
            model=model,
            base_url=settings.base_url,
            temperature=0.0,
            max_tokens=settings.max_tokens,
            timeout=settings.timeout,
            api_key=settings.api_key,
        )
    )


def show_result(result, used_fallback: bool = False, error: str | None = None) -> None:
    left, mid, right = st.columns([1, 1, 1])
    left.metric("Label", f"{result.emoji} {result.label}")
    mid.metric("Score", f"{result.score:+.2f}")
    right.metric("Confidence", f"{result.confidence:.0%}")
    st.progress((result.score + 1) / 2, text="negative ← → positive")

    if result.emotions:
        st.write("**Emotions:** " + " · ".join(result.emotions))
    if result.aspects:
        st.write("**Aspects**")
        st.dataframe(
            [a.model_dump() for a in result.aspects],
            use_container_width=True,
            hide_index=True,
        )
    st.info(result.rationale)
    if used_fallback:
        st.warning(f"LLM unavailable or invalid output, lexicon fallback used. ({error})")


def main() -> None:
    st.title("💬 Sentiment Analysis with LangChain")
    st.caption("Structured JSON output · aspect-based sentiment · lexicon fallback")

    with st.sidebar:
        st.header("Engine")
        use_llm = st.toggle("Use the LLM", value=True)
        provider = st.selectbox("Provider", ["ollama", "openai"])
        model = st.text_input("Model", "mistral")
        st.caption("Turn the LLM off to compare with the deterministic baseline.")

    tab_single, tab_batch = st.tabs(["Single text", "Batch CSV"])

    with tab_single:
        default = st.selectbox("Load an example", ["—"] + EXAMPLES)
        text = st.text_area(
            "Text to analyse",
            value=default if default != "—" else "",
            height=160,
            placeholder="Paste a review, a tweet, a customer email…",
        )
        if st.button("Analyse", type="primary", disabled=not text.strip()):
            with st.spinner("Analysing…"):
                if use_llm:
                    outcome = get_analyzer(provider, model, True).analyze(text)
                    show_result(outcome.result, outcome.used_fallback, outcome.error)
                else:
                    show_result(LexiconAnalyzer().analyze(text))

    with tab_batch:
        uploaded = st.file_uploader("CSV with a 'text' column", type="csv")
        if uploaded is not None:
            frame = pd.read_csv(uploaded)
            if "text" not in frame.columns:
                st.error("The CSV needs a 'text' column.")
            elif st.button("Run batch analysis", type="primary"):
                engine = get_analyzer(provider, model, True) if use_llm else None
                progress = st.progress(0.0)
                rows = []
                for index, value in enumerate(frame["text"].astype(str)):
                    if engine:
                        outcome = engine.analyze(value)
                        result = outcome.result
                        rows.append({"text": value, "label": result.label,
                                     "score": result.score,
                                     "confidence": result.confidence,
                                     "engine": "lexicon" if outcome.used_fallback else "llm"})
                    else:
                        result = LexiconAnalyzer().analyze(value)
                        rows.append({"text": value, "label": result.label,
                                     "score": result.score,
                                     "confidence": result.confidence,
                                     "engine": "lexicon"})
                    progress.progress((index + 1) / len(frame))
                analysed = pd.DataFrame(rows)
                st.dataframe(analysed, use_container_width=True, hide_index=True)
                counts = analysed["label"].value_counts()
                st.bar_chart(counts)
                buffer = io.StringIO()
                analysed.to_csv(buffer, index=False)
                st.download_button("Download results", buffer.getvalue(),
                                   file_name="sentiment_results.csv")


if __name__ == "__main__":
    main()
