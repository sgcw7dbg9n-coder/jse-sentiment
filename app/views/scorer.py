from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import loaders


@st.cache_resource(show_spinner=False)
def lm_dictionary(path: str) -> tuple[set[str], set[str]]:
    from score_news import load_lm_dictionary
    return load_lm_dictionary(Path(path))


@st.cache_resource(show_spinner="Loading FinBERT (only the first time)…")
def finbert(model_name: str):
    from transformers import pipeline
    return pipeline("text-classification", model=model_name, top_k=None, truncation=True)


def score_headline(text: str, use_finbert: bool) -> tuple[dict, dict]:
    import score_news

    cfg = loaders.settings()["sentiment"]
    titles = pd.Series([text])
    scores, detail = {"vader_score": float(score_news.vader_scores(titles).iloc[0])}, {}

    path = loaders.ROOT / cfg["lm_dictionary"]
    if path.exists():
        positive, negative = lm_dictionary(str(path))
        scores["lm_score"] = float(score_news.lm_scores(titles, positive, negative)["lm_score"].iloc[0])
        tokens = score_news.TOKEN.findall(text.upper())
        detail["Loughran-McDonald positive words"] = ", ".join(w.lower() for w in tokens if w in positive) or "none"
        detail["Loughran-McDonald negative words"] = ", ".join(w.lower() for w in tokens if w in negative) or "none"
    else:
        detail["Loughran-McDonald"] = f"dictionary not found at {cfg['lm_dictionary']}"

    if use_finbert:
        try:
            result = finbert(cfg["finbert_model"])([text])[0]
        except OSError as exc:
            detail["FinBERT"] = f"could not load the model ({exc})"
        else:
            probs = {item["label"].lower(): item["score"] for item in result}
            scores["finbert_score"] = probs["positive"] - probs["negative"]
            detail["FinBERT probabilities"] = " · ".join(f"{label} {p:.2f}" for label, p in probs.items())
    return scores, detail


def score_chart(scores: dict) -> go.Figure:
    keys = [k for k in charts.SCORERS if k in scores]
    values = [scores[k] for k in keys]
    fig = go.Figure(go.Bar(
        x=values, y=[charts.SCORERS[k] for k in keys], orientation="h",
        marker_color=[charts.colour(k) for k in keys], width=0.4,
        text=values, texttemplate="%{text:+.2f}", textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{x:+.3f}</b><br>%{y}<extra></extra>",
    ))
    charts.reference_line(fig, "x", 0)
    fig.update_xaxes(range=[-1.15, 1.15], tickvals=[-1, -0.5, 0, 0.5, 1], tickformat="+.1f")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(title_text="Scores run from -1 (negative) to +1 (positive)")
    return charts.style(fig, height=90 + 56 * len(keys), legend=False)


def render() -> None:
    news = loaders.news()

    st.title("Score a headline")
    st.caption("Runs the same VADER, Loughran-McDonald and FinBERT code as score_news.py on any headline you type. "
               "GDELT tone comes from GDELT's own processing of the full article, so it can't be computed here.")

    stored, pick = None, ""
    if news is not None:
        examples = news.sort_values("finbert_score").iloc[[0, 1, 2, len(news) // 2, -3, -2, -1]]
        pick = st.selectbox("Start from a headline in the dataset (optional)", ["", *examples["title"]],
                            format_func=lambda t: t or "Type your own below")
        if pick:
            stored = examples[examples["title"] == pick].iloc[0]

    with st.form("score_form"):
        text = st.text_area("Headline", value=pick if stored is not None else "", max_chars=500,
                            placeholder="e.g. Sasol cuts dividend as fuel sales slump")
        use_finbert = st.toggle("Include FinBERT", value=True,
                                help="The model loads once per session, which can take up to a minute")
        submitted = st.form_submit_button("Score headline", type="primary")

    if not submitted or not text.strip():
        return
    scores, detail = score_headline(text.strip(), use_finbert)
    left, right = st.columns([3, 2], gap="large")
    with left:
        charts.show(score_chart(scores))
    with right:
        for label, value in detail.items():
            st.markdown(f"**{label}:** {value}")
        if stored is not None and text.strip() == stored["title"]:
            st.markdown("**Stored scores from the pipeline run**")
            st.dataframe(pd.DataFrame({
                "Scorer": [charts.SCORERS[k] for k in charts.SCORERS],
                "Stored": [stored[k] for k in charts.SCORERS],
                "Now": [scores.get(k) for k in charts.SCORERS],
            }), hide_index=True, column_config={
                "Stored": st.column_config.NumberColumn(format="%+.3f"),
                "Now": st.column_config.NumberColumn(format="%+.3f"),
            })
