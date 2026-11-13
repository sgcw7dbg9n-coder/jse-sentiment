import html
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import charts
import loaders
from views.companies import headline_table, period_filter

SCORES = list(charts.SCORERS)


def correlation_chart(news: pd.DataFrame) -> go.Figure:
    corr = news[SCORES].corr(method="spearman")
    labels = [charts.SCORERS[s] for s in SCORES]
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=labels, y=labels, zmin=-1, zmax=1, colorscale=charts.diverging_scale(),
        texttemplate="%{z:.2f}", xgap=2, ygap=2, colorbar=dict(thickness=10, outlinewidth=0),
        hovertemplate="<b>%{z:.2f}</b><br>%{y} and %{x}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, showline=False)
    fig.update_layout(title_text="Spearman rank correlation between scorers")
    return charts.style(fig, height=380, legend=False)


def agreement_table(news: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in combinations(SCORES, 2):
        sign_a, sign_b = np.sign(news[a]), np.sign(news[b])
        both = (sign_a != 0) & (sign_b != 0)
        rows.append({
            "Pair": f"{charts.SCORERS[a]} · {charts.SCORERS[b]}",
            "Same": 100 * (both & (sign_a == sign_b)).mean(),
            "Opposite": 100 * (both & (sign_a != sign_b)).mean(),
            "Neutral": 100 * (~both).mean(),
        })
    return pd.DataFrame(rows)


def distribution_chart(news: pd.DataFrame) -> go.Figure:
    titles = [f"{charts.SCORERS[s]} · {(news[s] == 0).mean():.0%} exactly 0" for s in SCORES]
    fig = make_subplots(rows=1, cols=len(SCORES), subplot_titles=titles, horizontal_spacing=0.05)
    for i, score in enumerate(SCORES, start=1):
        fig.add_trace(go.Histogram(x=news[score], nbinsx=40, marker_color=charts.colour(score), name=charts.SCORERS[score],
                                   hovertemplate="%{x}: <b>%{y:,} headlines</b><extra></extra>"), row=1, col=i)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig = charts.style(fig, height=260, legend=False)
    fig.update_layout(bargap=0.08)
    return fig


def scatter_chart(news: pd.DataFrame, x: str, y: str) -> go.Figure:
    hover = np.stack([news["company"], news["title"].map(html.escape),
                      news["trading_date"].dt.strftime("%d %b %Y")], axis=-1)
    fig = go.Figure(go.Scattergl(
        x=news[x], y=news[y], mode="markers", customdata=hover,
        marker=dict(color=charts.NEUTRAL, size=8, opacity=0.45),
        hovertemplate=(f"<b>{charts.SCORERS[x]} %{{x:+.2f}} · {charts.SCORERS[y]} %{{y:+.2f}}</b>"
                       "<br>%{customdata[1]}<br>%{customdata[0]}, %{customdata[2]}<extra></extra>"),
    ))
    charts.reference_line(fig, "x", 0)
    charts.reference_line(fig, "y", 0)
    fig.update_xaxes(title_text=charts.SCORERS[x])
    fig.update_yaxes(title_text=charts.SCORERS[y])
    fig.update_layout(title_text="Each point is one headline")
    return charts.style(fig, height=440, legend=False)


def disagreements(news: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    opposite = news[np.sign(news[a]) * np.sign(news[b]) < 0].copy()
    z = {s: (opposite[s] - news[s].mean()) / news[s].std() for s in (a, b)}
    opposite["gap"] = (z[a] - z[b]).abs()
    return opposite.sort_values("gap", ascending=False)


def render() -> None:
    cfg = loaders.settings()
    news = loaders.require(loaders.news(), "score_news.py")
    names = loaders.company_names()

    st.title("Headlines and scorers")
    st.caption("How the four scorers rate the same headlines. VADER, FinBERT and Loughran-McDonald score the headline "
               "text; GDELT tone is GDELT's own score for the full article.")

    row = st.columns([2, 2, 5])
    with row[0]:
        ticker = st.selectbox("Company", ["all", *names], format_func=lambda t: "All companies" if t == "all" else names[t])
    with row[1]:
        start, end = period_filter(cfg, "headlines")
    selected = news[news["trading_date"].between(start, end)]
    if ticker != "all":
        selected = selected[selected["ticker"] == ticker]
    if len(selected) < 3:
        st.warning("Too few headlines in that selection to compare scorers.")
        st.stop()

    left, right = st.columns(2, gap="large")
    with left:
        charts.show(correlation_chart(selected))
    with right:
        st.markdown("**Do they point the same way?**")
        st.dataframe(agreement_table(selected), hide_index=True, column_config={
            column: st.column_config.NumberColumn(format="%.0f%%")
            for column in ("Same", "Opposite", "Neutral")
        })
        st.caption(f"Share of {len(selected):,} headlines where both scorers lean the same way, opposite ways, "
                   "or at least one scores exactly 0 (neutral).")

    charts.show(distribution_chart(selected))

    st.subheader("Compare two scorers")
    pick = st.columns([2, 2, 5])
    x = pick[0].selectbox("Horizontal axis", SCORES, index=1, format_func=charts.SCORERS.get)
    y = pick[1].selectbox("Vertical axis", [s for s in SCORES if s != x], format_func=charts.SCORERS.get)
    charts.show(scatter_chart(selected, x, y))

    opposite = disagreements(selected, x, y)
    st.markdown(f"**{charts.SCORERS[x]} and {charts.SCORERS[y]} point in opposite directions for "
                f"{len(opposite):,} headlines ({len(opposite) / len(selected):.0%}).** Largest disagreements first:")
    headline_table(opposite.head(200), newest_first=False)

    st.subheader("All headlines in this selection")
    query = st.text_input("Search headlines", placeholder="e.g. dividend, results, strike", key="headlines_search")
    if query:
        selected = selected[selected["title"].str.contains(query, case=False, regex=False)]
    headline_table(selected)
