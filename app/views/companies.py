import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import charts
import loaders

WINDOWS = {"Daily": "", "3-session": "_3d", "5-session": "_5d"}


def period_filter(cfg: dict, key: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = (pd.Timestamp(cfg["study_period"][k]) for k in ("start", "end"))
    train_end = pd.Timestamp(cfg["modelling"]["train_end"])
    presets = {
        "Whole study": (start, end),
        "Last 90 days": (end - pd.Timedelta(days=90), end),
        **{str(year): (pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))
           for year in range(start.year, train_end.year + 1)},
        "Test period": (train_end + pd.Timedelta(days=1), end),
    }
    choice = st.selectbox("Period", [*presets, "Custom"], key=f"{key}_period")
    if choice != "Custom":
        return presets[choice]
    picked = st.date_input("Dates", value=(start, end), min_value=start, max_value=end, key=f"{key}_dates")
    if len(picked) != 2:
        st.stop()
    return pd.Timestamp(picked[0]), pd.Timestamp(picked[1])


def standardise(news: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {score: (news[score].mean(), news[score].std()) for score in charts.SCORERS}


def explorer_chart(days: pd.DataFrame, scores: list[str], window: str, scale: dict | None,
                   train_end: pd.Timestamp) -> go.Figure:
    suffix = WINDOWS[window]
    count_column = "news_count" + suffix
    sentiment_title = "Sentiment" + (" (z-score)" if scale else "") + ("" if window == "Daily" else f", {window} average")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.42, 0.38, 0.2], vertical_spacing=0.06,
                        subplot_titles=("Close (R)", sentiment_title, "Headlines per session"))

    fig.add_trace(go.Scatter(x=days["date"], y=days["close_rand"], name="Close", mode="lines",
                             line=dict(color=charts.NEUTRAL, width=2), hovertemplate="R%{y:,.2f}",
                             showlegend=False), row=1, col=1)

    for score in scores:
        values = days[f"{score}_mean{suffix}"].where(days[count_column] > 0)
        if scale:
            mean, sd = scale[score]
            values = (values - mean) / sd
        daily = window == "Daily"
        fig.add_trace(go.Scatter(
            x=days["date"], y=values, name=charts.SCORERS[score], mode="markers" if daily else "lines",
            line=dict(color=charts.colour(score), width=2),
            marker=dict(color=charts.colour(score), size=8, line=dict(color=charts.surface(), width=2)),
            connectgaps=False, hovertemplate="%{y:+.2f}", showlegend=len(scores) > 1,
        ), row=2, col=1)
    charts.reference_line(fig, "y", 0, row=2, col=1)

    fig.add_trace(go.Bar(x=days["date"], y=days["news_count"], name="Headlines", marker_color=charts.NEUTRAL,
                         hovertemplate="%{y} headline(s)", showlegend=False), row=3, col=1)

    test_start = train_end + pd.Timedelta(days=1)
    if days["date"].max() >= test_start:
        fig.add_vrect(x0=max(test_start, days["date"].min()), x1=days["date"].max(), row="all", col=1,
                      fillcolor=charts.NEUTRAL, opacity=0.08, line_width=0, layer="below")
        fig.add_annotation(x=max(test_start, days["date"].min()), y=1, xref="x", yref="y domain", text="Test period",
                           showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=12))
    fig.update_yaxes(tickprefix="R", row=1, col=1)
    fig.update_yaxes(showgrid=False, row=3, col=1)
    fig = charts.style(fig, height=640, hovermode="x unified", legend=len(scores) > 1)
    return charts.legend_on_row(fig, 2)


def headline_table(news: pd.DataFrame, newest_first: bool = True) -> None:
    columns = ["trading_date", "published_local", "title", "domain", "vader_score", "finbert_score", "lm_score",
               "gdelt_tone", "finbert_label", "outlets", "url"]
    st.dataframe(
        (news.sort_values("published_local", ascending=False) if newest_first else news)[columns],
        hide_index=True,
        column_config={
            "trading_date": st.column_config.DateColumn("Session", format="YYYY-MM-DD"),
            "published_local": st.column_config.DatetimeColumn("Published (SAST)", format="YYYY-MM-DD HH:mm"),
            "title": st.column_config.TextColumn("Headline", width="large"),
            "domain": "Outlet",
            "vader_score": st.column_config.NumberColumn("VADER", format="%+.2f"),
            "finbert_score": st.column_config.NumberColumn("FinBERT", format="%+.2f"),
            "lm_score": st.column_config.NumberColumn("LM", format="%+.2f"),
            "gdelt_tone": st.column_config.NumberColumn("GDELT tone", format="%+.2f"),
            "finbert_label": "FinBERT label",
            "outlets": st.column_config.NumberColumn("Outlets", help="Outlets that carried the same headline that session"),
            "url": st.column_config.LinkColumn("Link", display_text="open"),
        },
    )


def render() -> None:
    cfg = loaders.settings()
    data = loaders.require(loaders.dataset(), "build_features.py")
    news = loaders.require(loaders.news(), "score_news.py")
    names = loaders.company_names()
    train_end = pd.Timestamp(cfg["modelling"]["train_end"])

    st.title("Company explorer")
    st.caption("Share price, headline sentiment and news volume for one company. Headlines published at or after "
               "17:00 SAST, or on a non-trading day, count towards the next session.")

    row = st.columns([3, 3, 2])
    with row[0]:
        ticker = st.selectbox("Company", list(names), format_func=names.get)
    with row[1]:
        start, end = period_filter(cfg, "companies")
    with row[2]:
        window = st.selectbox("Sentiment window", list(WINDOWS), index=2)
    row = st.columns([5, 3], vertical_alignment="bottom")
    with row[0]:
        scores = st.pills("Scorers", list(charts.SCORERS), format_func=charts.SCORERS.get, selection_mode="multi",
                          default=["vader_score", "finbert_score"])
    with row[1]:
        standardised = st.toggle("Standardise scores", value=True,
                                 help="GDELT tone runs roughly from -10 to +10 while the others run from -1 to +1. "
                                      "Standardising puts every scorer on the same scale so they share one axis.")

    days = data[(data["ticker"] == ticker) & data["date"].between(start, end)].sort_values("date")
    headlines = news[(news["ticker"] == ticker) & news["trading_date"].between(start, end)]
    if days.empty:
        st.warning("No trading sessions in that period.")
        st.stop()

    tiles = st.columns(4)
    tiles[0].metric("Headlines", f"{len(headlines):,}", border=True)
    tiles[1].metric("Sessions with news", f"{days['has_news'].mean():.0%}", border=True,
                    help=f"{int(days['has_news'].sum())} of {len(days)} sessions")
    tiles[2].metric("Average FinBERT score", f"{headlines['finbert_score'].mean():+.2f}" if len(headlines) else "–",
                    border=True)
    change = days["close"].iloc[-1] / days["close"].iloc[0] - 1
    tiles[3].metric("Price change", f"{change:+.1%}", border=True,
                    help=f"Close on {days['date'].iloc[0]:%d %b %Y} to close on {days['date'].iloc[-1]:%d %b %Y}")

    if not scores:
        st.info("Pick at least one scorer to plot sentiment.")
    else:
        charts.show(explorer_chart(days, scores, window, standardise(news) if standardised else None, train_end))

    st.subheader(f"Headlines ({len(headlines):,})")
    query = st.text_input("Search headlines", placeholder="e.g. dividend, results, strike")
    if query:
        headlines = headlines[headlines["title"].str.contains(query, case=False, regex=False)]
    headline_table(headlines)
