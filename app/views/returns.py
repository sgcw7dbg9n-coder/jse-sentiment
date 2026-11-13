import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import charts
import loaders
from robustness import sentiment_terciles

HORIZONS = {"same_day": "Same session", "next_day": "Next session"}
TERCILES = ["negative", "neutral", "positive"]


def coefficient_chart(panel: pd.DataFrame) -> go.Figure:
    labels = [charts.SCORERS[s] for s in charts.SCORERS]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=list(HORIZONS.values()))
    for col, horizon in enumerate(HORIZONS, start=1):
        for row in panel[panel["horizon"] == horizon].itertuples():
            margin = 1.96 * row.std_err
            fig.add_trace(go.Scatter(
                x=[row.coef_pct_per_sd], y=[charts.SCORERS[row.score]], mode="markers",
                marker=dict(size=12, color=charts.colour(row.score), line=dict(color=charts.surface(), width=2)),
                error_x=dict(type="constant", value=margin, thickness=2, width=0, color=charts.colour(row.score)),
                hovertemplate=(f"<b>%{{x:+.3f}}% per SD</b> ± {margin:.3f}<br>%{{y}}<br>p {row.p:.3f}, "
                               f"Holm {row.p_holm:.3f} · {row.n:,} company-days<extra></extra>"),
                showlegend=False,
            ), row=1, col=col)
        charts.reference_line(fig, "x", 0, row=1, col=col)
    fig.update_yaxes(autorange="reversed", showgrid=False, categoryorder="array", categoryarray=labels)
    fig.update_xaxes(ticksuffix="%", tickformat="+.2f")
    fig.update_layout(title_text="Return per one standard deviation of sentiment, with 95% intervals")
    return charts.style(fig, height=320, legend=False)


def tercile_chart(terciles: pd.DataFrame, score: str) -> go.Figure:
    parts = terciles.loc[TERCILES]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.08,
                        subplot_titles=("Same-session return", "Next-session return"))
    for col, column in enumerate(("same_day_ret_pct", "next_day_ret_pct"), start=1):
        fig.add_trace(go.Bar(
            y=[f"{t.capitalize()} third" for t in TERCILES], x=parts[column], orientation="h",
            marker_color=charts.colour(score), text=parts[column], texttemplate="%{text:+.3f}%",
            textposition="outside", cliponaxis=False,
            customdata=parts[["company_days", "mean_sentiment"]].to_numpy(),
            hovertemplate=("<b>%{x:+.3f}%</b> average return<br>%{y} · %{customdata[0]:,} company-days"
                           "<br>average score %{customdata[1]:+.2f}<extra></extra>"),
            showlegend=False,
        ), row=1, col=col)
        charts.reference_line(fig, "x", 0, row=1, col=col)
    reach = parts[["same_day_ret_pct", "next_day_ret_pct"]].abs().max().max() * 1.5
    fig.update_xaxes(ticksuffix="%", range=[-reach, reach], showline=False, matches="x")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(title_text=f"News days split into thirds by {charts.SCORERS[score]} score")
    fig = charts.style(fig, height=300, legend=False)
    # Keeps each bar under 24px tall
    return fig.update_layout(bargap=0.65)


def render() -> None:
    data = loaders.require(loaders.dataset(), "build_features.py")
    panel = loaders.require(loaders.report("panel_regressions.csv"), "robustness.py")

    st.title("Sentiment and returns")
    st.caption("Exploratory checks on company-days with at least one headline. These look at the size of the "
               "relationship between sentiment and returns rather than at classification accuracy.")

    st.subheader("Panel regressions")
    charts.show(coefficient_chart(panel))
    st.caption("Return (%) regressed on standardised sentiment with the Satrix 40 return and company fixed effects; "
               "the next-session model also controls for the same-session return. Standard errors are clustered by "
               "date because all ten companies share the same market news on a given day.")
    st.dataframe(
        panel.assign(score=panel["score"].map(charts.SCORERS), horizon=panel["horizon"].map(HORIZONS))
        [["horizon", "score", "coef_pct_per_sd", "std_err", "t", "p", "p_holm", "n", "r2"]],
        hide_index=True,
        column_config={
            "horizon": "Return", "score": "Scorer",
            "coef_pct_per_sd": st.column_config.NumberColumn("% per SD", format="%+.4f"),
            "std_err": st.column_config.NumberColumn("Std error", format="%.4f"),
            "t": st.column_config.NumberColumn("t", format="%+.2f"),
            "p": st.column_config.NumberColumn("p", format="%.3f"),
            "p_holm": st.column_config.NumberColumn("p (Holm)", format="%.3f"),
            "n": st.column_config.NumberColumn("Company-days", format="%d"),
            "r2": st.column_config.NumberColumn("R²", format="%.3f"),
        },
    )

    st.subheader("Returns by sentiment third")
    score = st.pills("Scorer", list(charts.SCORERS), format_func=charts.SCORERS.get, default="finbert_score") \
        or "finbert_score"
    terciles = sentiment_terciles(data, score)
    charts.show(tercile_chart(terciles, score))
    spread = terciles.loc["positive minus negative"]
    st.markdown(f"Average return on positive-third days minus negative-third days: "
                f"**{spread['same_day_ret_pct']:+.3f} percentage points** in the same session and "
                f"**{spread['next_day_ret_pct']:+.3f}** in the next session.")
    st.dataframe(
        terciles.rename(index=str.capitalize).rename_axis("Third").reset_index(),
        hide_index=True,
        column_config={
            "company_days": st.column_config.NumberColumn("Company-days", format="%d"),
            "mean_sentiment": st.column_config.NumberColumn("Average score", format="%+.3f"),
            "same_day_ret_pct": st.column_config.NumberColumn("Same-session return %", format="%+.3f"),
            "next_day_ret_pct": st.column_config.NumberColumn("Next-session return %", format="%+.3f"),
            "next_day_up_share": st.column_config.NumberColumn("Share up next session", format="%.3f"),
        },
    )
