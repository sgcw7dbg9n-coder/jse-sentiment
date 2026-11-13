import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import charts
import loaders

ROLLING = 20


def prediction_chart(days: pd.DataFrame, compare: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.4, 0.2], vertical_spacing=0.06,
                        subplot_titles=("Close (R)", f"Share of the last {ROLLING} sessions predicted correctly",
                                        "Headlines per session"))
    fig.add_trace(go.Scatter(x=days["date"], y=days["close_rand"], mode="lines", name="Close",
                             line=dict(color=charts.NEUTRAL, width=2), hovertemplate="R%{y:,.2f}", showlegend=False),
                  row=1, col=1)
    for column, key in (("hit_base", "baseline"), ("hit_enh", compare)):
        rolling = days[column].astype(float).rolling(ROLLING, min_periods=ROLLING // 2).mean()
        fig.add_trace(go.Scatter(x=days["date"], y=rolling, mode="lines", name=charts.FEATURE_SETS[key],
                                 line=dict(color=charts.colour(key), width=2), hovertemplate="%{y:.0%}"), row=2, col=1)
    charts.reference_line(fig, "y", 0.5, row=2, col=1)
    fig.add_trace(go.Bar(x=days["date"], y=days["news_count"], name="Headlines", marker_color=charts.NEUTRAL,
                         hovertemplate="%{y} headline(s)", showlegend=False), row=3, col=1)
    fig.update_yaxes(tickprefix="R", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], row=2, col=1)
    fig.update_yaxes(showgrid=False, row=3, col=1)
    fig = charts.style(fig, height=600, hovermode="x unified")
    return charts.legend_on_row(fig, 2)


def render() -> None:
    cfg = loaders.settings()
    main = cfg["modelling"]["main_target"]
    predictions = loaders.require(loaders.predictions("all_days"), "train_models.py")
    data = loaders.require(loaders.dataset(), "build_features.py")
    predictions = predictions[predictions["target"] == main]
    names = loaders.company_names()

    st.title("Test-period predictions")
    st.caption("Out-of-sample predictions of next-session direction for the test period, from models trained only "
               "on earlier data. Research prototype for an academic study: not investment advice.")

    sets = [s for s in charts.FEATURE_SETS if s in set(predictions["feature_set"]) and s != "baseline"]
    row = st.columns([2, 2, 2, 4])
    ticker = row[0].selectbox("Company", list(names), format_func=names.get)
    model = row[1].selectbox("Model", list(charts.MODELS), format_func=charts.MODELS.get)
    compare = row[2].selectbox("Compare baseline with", sets, format_func=charts.FEATURE_SETS.get)

    chosen = predictions[(predictions["ticker"] == ticker) & (predictions["model"] == model)]
    key = ["date", "ticker", "y_true"]
    days = (chosen[chosen["feature_set"] == "baseline"][key + ["score", "pred"]]
            .merge(chosen[chosen["feature_set"] == compare][key + ["score", "pred"]], on=key, suffixes=("_base", "_enh"))
            .merge(data[["date", "ticker", "close_rand", "fwd_ret_1d", "news_count", "finbert_score_mean"]],
                   on=["date", "ticker"], how="left")
            .sort_values("date"))
    days["hit_base"] = days["pred_base"] == days["y_true"]
    days["hit_enh"] = days["pred_enh"] == days["y_true"]

    hit_base, hit_enh = days["hit_base"].mean(), days["hit_enh"].mean()
    tiles = st.columns(4)
    tiles[0].metric("Test sessions", len(days), border=True)
    tiles[1].metric("Sessions followed by a rise", f"{days['y_true'].mean():.0%}", border=True)
    tiles[2].metric("Correct, prices only", f"{hit_base:.1%}", border=True)
    gap = 100 * (hit_enh - hit_base)
    tiles[3].metric(f"Correct, with {charts.FEATURE_SETS[compare].removeprefix('+ ')}", f"{hit_enh:.1%}",
                    delta=f"{gap:+.1f} points vs baseline" if round(gap, 1) else None, border=True)
    st.caption(f"With about {len(days)} sessions for one company, a few points either way is within chance; the "
               "Model comparison page tests all ten companies together.")

    charts.show(prediction_chart(days, compare))

    direction = days["y_true"].map({1: "Up", 0: "Down"})
    st.dataframe(
        days.sort_values("date", ascending=False).assign(
            actual=direction,
            next_ret=days["fwd_ret_1d"] * 100,
            pred_base=days["pred_base"].map({1: "Up", 0: "Down"}),
            pred_enh=days["pred_enh"].map({1: "Up", 0: "Down"}),
        )[["date", "close_rand", "next_ret", "actual", "pred_base", "score_base", "pred_enh", "score_enh",
           "news_count", "finbert_score_mean"]],
        hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Session", format="YYYY-MM-DD"),
            "close_rand": st.column_config.NumberColumn("Close (R)", format="%.2f"),
            "next_ret": st.column_config.NumberColumn("Next-session return %", format="%+.2f"),
            "actual": "Actual",
            "pred_base": "Prices only",
            "score_base": st.column_config.NumberColumn(
                "Score, prices only", format="%.3f",
                help="Probability of an up move for logistic regression and random forest; signed distance from "
                     "the decision boundary for the SVM"),
            "pred_enh": "With sentiment",
            "score_enh": st.column_config.NumberColumn("Score, with sentiment", format="%.3f"),
            "news_count": st.column_config.NumberColumn("Headlines", format="%d"),
            "finbert_score_mean": st.column_config.NumberColumn("Average FinBERT", format="%+.2f"),
        },
    )
