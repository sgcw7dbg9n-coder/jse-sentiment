import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import loaders

ALPHA = 0.05


def significant(tests: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hits = tests[tests["delong_p_holm"] < ALPHA]
    return hits[hits["auc_diff"] > 0], hits[hits["auc_diff"] < 0]


def describe(rows: pd.DataFrame) -> str:
    return "; ".join(
        f"{charts.MODELS_INLINE[r.model]} with {charts.FEATURE_SETS[r.feature_set].removeprefix('+ ')}, "
        f"{charts.target_label(r.target).lower()} ({r.auc_diff:+.3f} AUC, Holm p = {r.delong_p_holm:.3f})"
        for r in rows.itertuples()
    )


def hero(value: str, caption: str) -> None:
    st.markdown(f'<div style="font-size:56px;font-weight:600;line-height:1.1">{value}</div>', unsafe_allow_html=True)
    st.caption(caption)


def coverage_chart(news: pd.DataFrame, sessions: int) -> go.Figure:
    coverage = (news.groupby("company")
                .agg(headlines=("title", "size"), days=("trading_date", "nunique"))
                .sort_values("headlines", ascending=False))
    fig = go.Figure(go.Bar(
        x=coverage["headlines"], y=coverage.index, orientation="h", marker_color=charts.NEUTRAL,
        text=coverage["headlines"], texttemplate="%{text:,}", textposition="outside", cliponaxis=False,
        customdata=(coverage["days"] / sessions).to_numpy(),
        hovertemplate="<b>%{x:,} headlines</b><br>%{y}<br>news on %{customdata:.0%} of sessions<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, showticklabels=False, showline=False)
    fig.update_layout(title_text="Headlines per company after cleaning")
    fig = charts.style(fig, height=40 * len(coverage) + 70, legend=False)
    # Keeps each bar under 24px tall
    return fig.update_layout(bargap=0.5)


def findings(cfg: dict, tests: pd.DataFrame) -> list[str]:
    main = cfg["modelling"]["main_target"]
    lines = []

    secondary = tests[tests["target"] != main]
    if secondary.empty:
        lines.append("**Secondary (planned):** not run yet. Use `python src\\train_models.py --sensitivity`.")
    else:
        better, worse = significant(secondary)
        text = (f"**Secondary (planned):** {secondary['target'].nunique()} other targets "
                f"(3- and 5-session horizons and market-relative direction), {len(secondary)} comparisons. ")
        text += f"Sentiment significantly improved: {describe(better)}. " if len(better) else "No significant improvement. "
        text += f"Sentiment significantly hurt: {describe(worse)}." if len(worse) else ""
        lines.append(text)

    boot = loaders.bootstrap("all_days")
    if boot is not None and not boot.empty:
        clear = boot[(boot["feature_set"] == "all_sentiment") & ((boot["auc_ci_low"] > 0) | (boot["auc_ci_high"] < 0))]
        primary_clear = clear[clear["target"] == main]
        text = ("**Robustness:** a moving-block bootstrap over trading days "
                f"{'leaves zero inside every primary 95% interval' if primary_clear.empty else 'excludes zero for the primary target'}")
        if len(clear):
            text += ". Intervals that exclude zero: " + "; ".join(
                f"{charts.MODELS_INLINE[r.model]}, {charts.target_label(r.target).lower()} "
                f"({r.auc_gap:+.3f} [{r.auc_ci_low:+.3f}, {r.auc_ci_high:+.3f}])" for r in clear.itertuples())
        lines.append(text + ".")

    news_tests = loaders.tests("news_days")
    if news_tests is not None:
        better, worse = significant(news_tests)
        lines.append("**Exploratory, news days only:** "
                     + (f"significant gains for {describe(better)}." if len(better)
                        else f"no significant gain across {len(news_tests)} comparisons."))

    panel = loaders.report("panel_regressions.csv")
    if panel is not None:
        next_day = panel[panel["horizon"] == "next_day"]
        strongest = panel[panel["horizon"] == "same_day"].sort_values("p").iloc[0]
        lines.append(
            "**Exploratory, panel regressions:** "
            f"the strongest same-session link is {charts.SCORERS[strongest.score]} "
            f"({strongest.coef_pct_per_sd:+.3f}% return per standard deviation, p = {strongest.p:.3f}, "
            f"Holm {strongest.p_holm:.2f}); for the next session the smallest Holm-adjusted p is "
            f"{next_day['p_holm'].min():.2f}.")
    return lines


def render() -> None:
    cfg = loaders.settings()
    main = cfg["modelling"]["main_target"]
    train_end = pd.Timestamp(cfg["modelling"]["train_end"])
    news = loaders.require(loaders.news(), "score_news.py")
    data = loaders.require(loaders.dataset(), "build_features.py")
    results = loaders.require(loaders.results("all_days"), "train_models.py")
    tests = loaders.require(loaders.tests("all_days"), "train_models.py")

    start, end = (pd.Timestamp(cfg["study_period"][k]) for k in ("start", "end"))
    st.title("Does news sentiment help predict JSE share prices?")
    st.caption(f"{data['ticker'].nunique()} JSE Top 40 companies · headlines from South African outlets via GDELT · "
               f"trained on {start:%b %Y} to {train_end:%b %Y}, tested on {train_end + pd.Timedelta(days=1):%b} "
               f"to {end:%b %Y}")

    primary = tests[tests["target"] == main]
    better, worse = significant(primary)
    scored = results[(results["target"] == main) & (results["model"] != "majority_class")]
    majority = results[(results["target"] == main) & (results["model"] == "majority_class")].iloc[0]
    largest = primary.loc[primary["auc_diff"].idxmax()]
    sessions = data["date"].nunique()

    left, right = st.columns([2, 3], gap="large")
    with left:
        hero(f"{len(better)} of {len(primary)}",
             "sentiment-enhanced models predicted next-session direction significantly better than the "
             "prices-only baseline (DeLong test, Holm-adjusted, 5% level)")
        verdict = ("**The null hypothesis is not rejected.** " if better.empty
                   else f"**Sentiment helped in {len(better)} comparison(s):** {describe(better)}. ")
        verdict += (f"Test ROC-AUC ranged from {scored['roc_auc'].min():.3f} to {scored['roc_auc'].max():.3f} "
                    f"across every model and feature set, baseline included, against 0.5 for guessing. "
                    f"Always predicting the more common training class was right "
                    f"{majority['accuracy']:.1%} of the time. The largest AUC gain from sentiment was "
                    f"{largest['auc_diff']:+.3f} ({charts.MODELS_INLINE[largest['model']]}, "
                    f"{charts.FEATURE_SETS[largest['feature_set']].removeprefix('+ ')}), "
                    f"Holm-adjusted p = {largest['delong_p_holm']:.2f}.")
        if len(worse):
            verdict += f" Sentiment made the prediction significantly worse for {describe(worse)}."
        st.markdown(verdict)
    with right:
        charts.show(coverage_chart(news, sessions))

    tiles = st.columns(5)
    tiles[0].metric("Companies", data["ticker"].nunique(), border=True)
    tiles[1].metric("Headlines scored", f"{len(news):,}", border=True)
    tiles[2].metric("Company-days", f"{len(data):,}", border=True)
    tiles[3].metric("Days with news", f"{data['has_news'].mean():.1%}", border=True,
                    help="Share of company-days with at least one headline")
    tiles[4].metric("Test sessions", data.loc[data["date"] > train_end, "date"].nunique(), border=True)

    st.subheader("What the other tests show")
    for line in findings(cfg, tests):
        st.markdown(f"- {line}")
    st.caption("Primary and secondary tests were planned before looking at test results; exploratory checks "
               "were added afterwards and are reported as such.")
