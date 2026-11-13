import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import loaders

PIPELINE = [
    ("Collect daily prices", "collect_prices.py", "data/raw/prices.csv"),
    ("Collect GDELT headlines", "collect_gdelt_news.py", "data/raw/gdelt_headlines.parquet"),
    ("Clean and date headlines", "clean_news.py", "data/interim/news_clean.parquet"),
    ("Score sentiment", "score_news.py", "data/interim/news_scored.parquet"),
    ("Build features and targets", "build_features.py", "data/processed/model_dataset.parquet"),
    ("Train and compare models", "train_models.py", "reports/model_results.csv"),
    ("Robustness checks", "robustness.py", "reports/bootstrap_comparisons.csv"),
]


def pipeline_table() -> pd.DataFrame:
    rows = []
    for stage, script, output in PIPELINE:
        frame = loaders.read(loaders.ROOT / output)
        rows.append({"Stage": stage, "Script": f"src/{script}", "Main output": output,
                     "Rows": "not run yet" if frame is None else f"{len(frame):,}"})
    return pd.DataFrame(rows)


def funnel_chart(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=summary["rows"], y=summary["step"], orientation="h", marker_color=charts.NEUTRAL,
        text=summary["rows"], texttemplate="%{text:,}", textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{x:,} rows</b><br>%{y}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, showticklabels=False, showline=False)
    fig.update_layout(title_text="Rows left after each cleaning step")
    fig = charts.style(fig, height=44 * len(summary) + 80, legend=False)
    # Keeps each bar under 24px tall
    return fig.update_layout(bargap=0.5)


def render() -> None:
    cfg = loaders.settings()
    model_cfg = cfg["modelling"]
    start, end = (pd.Timestamp(cfg["study_period"][k]) for k in ("start", "end"))
    train_end = pd.Timestamp(model_cfg["train_end"])

    st.title("Data and method")
    horizons = [h for h in cfg["features"]["horizons"] if h != 1]

    st.subheader("Study design")
    st.markdown(f"""
- **Period:** {start.day} {start:%B %Y} to {end.day} {end:%B %Y}. Models are tuned on sessions up to
  {train_end.day} {train_end:%B %Y} and
  tested once on the sessions after it.
- **Market proxy:** {cfg['market_proxy']['name']} ({cfg['market_proxy']['ticker']}).
- **News timing:** a headline published at or after {cfg['market']['close_time']} SAST, or on a weekend or public
  holiday, is assigned to the next trading session, so no model sees news before the market could.
- **Targets:** the primary target is whether the close one session ahead is above today's close. Sensitivity runs use
  {' and '.join(f'{h}-session' for h in horizons)} horizons, and direction relative to the market proxy at every
  horizon.
- **Validation:** {model_cfg['cv_splits']} expanding, date-based folds with a purge gap equal to the target horizon,
  tuned on ROC-AUC.
- **Comparison:** DeLong test for AUC and McNemar test for accuracy, each sentiment model against the same model on
  prices only, with a Holm adjustment across the comparisons for each target.
""")

    st.subheader("Pipeline")
    st.dataframe(pipeline_table(), hide_index=True)

    summary = loaders.report("news_cleaning_summary.csv")
    if summary is not None:
        charts.show(funnel_chart(summary))

    st.subheader("Companies")
    companies = pd.DataFrame(cfg["companies"])
    st.dataframe(companies[["ticker", "name", "sector", "pattern", "exclude"]], hide_index=True, column_config={
        "ticker": "Ticker", "name": "Company", "sector": "Sector",
        "pattern": st.column_config.TextColumn("Headline pattern", help="Regular expression matched against lowercased headlines"),
        "exclude": st.column_config.TextColumn("Excluded when matching", help="Sponsorships, indices and economist commentary"),
    })
    selection = loaders.report("sample_selection.csv")
    if selection is not None:
        with st.expander("How the ten companies were chosen"):
            st.dataframe(selection, hide_index=True)
            excluded = cfg.get("excluded_candidates")
            if excluded:
                st.markdown("**Candidates left out for thin South African headline coverage**")
                st.dataframe(pd.DataFrame(excluded), hide_index=True)

    dictionary = loaders.report("feature_dictionary.csv")
    if dictionary is not None:
        st.subheader("Features")
        groups = st.pills("Group", sorted(dictionary["group"].unique()), selection_mode="multi",
                          default=sorted(dictionary["group"].unique()))
        st.dataframe(dictionary[dictionary["group"].isin(groups)], hide_index=True,
                     column_config={"feature": "Feature", "group": "Group", "description": "Description"})

    metadata = {name: loaders.report_json(name) for name in ("scoring_metadata.json", "modelling_run.json")}
    if any(metadata.values()):
        with st.expander("Run metadata (model versions, dates, settings)"):
            for name, content in metadata.items():
                if content:
                    st.markdown(f"**{name}**")
                    st.json(content)

    test_months = (end.year - train_end.year) * 12 + end.month - train_end.month
    st.subheader("Ethics and limitations")
    st.markdown(f"""
- **Data collected:** GDELT Global Knowledge Graph records (outlet, link, headline and GDELT tone) and daily share
  prices from Yahoo Finance through the yfinance library. Article text is not collected or stored, and nothing about
  private individuals is analysed.
- **SENS announcements** are not used in the models. Automated access to the JSE client portal was blocked, and a
  partial sample saved by hand would favour the most recent announcements.
- **Coverage:** only headlines from South African outlets that GDELT indexes, so quiet sessions and smaller outlets
  are missed. Ten large companies were studied; results may differ for smaller shares.
- **Scope:** headline text only; no trading costs; a {test_months}-month test period.
- **Use:** this is a research prototype. Nothing in it is investment advice.
""")
