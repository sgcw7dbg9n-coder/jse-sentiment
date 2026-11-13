import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score, roc_curve

import charts
import loaders

ALPHA = 0.05
SAMPLES = {"all_days": "All trading days", "news_days": "News days only"}


def dot_marker(keys: list[str], size: int = 12) -> dict:
    return dict(size=size, color=[charts.colour(k) for k in keys], line=dict(color=charts.surface(), width=2))


def auc_chart(scored: pd.DataFrame, sets: list[str]) -> go.Figure:
    labels = [charts.FEATURE_SETS[s] for s in sets]
    fig = make_subplots(rows=1, cols=len(charts.MODELS), shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=list(charts.MODELS.values()))
    for col, model in enumerate(charts.MODELS, start=1):
        rows = scored[scored["model"] == model].set_index("feature_set").reindex(sets)
        fig.add_trace(go.Scatter(
            x=rows["roc_auc"], y=labels, mode="markers", marker=dot_marker(sets), name=charts.MODELS[model],
            customdata=rows[["accuracy", "cv_auc"]].to_numpy(),
            hovertemplate="<b>AUC %{x:.3f}</b><br>%{y}<br>accuracy %{customdata[0]:.3f} · "
                          "cross-validation AUC %{customdata[1]:.3f}<extra></extra>",
        ), row=1, col=col)
        charts.reference_line(fig, "x", 0.5, row=1, col=col)
    spread = max(0.03, (scored["roc_auc"] - 0.5).abs().max() + 0.01)
    fig.update_xaxes(range=[0.5 - spread, 0.5 + spread], tickformat=".2f", matches="x")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(title_text="Test ROC-AUC by model and feature set (0.5 is guessing)")
    return charts.style(fig, height=110 + 44 * len(sets), legend=False)


def comparison_table(tests: pd.DataFrame, boot: pd.DataFrame | None) -> pd.DataFrame:
    table = tests.copy()
    if boot is not None and not boot.empty:
        table = table.merge(boot[["target", "model", "feature_set", "auc_ci_low", "auc_ci_high", "auc_p_boot_holm"]],
                            on=["target", "model", "feature_set"], how="left")
    else:
        # Without the bootstrap, fall back to the normal-approximation interval implied by the DeLong z statistic
        se = (table["auc_diff"] / table["delong_z"].replace(0, np.nan)).abs()
        table["auc_ci_low"], table["auc_ci_high"] = table["auc_diff"] - 1.96 * se, table["auc_diff"] + 1.96 * se
        table["auc_p_boot_holm"] = np.nan
    return table


def gap_chart(table: pd.DataFrame, sets: list[str], interval: str) -> go.Figure:
    labels = {s: charts.FEATURE_SETS[s] for s in sets}
    fig = make_subplots(rows=1, cols=len(charts.MODELS), shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=list(charts.MODELS.values()))
    for col, model in enumerate(charts.MODELS, start=1):
        for row in table[table["model"] == model].itertuples():
            fig.add_trace(go.Scatter(
                x=[row.auc_diff], y=[labels[row.feature_set]], mode="markers", marker=dot_marker([row.feature_set]),
                error_x=dict(type="data", symmetric=False, array=[row.auc_ci_high - row.auc_diff],
                             arrayminus=[row.auc_diff - row.auc_ci_low], thickness=2, width=0,
                             color=charts.colour(row.feature_set)),
                hovertemplate=(f"<b>%{{x:+.3f}}</b> [{row.auc_ci_low:+.3f}, {row.auc_ci_high:+.3f}]<br>%{{y}}"
                               f"<br>DeLong p {row.delong_p:.3f}, Holm {row.delong_p_holm:.3f}<extra></extra>"),
                showlegend=False,
            ), row=1, col=col)
            if row.delong_p_holm < ALPHA:
                fig.add_annotation(x=row.auc_ci_high, y=labels[row.feature_set], text=f"Holm p {row.delong_p_holm:.3f}",
                                   showarrow=False, xanchor="left", xshift=6, row=1, col=col)
        charts.reference_line(fig, "x", 0, row=1, col=col)
    fig.update_yaxes(autorange="reversed", showgrid=False, categoryorder="array", categoryarray=list(labels.values()))
    fig.update_xaxes(tickformat="+.2f", matches="x")
    fig.update_layout(title_text=f"Change in test AUC from adding sentiment, with {interval}")
    return charts.style(fig, height=110 + 44 * len(sets), legend=False)


def roc_chart(base: pd.DataFrame, enhanced: pd.DataFrame, feature_set: str) -> go.Figure:
    fig = go.Figure()
    charts.diagonal(fig)
    for frame, key in ((base, "baseline"), (enhanced, feature_set)):
        fpr, tpr, _ = roc_curve(frame["y_true"], frame["score"])
        auc = roc_auc_score(frame["y_true"], frame["score"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{charts.FEATURE_SETS[key]} (AUC {auc:.3f})",
                                 line=dict(color=charts.colour(key), width=2),
                                 hovertemplate="false positive rate %{x:.2f}<br>true positive rate %{y:.2f}"))
    fig.update_xaxes(title_text="False positive rate", range=[0, 1], constrain="domain")
    fig.update_yaxes(title_text="True positive rate", range=[0, 1], scaleanchor="x")
    fig.update_layout(title_text="ROC curve on the test period")
    return charts.style(fig, height=460)


def confusion(result: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"Predicted down": [result["tn"], result["fn"]], "Predicted up": [result["fp"], result["tp"]]},
                        index=["Actual down", "Actual up"]).astype(int)


def render() -> None:
    cfg = loaders.settings()
    main = cfg["modelling"]["main_target"]

    st.title("Model comparison")
    st.caption("Every model is tuned on 2024–2025 with expanding, date-based cross-validation and evaluated once on "
               "the 2026 test period. Each sentiment-enhanced model is compared with the same model on prices only.")

    row = st.columns([2, 3, 2, 2])
    with row[0]:
        sample = st.selectbox("Sample", list(SAMPLES), format_func=SAMPLES.get)
    flag = "--news-days-only" if sample == "news_days" else ""
    results = loaders.require(loaders.results(sample), "train_models.py", flag)
    tests = loaders.require(loaders.tests(sample), "train_models.py", flag)
    targets = list(dict.fromkeys(results["target"]))
    with row[1]:
        target = st.selectbox("Target", targets, format_func=lambda t: charts.target_label(t) + (" (primary)" if t == main else ""))
    scored = results[(results["target"] == target) & (results["model"] != "majority_class")]
    sets = [s for s in charts.FEATURE_SETS if s in set(scored["feature_set"])]
    with row[2]:
        model = st.selectbox("Model to inspect", list(charts.MODELS), format_func=charts.MODELS.get)
    with row[3]:
        compare = st.selectbox("Compare baseline with", sets[1:], format_func=charts.FEATURE_SETS.get)

    majority = results[(results["target"] == target) & (results["model"] == "majority_class")].iloc[0]
    target_tests = tests[tests["target"] == target]
    best = scored.loc[scored["roc_auc"].idxmax()]
    gains = target_tests[(target_tests["delong_p_holm"] < ALPHA) & (target_tests["auc_diff"] > 0)]

    tiles = st.columns(4)
    tiles[0].metric("Test rows", f"{int(majority['test_rows']):,}", border=True,
                    help=f"Company-days in the test period; {int(majority['train_rows']):,} used for training")
    tiles[1].metric("Majority-class accuracy", f"{majority['accuracy']:.1%}", border=True,
                    help="Accuracy of always predicting the more common class in the training data")
    tiles[2].metric("Best test AUC", f"{best['roc_auc']:.3f}", border=True,
                    help=f"{charts.MODELS[best['model']]}, {charts.FEATURE_SETS[best['feature_set']]}")
    tiles[3].metric("Significant gains", f"{len(gains)} of {len(target_tests)}", border=True,
                    help="DeLong test on test-period AUC, Holm-adjusted across the comparisons for this target")

    charts.show(auc_chart(scored, sets))

    boot = loaders.bootstrap(sample)
    boot = None if boot is None else boot[boot["target"] == target]
    table = comparison_table(target_tests, boot)
    interval = "95% block-bootstrap intervals" if boot is not None and len(boot) else "95% DeLong intervals"
    charts.show(gap_chart(table, sets[1:], interval))
    st.caption("Dots right of zero mean sentiment helped. Intervals that cross zero are consistent with no effect. "
               "Only comparisons that stay significant after the Holm adjustment are labelled.")

    st.dataframe(
        table.assign(model=table["model"].map(charts.MODELS), feature_set=table["feature_set"].map(charts.FEATURE_SETS),
                     interval=[f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(table["auc_ci_low"], table["auc_ci_high"])])
        [["model", "feature_set", "auc_baseline", "auc_enhanced", "auc_diff", "interval", "delong_p", "delong_p_holm",
          "mcnemar_p_holm", "auc_p_boot_holm"]],
        hide_index=True,
        column_config={
            "model": "Model", "feature_set": "Feature set",
            "auc_baseline": st.column_config.NumberColumn("AUC, prices only", format="%.3f"),
            "auc_enhanced": st.column_config.NumberColumn("AUC, with sentiment", format="%.3f"),
            "auc_diff": st.column_config.NumberColumn("Change", format="%+.3f"),
            "interval": st.column_config.TextColumn("95% interval", help=interval),
            "delong_p": st.column_config.NumberColumn("DeLong p", format="%.3f"),
            "delong_p_holm": st.column_config.NumberColumn("DeLong p (Holm)", format="%.3f"),
            "mcnemar_p_holm": st.column_config.NumberColumn("McNemar p (Holm)", format="%.3f",
                                                            help="Tests accuracy rather than AUC"),
            "auc_p_boot_holm": st.column_config.NumberColumn("Bootstrap p (Holm)", format="%.3f"),
        },
    )

    st.subheader(f"{charts.MODELS[model]}: prices only vs {charts.FEATURE_SETS[compare].removeprefix('+ ')}")
    left, right = st.columns([3, 2], gap="large")
    predictions = loaders.predictions(sample)
    with left:
        if predictions is None:
            st.info("ROC curves need the saved test predictions; run train_models.py again to create them.")
        else:
            chosen = predictions[(predictions["target"] == target) & (predictions["model"] == model)]
            charts.show(roc_chart(chosen[chosen["feature_set"] == "baseline"],
                                  chosen[chosen["feature_set"] == compare], compare))
    with right:
        for key in ("baseline", compare):
            result = scored[(scored["model"] == model) & (scored["feature_set"] == key)].iloc[0]
            st.markdown(f"**{charts.FEATURE_SETS[key]}**")
            st.dataframe(confusion(result))
            st.caption(f"Accuracy {result['accuracy']:.3f} · precision {result['precision']:.3f} · "
                       f"recall {result['recall']:.3f} · F1 {result['f1']:.3f} · tuned {result['best_params']}")

    with st.expander("Every result for this target"):
        st.dataframe(results[results["target"] == target].assign(
            model=lambda f: f["model"].map({**charts.MODELS, "majority_class": "Majority class (benchmark)"}),
            feature_set=lambda f: f["feature_set"].map(charts.FEATURE_SETS).fillna(f["feature_set"]),
        ), hide_index=True)
