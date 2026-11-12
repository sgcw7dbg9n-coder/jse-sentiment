import logging

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from common import ROOT, load_settings

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
PREDICTION_FILES = {
    "all_days": "test_predictions_sensitivity.parquet",
    "news_days_only": "test_predictions_newsdays.parquet",
}
# Five sessions covers the longest target horizon, so overlapping targets stay inside one block
BLOCK_SESSIONS = 5
REPLICATES = 2000

log = logging.getLogger("robustness")


def block_bootstrap(paired: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Moving-block bootstrap over trading days for the AUC and accuracy gap (enhanced minus baseline)."""
    dates = np.sort(paired["date"].unique())
    date_values = paired["date"].to_numpy()
    rows_by_date = [np.flatnonzero(date_values == d) for d in dates]
    y = paired["y_true"].to_numpy()
    score_base, score_enh = paired["score_base"].to_numpy(), paired["score_enh"].to_numpy()
    hit_base, hit_enh = paired["pred_base"].to_numpy() == y, paired["pred_enh"].to_numpy() == y

    n_blocks = int(np.ceil(len(dates) / BLOCK_SESSIONS))
    auc_gaps, acc_gaps = [], []
    for _ in range(REPLICATES):
        starts = rng.integers(0, len(dates) - BLOCK_SESSIONS + 1, n_blocks)
        idx = np.concatenate([rows_by_date[s + k] for s in starts for k in range(BLOCK_SESSIONS)])
        if y[idx].min() == y[idx].max():
            continue
        auc_gaps.append(roc_auc_score(y[idx], score_enh[idx]) - roc_auc_score(y[idx], score_base[idx]))
        acc_gaps.append(hit_enh[idx].mean() - hit_base[idx].mean())

    auc_gaps, acc_gaps = np.array(auc_gaps), np.array(acc_gaps)
    return {
        "auc_gap": roc_auc_score(y, score_enh) - roc_auc_score(y, score_base),
        "auc_ci_low": np.percentile(auc_gaps, 2.5), "auc_ci_high": np.percentile(auc_gaps, 97.5),
        "auc_p_boot": min(1.0, 2 * min(np.mean(auc_gaps <= 0), np.mean(auc_gaps >= 0))),
        "acc_gap": hit_enh.mean() - hit_base.mean(),
        "acc_ci_low": np.percentile(acc_gaps, 2.5), "acc_ci_high": np.percentile(acc_gaps, 97.5),
    }


def bootstrap_comparisons(predictions: pd.DataFrame, sample: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    key = ["date", "ticker", "y_true"]
    rows = []
    for (target, model), group in predictions.groupby(["target", "model"]):
        base = group[group["feature_set"] == "baseline"][key + ["score", "pred"]]
        for feature_set in sorted(set(group["feature_set"]) - {"baseline"}):
            enhanced = group[group["feature_set"] == feature_set][key + ["score", "pred"]]
            paired = base.merge(enhanced, on=key, suffixes=("_base", "_enh"))
            rows.append({"sample": sample, "target": target, "model": model, "feature_set": feature_set,
                         "test_rows": len(paired), **block_bootstrap(paired, rng)})
            log.info("Bootstrapped %s | %s | %s | %s", sample, target, model, feature_set)
    frame = pd.DataFrame(rows)
    frame["auc_p_boot_holm"] = frame.groupby("target")["auc_p_boot"].transform(lambda p: multipletests(p, method="holm")[1])
    return frame


def panel_regressions(data: pd.DataFrame, scores: list[str]) -> pd.DataFrame:
    news = data[data["has_news"] == 1].dropna(subset=["ret", "fwd_ret_1d", "mkt_ret"]).copy()
    news[["ret_pct", "fwd_pct", "mkt_pct"]] = news[["ret", "fwd_ret_1d", "mkt_ret"]] * 100
    # Clustering by date allows for all ten companies reacting to the same market news on a given day
    groups = pd.factorize(news["date"])[0]
    formulas = {
        "same_day": "ret_pct ~ sentiment_z + mkt_pct + C(ticker)",
        "next_day": "fwd_pct ~ sentiment_z + ret_pct + mkt_pct + C(ticker)",
    }

    rows = []
    for score in scores:
        column = news[f"{score}_mean"]
        news["sentiment_z"] = (column - column.mean()) / column.std()
        for horizon, formula in formulas.items():
            fit = smf.ols(formula, data=news).fit(cov_type="cluster", cov_kwds={"groups": groups})
            rows.append({"score": score, "horizon": horizon, "coef_pct_per_sd": fit.params["sentiment_z"],
                         "std_err": fit.bse["sentiment_z"], "t": fit.tvalues["sentiment_z"],
                         "p": fit.pvalues["sentiment_z"], "n": int(fit.nobs), "r2": fit.rsquared})
    frame = pd.DataFrame(rows)
    frame["p_holm"] = frame.groupby("horizon")["p"].transform(lambda p: multipletests(p, method="holm")[1])
    return frame


def sentiment_terciles(data: pd.DataFrame, score: str) -> pd.DataFrame:
    news = data[data["has_news"] == 1]
    # Ranking first avoids duplicate bin edges, because many headlines score exactly 0
    bucket = pd.qcut(news[f"{score}_mean"].rank(method="first"), 3, labels=["negative", "neutral", "positive"])
    summary = news.groupby(bucket, observed=True).agg(
        company_days=("ret", "size"),
        mean_sentiment=(f"{score}_mean", "mean"),
        same_day_ret_pct=("ret", lambda r: r.mean() * 100),
        next_day_ret_pct=("fwd_ret_1d", lambda r: r.mean() * 100),
        next_day_up_share=("target_up_1d", "mean"),
    )
    summary.loc["positive minus negative"] = summary.loc["positive"] - summary.loc["negative"]
    summary.loc["positive minus negative", "company_days"] = np.nan
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    seed = settings["modelling"]["random_state"]
    scores = settings["features"]["scores"]
    REPORTS.mkdir(parents=True, exist_ok=True)

    frames = []
    for sample, filename in PREDICTION_FILES.items():
        path = PROCESSED / filename
        if path.exists():
            frames.append(bootstrap_comparisons(pd.read_parquet(path), sample, seed))
        else:
            log.warning("%s not found; skipping the %s sample", filename, sample)
    if frames:
        bootstrap = pd.concat(frames, ignore_index=True)
        bootstrap.to_csv(REPORTS / "bootstrap_comparisons.csv", index=False)
        view = bootstrap[bootstrap["feature_set"] == "all_sentiment"]
        print(f"\nAll-sentiment minus baseline, block bootstrap over trading days ({REPLICATES} replicates):")
        print(view[["sample", "target", "model", "auc_gap", "auc_ci_low", "auc_ci_high", "auc_p_boot_holm"]]
              .round(3).to_string(index=False))

    data = pd.read_parquet(PROCESSED / "model_dataset.parquet")
    regressions = panel_regressions(data, scores)
    regressions.to_csv(REPORTS / "panel_regressions.csv", index=False)
    print("\nReturn (%) per 1 SD of sentiment, news days, company fixed effects, SEs clustered by date:")
    print(regressions[["horizon", "score", "coef_pct_per_sd", "std_err", "p", "p_holm", "n"]].round(4).to_string(index=False))

    terciles = sentiment_terciles(data, "finbert_score")
    terciles.to_csv(REPORTS / "finbert_terciles.csv")
    print("\nNews days grouped by FinBERT sentiment:")
    print(terciles.round(3).to_string())


if __name__ == "__main__":
    main()
