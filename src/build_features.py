import logging

import numpy as np
import pandas as pd

from common import ROOT, load_settings

PRICES_PATH = ROOT / "data" / "raw" / "prices.csv"
NEWS_PATH = ROOT / "data" / "interim" / "news_scored.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "model_dataset.parquet"
DICTIONARY_PATH = ROOT / "reports" / "feature_dictionary.csv"

log = logging.getLogger("build_features")


def rolling(series_by_ticker, window: int, how: str) -> pd.Series:
    return series_by_ticker.transform(lambda s: getattr(s.rolling(window, min_periods=window), how)())


def price_features(prices: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[tuple]]:
    prices = prices.sort_values(["ticker", "date"]).copy()
    by = prices.groupby("ticker", group_keys=False)
    docs = []

    prices["ret"] = prices["close"] / by["close"].shift(1) - 1
    docs.append(("ret", "market", "Return on day t, close t-1 to close t"))
    by = prices.groupby("ticker", group_keys=False)
    for lag in range(1, cfg["return_lags"] + 1):
        prices[f"ret_lag{lag}"] = by["ret"].shift(lag)
        docs.append((f"ret_lag{lag}", "market", f"Return {lag} session(s) before day t"))

    for window in cfg["volatility_windows"]:
        prices[f"vol_{window}d"] = rolling(by["ret"], window, "std")
        docs.append((f"vol_{window}d", "market", f"Standard deviation of daily returns over the last {window} sessions"))

    average_volume = rolling(by["volume"], cfg["volume_window"], "mean")
    prices["volume_ratio"] = (prices["volume"] / average_volume).replace([np.inf, -np.inf], np.nan)
    prices["log_volume"] = np.log1p(prices["volume"])
    prices["hl_range"] = (prices["high"] - prices["low"]) / prices["close"]
    docs += [
        ("volume_ratio", "market", f"Volume on day t divided by its {cfg['volume_window']}-session average"),
        ("log_volume", "market", "Natural log of (1 + shares traded on day t)"),
        ("hl_range", "market", "Intraday high minus low, as a share of the close"),
    ]

    for h in cfg["horizons"]:
        prices[f"fwd_ret_{h}d"] = by["close"].shift(-h) / prices["close"] - 1
    return prices, docs


def news_features(news: pd.DataFrame, panel: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[tuple]]:
    scores = cfg["scores"]
    label = news[["finbert_positive", "finbert_negative", "finbert_neutral"]].idxmax(axis=1)
    news = news.assign(finbert_is_pos=label.eq("finbert_positive"), finbert_is_neg=label.eq("finbert_negative"))

    daily = news.groupby(["ticker", "trading_date"]).agg(
        news_count=("title", "size"),
        news_outlets=("outlets", "sum"),
        finbert_pos_count=("finbert_is_pos", "sum"),
        finbert_neg_count=("finbert_is_neg", "sum"),
        **{f"{score}_sum": (score, "sum") for score in scores},
    ).reset_index().rename(columns={"trading_date": "date"})

    panel = panel.merge(daily, on=["ticker", "date"], how="left")
    count_columns = ["news_count", "news_outlets", "finbert_pos_count", "finbert_neg_count"] + [f"{s}_sum" for s in scores]
    panel[count_columns] = panel[count_columns].fillna(0)

    # Days without news get a neutral 0 plus the has_news flag, so the model can tell "neutral" from "silent"
    has_news = panel["news_count"] > 0
    panel["has_news"] = has_news.astype(int)
    docs = [
        ("has_news", "news", "1 if at least one headline was assigned to day t"),
        ("news_count", "news", "Headlines assigned to day t (after 17:00 rule)"),
        ("news_outlets", "news", "Outlets carrying those headlines, counting syndicated copies"),
        ("finbert_pos_share", "news", "Share of day t headlines FinBERT labels positive"),
        ("finbert_neg_share", "news", "Share of day t headlines FinBERT labels negative"),
    ]
    safe_count = panel["news_count"].where(has_news)
    panel["finbert_pos_share"] = (panel["finbert_pos_count"] / safe_count).fillna(0)
    panel["finbert_neg_share"] = (panel["finbert_neg_count"] / safe_count).fillna(0)
    for score in scores:
        panel[f"{score}_mean"] = (panel[f"{score}_sum"] / safe_count).fillna(0)
        docs.append((f"{score}_mean", "news", f"Average {score} of day t headlines (0 if none)"))

    by = panel.groupby("ticker", group_keys=False)
    for window in cfg["rolling_windows"]:
        count = by["news_count"].transform(lambda s: s.rolling(window, min_periods=1).sum())
        panel[f"news_count_{window}d"] = count
        docs.append((f"news_count_{window}d", "news", f"Headlines over the last {window} sessions including day t"))
        for score in scores:
            total = by[f"{score}_sum"].transform(lambda s: s.rolling(window, min_periods=1).sum())
            # Weighting by headline count stops one quiet day with a single headline dominating the window
            panel[f"{score}_mean_{window}d"] = (total / count.where(count > 0)).fillna(0)
            docs.append((f"{score}_mean_{window}d", "news",
                         f"Headline-weighted average {score} over the last {window} sessions (0 if none)"))

    return panel.drop(columns=["finbert_pos_count", "finbert_neg_count"] + [f"{s}_sum" for s in scores]), docs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    cfg = settings["features"]
    tickers = [c["ticker"] for c in settings["companies"]]
    proxy = settings["market_proxy"]["ticker"]
    study_start = pd.Timestamp(settings["study_period"]["start"])
    study_end = pd.Timestamp(settings["study_period"]["end"])

    prices = pd.read_csv(PRICES_PATH, parse_dates=["date"])
    prices = prices[prices["ticker"].isin(tickers + [proxy])]
    prices, docs = price_features(prices, cfg)

    market = prices.loc[prices["ticker"] == proxy, ["date", "ret", "ret_lag1", "vol_20d"]
                        + [f"fwd_ret_{h}d" for h in cfg["horizons"]]]
    market = market.rename(columns=lambda c: c if c == "date" else f"mkt_{c}")
    panel = prices[prices["ticker"].isin(tickers)].merge(market, on="date", how="left")
    panel["excess_ret"] = panel["ret"] - panel["mkt_ret"]
    docs += [
        ("mkt_ret", "market", "Satrix 40 return on day t (market proxy)"),
        ("mkt_ret_lag1", "market", "Satrix 40 return on the session before day t"),
        ("mkt_vol_20d", "market", "Satrix 40 20-session return volatility"),
        ("excess_ret", "market", "Company return minus Satrix 40 return on day t"),
    ]

    for h in cfg["horizons"]:
        fwd, excess = panel[f"fwd_ret_{h}d"], panel[f"fwd_ret_{h}d"] - panel[f"mkt_fwd_ret_{h}d"]
        panel[f"target_up_{h}d"] = (fwd > 0).astype(float).where(fwd.notna())
        panel[f"target_excess_up_{h}d"] = (excess > 0).astype(float).where(excess.notna())
        docs += [
            (f"target_up_{h}d", "target", f"1 if the close {h} session(s) after day t is above the close on day t"),
            (f"target_excess_up_{h}d", "target", f"1 if the {h}-session return beats the Satrix 40 over the same sessions"),
        ]
    panel = panel.drop(columns=[c for c in panel.columns if c.startswith("mkt_fwd_ret")])

    news = pd.read_parquet(NEWS_PATH)
    missing_scores = [score for score in cfg["scores"] if score not in news.columns]
    if missing_scores:
        raise SystemExit(f"{NEWS_PATH.name} has no {', '.join(missing_scores)} column; run score_news.py first "
                         "or remove it from features.scores")
    panel, news_docs = news_features(news, panel, cfg)
    docs += news_docs

    # Rolling windows were computed on the full price history, so trimming now keeps the warm-up out of the data
    panel = panel[panel["date"].between(study_start, study_end)].sort_values(["date", "ticker"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT_PATH, index=False)
    DICTIONARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(docs, columns=["feature", "group", "description"]).to_csv(DICTIONARY_PATH, index=False)

    log.info("Saved %d rows (%d companies x %d sessions) to %s", len(panel), panel["ticker"].nunique(),
             panel["date"].nunique(), OUTPUT_PATH.relative_to(ROOT))
    log.info("Wrote %d feature descriptions to %s", len(docs), DICTIONARY_PATH.relative_to(ROOT))

    feature_columns = [d[0] for d in docs if d[1] != "target"]
    missing = panel[feature_columns].isna().sum()
    print(f"\nSessions with news: {panel['has_news'].mean():.1%} of company-days")
    print(f"Feature cells missing: {int(missing.sum())}" + (f" ({', '.join(missing[missing > 0].index)})" if missing.any() else ""))

    print("\nClass balance (share of 1s):")
    print(panel[[c for c in panel.columns if c.startswith("target_")]].mean().map("{:.1%}".format).to_string())

    with_news = panel[panel["has_news"] == 1]
    correlations = pd.DataFrame({
        score: {
            "same_day_ret": with_news[f"{score}_mean"].corr(with_news["ret"], method="spearman"),
            "next_day_ret": with_news[f"{score}_mean"].corr(with_news["fwd_ret_1d"], method="spearman"),
        }
        for score in cfg["scores"]
    }).T
    print(f"\nSpearman correlation of day-t sentiment with returns ({len(with_news)} company-days with news):")
    print(correlations.round(3).to_string())


if __name__ == "__main__":
    main()