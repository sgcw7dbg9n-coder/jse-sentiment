import html
import logging
import re
from datetime import time

import pandas as pd

from common import ROOT, is_sa_domain, load_settings

PRICES_PATH = ROOT / "data" / "raw" / "prices.csv"
OUTPUT_PATH = ROOT / "data" / "interim" / "news_clean.parquet"
SUMMARY_PATH = ROOT / "reports" / "news_cleaning_summary.csv"

log = logging.getLogger("clean_news")


def normalise_title(title: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(title)).strip()


def trading_calendar(proxy_ticker: str) -> pd.DatetimeIndex:
    prices = pd.read_csv(PRICES_PATH, usecols=["ticker", "date"], parse_dates=["date"])
    dates = prices.loc[prices["ticker"] == proxy_ticker, "date"].drop_duplicates().sort_values()
    if dates.empty:
        raise SystemExit(f"No prices for {proxy_ticker}; run collect_prices.py first")
    return pd.DatetimeIndex(dates).astype("datetime64[ns]")


def assign_trading_date(published_at: pd.Series, calendar: pd.DatetimeIndex, tz: str, close: time) -> pd.Series:
    local = published_at.dt.tz_convert(tz)
    # Text published at or after the close can only move the next session's price.
    # GDELT stamps an article when it processes it, slightly after publication, so this errs late, never early.
    after_close = (local.dt.time >= close).astype(int)
    candidate = (local.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(after_close, unit="D")).astype("datetime64[ns]")

    # The first session on or after the candidate day covers weekends and public holidays
    positions = calendar.searchsorted(candidate.to_numpy())
    in_range = positions < len(calendar)
    trading_date = pd.Series(pd.NaT, index=published_at.index, dtype="datetime64[ns]")
    trading_date[in_range] = calendar.to_numpy()[positions[in_range]]
    return trading_date


def exclusion_mask(news: pd.DataFrame, companies: dict) -> pd.Series:
    mask = pd.Series(False, index=news.index)
    for ticker, company in companies.items():
        pattern = company.get("exclude")
        if pattern:
            mask |= (news["ticker"] == ticker) & news["title"].str.contains(pattern, case=False, regex=True)
    return mask


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    companies = {c["ticker"]: c for c in settings["companies"]}
    market = settings["market"]
    study_start = pd.Timestamp(settings["study_period"]["start"])
    study_end = pd.Timestamp(settings["study_period"]["end"])

    news = pd.read_parquet(ROOT / settings["gdelt"]["output"])
    steps = [("Raw article/ticker rows from GDELT", len(news))]

    news = news[news["ticker"].isin(companies)].copy()
    steps.append(("Rows for the 10 selected companies", len(news)))

    news = news[is_sa_domain(news["domain"], settings)].copy()
    steps.append(("Published by a South African outlet", len(news)))

    news = news[news["in_title"]].copy()
    steps.append(("Company named in the headline", len(news)))

    news["title"] = news["title"].map(normalise_title)
    news = news[~exclusion_mask(news, companies)]
    steps.append(("After removing sponsorships, indices and economist commentary", len(news)))

    calendar = trading_calendar(settings["market_proxy"]["ticker"])
    news["trading_date"] = assign_trading_date(
        news["published_at"], calendar, market["timezone"], time.fromisoformat(market["close_time"])
    )
    news = news[news["trading_date"].between(study_start, study_end)]
    steps.append(("Trading date inside the study period", len(news)))

    news["title_key"] = news["title"].str.lower()
    clean = (
        news.sort_values("published_at")
        .groupby(["ticker", "trading_date", "title_key"], as_index=False)
        .agg(
            published_at=("published_at", "first"),
            title=("title", "first"),
            domain=("domain", "first"),
            url=("url", "first"),
            gdelt_tone=("gdelt_tone", "mean"),
            outlets=("domain", "nunique"),
        )
        .drop(columns="title_key")
        .sort_values(["ticker", "published_at"])
        .reset_index(drop=True)
    )
    steps.append(("After merging the same headline from several outlets", len(clean)))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(OUTPUT_PATH, index=False)

    summary = pd.DataFrame(steps, columns=["step", "rows"])
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    per_company = (
        clean.groupby("ticker")
        .agg(headlines=("title", "size"), days_with_news=("trading_date", "nunique"))
        .reindex(list(companies), fill_value=0)
    )
    per_company.insert(0, "company", [companies[t]["name"] for t in per_company.index])
    sessions = calendar[(calendar >= study_start) & (calendar <= study_end)].size
    per_company["share_of_sessions"] = (per_company["days_with_news"] / sessions).map("{:.1%}".format)

    log.info("Saved %d headlines to %s", len(clean), OUTPUT_PATH.relative_to(ROOT))
    print()
    print(summary.to_string(index=False))
    print(f"\nTrading sessions in study period: {sessions}\n")
    print(per_company.sort_values("headlines", ascending=False).to_string())


if __name__ == "__main__":
    main()