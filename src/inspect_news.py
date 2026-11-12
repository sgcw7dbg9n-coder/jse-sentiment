import pandas as pd

from common import ROOT, load_settings

NEWS_PATH = ROOT / "data" / "raw" / "gdelt_sa_news.parquet"
LOCAL_TZ = "Africa/Johannesburg"
SAMPLES_PER_COMPANY = 5


def main() -> None:
    settings = load_settings()
    selected = [c["ticker"] for c in settings["companies"]]

    news = pd.read_parquet(NEWS_PATH)
    news = news[news["ticker"].isin(selected)].copy()
    news["local_time"] = news["published_at"].dt.tz_convert(LOCAL_TZ)
    news["title_key"] = news["title"].str.lower().str.strip()
    hour = news["local_time"].dt.hour

    print(f"Rows for the {len(selected)} selected companies: {len(news)} ({news['url'].nunique()} unique URLs)")
    print(f"Missing titles: {news['title'].isna().sum()}")
    print(f"Titles containing HTML entities: {news['title'].str.contains(r'&[#a-z0-9]+;', na=False).sum()}")
    print(f"Before 09:00 SAST: {(hour < 9).mean():.1%} | 09:00-17:00: {hour.between(9, 16).mean():.1%} "
          f"| 17:00 or later: {(hour >= 17).mean():.1%}")
    print(f"Published on weekends: {(news['local_time'].dt.dayofweek >= 5).mean():.1%}")

    outlets_per_headline = news.dropna(subset=["title_key"]).groupby(["ticker", "title_key"])["domain"].nunique()
    print(f"Headlines published by more than one outlet: {(outlets_per_headline > 1).sum()}")

    print("\nArticles per outlet:")
    print(news["domain"].value_counts().to_string())

    print(f"\nSample headlines ({SAMPLES_PER_COMPANY} per company, H = company named in headline):")
    for ticker, group in news.dropna(subset=["title"]).groupby("ticker"):
        print(f"\n{ticker}")
        for row in group.sample(min(SAMPLES_PER_COMPANY, len(group)), random_state=42).itertuples():
            flag = "H" if row.in_title else " "
            print(f"  [{flag}] {row.domain:22} {row.title[:100]}")


if __name__ == "__main__":
    main()
