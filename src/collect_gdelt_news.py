import logging
import os

import pandas as pd
import pydata_google_auth
from dotenv import load_dotenv
from google.cloud import bigquery

from common import ROOT, append_log, load_settings, utc_now

QUERY_PATH = ROOT / "sql" / "02_sa_news_articles.sql"
OUTPUT_PATH = ROOT / "data" / "raw" / "gdelt_sa_news.parquet"
COVERAGE_PATH = ROOT / "logs" / "gdelt_coverage.csv"
SCOPES = ["https://www.googleapis.com/auth/bigquery"]

log = logging.getLogger("collect_gdelt_news")


def job_config(settings: dict, *, dry_run: bool = False, max_bytes: int | None = None) -> bigquery.QueryJobConfig:
    companies = settings["companies"]
    params = [
        bigquery.ScalarQueryParameter("start_date", "STRING", settings["study_period"]["start"]),
        bigquery.ScalarQueryParameter("end_date", "STRING", settings["study_period"]["end"]),
        bigquery.ArrayQueryParameter("domains", "STRING", settings["gdelt"]["domains"]),
        bigquery.ArrayQueryParameter("tickers", "STRING", [c["ticker"] for c in companies]),
        bigquery.ArrayQueryParameter("patterns", "STRING", [c["pattern"] for c in companies]),
    ]
    # A cached result reports 0 bytes, so the dry run must bypass the cache to give a real estimate
    config = bigquery.QueryJobConfig(query_parameters=params, dry_run=dry_run, use_query_cache=not dry_run)
    # Assigning None sends the literal string "None" to the API, so only set a real limit
    if max_bytes is not None:
        config.maximum_bytes_billed = max_bytes
    return config


def coverage(articles: pd.DataFrame, settings: dict) -> pd.DataFrame:
    names = {c["ticker"]: c["name"] for c in settings["companies"]}
    summary = (
        articles.assign(day=articles["published_at"].dt.date)
        .groupby("ticker")
        .agg(articles=("url", "nunique"), headline_mentions=("in_title", "sum"), days_with_news=("day", "nunique"))
        .reindex(list(names), fill_value=0)
    )
    summary.insert(0, "company", [names[t] for t in summary.index])
    return summary.sort_values("articles", ascending=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv(ROOT / ".env")
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise SystemExit("Set GCP_PROJECT_ID in .env to your personal Google Cloud project ID")

    settings = load_settings()
    sql = QUERY_PATH.read_text(encoding="utf-8")
    max_gb = settings["gdelt"]["max_gb"]

    credentials = pydata_google_auth.get_user_credentials(SCOPES)
    client = bigquery.Client(project=project_id, credentials=credentials)

    estimate_gb = client.query(sql, job_config=job_config(settings, dry_run=True)).total_bytes_processed / 1e9
    log.info("Query will scan %.1f GB (limit %s GB)", estimate_gb, max_gb)
    if estimate_gb > max_gb:
        raise SystemExit(f"Estimated scan of {estimate_gb:.1f} GB is over gdelt.max_gb; query not run")

    run_at = utc_now()
    try:
        job = client.query(sql, job_config=job_config(settings, max_bytes=int(max_gb * 1e9)))
        raw = job.to_dataframe(create_bqstorage_client=False)
    except Exception as exc:
        append_log([{"run_at": run_at, "source": "gdelt_gkg", "ticker": "ALL", "rows": 0,
                     "status": "error", "message": str(exc)[:300]}])
        raise

    articles = (
        raw.sort_values("published_at")
        .drop_duplicates(subset=["url", "ticker"])
        .reset_index(drop=True)
    )
    duplicates = len(raw) - len(articles)
    log.info("Fetched %d rows, %d after removing %d duplicate url/ticker pairs", len(raw), len(articles), duplicates)

    if articles.empty:
        append_log([{"run_at": run_at, "source": "gdelt_gkg", "ticker": "ALL", "rows": 0,
                     "status": "empty", "message": "query returned no matching articles"}])
        raise SystemExit("No matching articles returned; check domains and patterns in settings.yaml")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    articles.to_parquet(OUTPUT_PATH, index=False)

    append_log([{
        "run_at": run_at, "source": "gdelt_gkg", "ticker": "ALL", "rows": len(articles),
        "first_date": articles["published_at"].min().date(), "last_date": articles["published_at"].max().date(),
        "status": "ok", "message": f"{job.total_bytes_processed / 1e9:.1f} GB scanned; {duplicates} duplicates removed",
    }])

    summary = coverage(articles, settings)
    summary.to_csv(COVERAGE_PATH)
    log.info("Saved %d articles to %s", len(articles), OUTPUT_PATH.relative_to(ROOT))
    print()
    print(summary.to_string())


if __name__ == "__main__":
    main()